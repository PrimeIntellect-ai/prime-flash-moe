// ----------------------------
    // Portable fallback implementation
    // ----------------------------
    __global__ void fused_moe_fallback_kernel_fp32(
        const __nv_bfloat16 *__restrict__ x,   // [M, K]
        const __nv_bfloat16 *__restrict__ w1,  // [E, N, K]
        const __nv_bfloat16 *__restrict__ w2,  // [E, K, N/2]
        float *__restrict__ out_fp32,          // [M, K]
        const int32_t *__restrict__ sorted_token_ids,       // [sorted_num]
        const int32_t *__restrict__ expert_idxs,            // [sorted_num/block_m]
        const int32_t *__restrict__ num_tokens_post_padded, // [1]
        const float *__restrict__ topk_weights,             // [M*top_k]
        int32_t top_k,
        int32_t M,
        int32_t K,
        int32_t N,
        int32_t block_m
    ) {
        int32_t p = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) + static_cast<int32_t>(threadIdx.x);
        int32_t total = *num_tokens_post_padded;
        if (p >= total) return;

        int32_t tdest = sorted_token_ids[p];
        if (tdest < 0) return;

        int32_t token = tdest / top_k;
        if (token < 0 || token >= M) return;

        int32_t expert = expert_idxs[p / block_m];
        float scale = topk_weights[tdest];

        for (int32_t k_out = 0; k_out < K; ++k_out) {
            float acc = 0.0f;
            for (int32_t n2 = 0; n2 < (N >> 1); ++n2) {
                float gate = 0.0f;
                float up = 0.0f;

                const __nv_bfloat16* x_row = x + static_cast<int64_t>(token) * K;
                const __nv_bfloat16* w1_gate = w1 + (static_cast<int64_t>(expert) * N + n2) * K;
                const __nv_bfloat16* w1_up   = w1 + (static_cast<int64_t>(expert) * N + (n2 + (N >> 1))) * K;

                #pragma unroll 1
                for (int32_t kk = 0; kk < K; ++kk) {
                    float xv = __bfloat162float(x_row[kk]);
                    gate += xv * __bfloat162float(w1_gate[kk]);
                    up   += xv * __bfloat162float(w1_up[kk]);
                }

                float act = swiglu(gate, up);
                float wv = __bfloat162float(w2[(static_cast<int64_t>(expert) * K + k_out) * (N >> 1) + n2]);
                acc += act * wv;
            }

            acc *= scale;
            atomicAdd(out_fp32 + static_cast<int64_t>(token) * K + k_out, acc);
        }
    }

    __global__ void cast_fp32_to_bf16_kernel(
        const float *__restrict__ inp,
        __nv_bfloat16 *__restrict__ out,
        int32_t elems
    ) {
        int32_t i = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) + static_cast<int32_t>(threadIdx.x);
        if (i >= elems) return;
        out[i] = __float2bfloat16(inp[i]);
    }

    void fused_moe_bf16(
        const __nv_bfloat16 *x,
        const __nv_bfloat16 *w,
        const __nv_bfloat16 *w2,
        __nv_bfloat16 *out,
        const int32_t *sorted_token_ids,
        const int32_t *expert_idxs,
        const int32_t *num_tokens_post_padded,
        const float *topk_weights,
        int32_t top_k,
        int32_t M,
        int32_t K,
        int32_t N,
        int32_t num_experts,
        int32_t sorted_num,
        int32_t block_m,
        int32_t block_n,
        int32_t warp_n,
        int32_t stages,
        cudaStream_t stream
    ) {
#if PRIME_MOE_ENABLE_TCGEN05
        switch (block_m) {
            case 8:
                dispatch_bn_wn<8>(block_n, warp_n, stages, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream);
                break;
            default:
                throw std::runtime_error {"Unsupported block_m for tcgen05 path: " + std::to_string(block_m)};
        }
#else
        (void)block_n;
        (void)warp_n;
        (void)stages;
        (void)sorted_num;
        (void)num_experts;

        int32_t total = 0;
        auto err = cudaMemcpyAsync(&total, num_tokens_post_padded, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) throw std::runtime_error {"cudaMemcpyAsync(num_tokens_post_padded) failed"};
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) throw std::runtime_error {"cudaStreamSynchronize failed"};

        int threads = 64;
        int blocks = (total + threads - 1) / threads;
        // Accumulate into fp32 then cast once to bf16.
        float *tmp = nullptr;
        err = cudaMallocAsync(reinterpret_cast<void **>(&tmp), static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float), stream);
        if (err != cudaSuccess) throw std::runtime_error {"cudaMallocAsync(tmp) failed"};
        err = cudaMemsetAsync(tmp, 0, static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float), stream);
        if (err != cudaSuccess) throw std::runtime_error {"cudaMemsetAsync(tmp) failed"};

        fused_moe_fallback_kernel_fp32<<<blocks, threads, 0, stream>>>(
            x, w, w2, tmp,
            sorted_token_ids, expert_idxs, num_tokens_post_padded,
            topk_weights, top_k, M, K, N, block_m
        );
        int32_t elems = M * K;
        int cast_threads = 256;
        int cast_blocks = (elems + cast_threads - 1) / cast_threads;
        cast_fp32_to_bf16_kernel<<<cast_blocks, cast_threads, 0, stream>>>(tmp, out, elems);

        err = cudaFreeAsync(tmp, stream);
        if (err != cudaSuccess) throw std::runtime_error {"cudaFreeAsync(tmp) failed"};
#endif