#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

extern void void fused_rmsnorm_residual_forward(
        __nv_fp8_e4m3* __restrict__ vals, __nv_fp8_e8m0* __restrict__ scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream);

namespace pi {
    extern void fused_moe_bf16(
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
    );

    static void fused_moe_bf16_torch_stub(
        const torch::Tensor &x,
        const torch::Tensor &w,
        const torch::Tensor &w2,
        const torch::Tensor &sorted_token_ids,
        const torch::Tensor &expert_ids,
        const torch::Tensor &num_tokens_post_padded,
        const torch::Tensor &topk_weights,
        torch::Tensor &out,
        int64_t top_k,
        int64_t block_m,
        int64_t block_n,
        int64_t warp_n,
        int64_t stages
    ) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        at::cuda::OptionalCUDAGuard device_guard {device_of(x)};
        TORCH_CHECK(x.size(0));
        TORCH_CHECK(x.size(1) == w.size(2));
        TORCH_CHECK((x.size(1)&127) == 0, "K dimension must be a multiple of 128");
        TORCH_CHECK((block_n == 64 && warp_n == 4) || (block_n == 32 && warp_n == 8));
        TORCH_CHECK(stages > 0 && stages < 6);
        TORCH_CHECK(block_m == 128, "Blackwell tcgen05 fused MoE requires block_m=128.");
        TORCH_CHECK(x.is_contiguous());
        TORCH_CHECK(w.is_contiguous());
        TORCH_CHECK(w2.is_contiguous());
        TORCH_CHECK(sorted_token_ids.is_contiguous());
        TORCH_CHECK(expert_ids.is_contiguous());
        TORCH_CHECK(num_tokens_post_padded.is_contiguous());
        TORCH_CHECK(topk_weights.is_contiguous());
        TORCH_CHECK(out.is_contiguous());
        TORCH_CHECK(x.dtype() == torch::kBFloat16);
        TORCH_CHECK(w.dtype() == torch::kBFloat16);
        TORCH_CHECK(w2.dtype() == torch::kBFloat16);
        TORCH_CHECK(topk_weights.dtype() == torch::kFloat32);
        TORCH_CHECK(sorted_token_ids.dtype() == torch::kInt32);
        TORCH_CHECK(expert_ids.dtype() == torch::kInt32);
        TORCH_CHECK(num_tokens_post_padded.dtype() == torch::kInt32);
        TORCH_CHECK(out.dtype() == torch::kBFloat16);
        fused_moe_bf16(
            static_cast<const __nv_bfloat16 *>(x.const_data_ptr()),
            static_cast<const __nv_bfloat16 *>(w.const_data_ptr()),
            static_cast<const __nv_bfloat16 *>(w2.const_data_ptr()),
            static_cast<__nv_bfloat16 *>(out.data_ptr()),
            static_cast<const int32_t *>(sorted_token_ids.const_data_ptr()),
            static_cast<const int32_t *>(expert_ids.const_data_ptr()),
            static_cast<const int32_t *>(num_tokens_post_padded.const_data_ptr()),
            static_cast<const float *>(topk_weights.const_data_ptr()),
            top_k,
            x.size(0),
            x.size(1),
            w.size(1),
            w.size(0),
            sorted_token_ids.size(0),
            block_m,
            block_n,
            warp_n,
            stages,
            stream
        );
    }

    void fused_rmsnorm_residual_stub(
            const at::Tensor& branch,
            const at::Tensor& residual,
            const at::Tensor& weight,
            double epsilon,
            at::Tensor& vals,
            at::Tensor& scales,
            at::Tensor& rrms
    ){
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        at::cuda::OptionalCUDAGuard device_guard {device_of(branch)};

        // check inputs
        TORCH_CHECK(branch.scalar_type() == at::kBFloat16);
        TORCH_CHECK(residual.scalar_type() == at::kBFloat16);
        TORCH_CHECK(weight.scalar_type() == at::kBFloat16);
        TORCH_CHECK(branch.is_contiguous());
        TORCH_CHECK(residual.is_contiguous());
        TORCH_CHECK(weight.is_contiguous());

        const int64_t C = branch.size(-1);
        const int64_t N = branch.numel() / C;

        TORCH_CHECK(residual.sizes() == branch.sizes());
        TORCH_CHECK(weight.dim() == 1 && weight.numel() == C);
        TORCH_CHECK(C % 128 == 0);
        TORCH_CHECK(N % 128 == 0);
        TORCH_CHECK(N <= std::numeric_limits<int>::max());

        // check outputs
        TORCH_CHECK(vals.scalar_type() == at::kFloat8_e4m3fn);
        TORCH_CHECK(scales.scalar_type() == at::kFloat8_e8m0fnu);
        TORCH_CHECK(rrms.scalar_type() == at::kFloat);

        TORCH_CHECK(vals.is_contiguous());
        TORCH_CHECK(scales.is_contiguous());
        TORCH_CHECK(rrms.is_contiguous());

        TORCH_CHECK(vals.sizes() == branch.sizes());
        TORCH_CHECK(rrms.dim() == 1 && rrms.numel() == N);
        TORCH_CHECK(scales.numel() == N * C / 32);

        fused_rmsnorm_residual_forward(
                reinterpret_cast<__nv_fp8_e4m3*>(vals.data_ptr()),
                reinterpret_cast<__nv_fp8_e8m0*>(scales.data_ptr()),
                rrms.data_ptr<float>(),
                reinterpret_cast<const nv_bfloat16*>(branch.const_data_ptr()),
                reinterpret_cast<const nv_bfloat16*>(residual.const_data_ptr()),
                reinterpret_cast<const nv_bfloat16*>(weight.const_data_ptr()),
                static_cast<float>(epsilon),
                static_cast<int>(N), static_cast<int>(C), stream);
    }
}

TORCH_LIBRARY_FRAGMENT(prime_moe, m) {
    m.def("fused_moe_bf16("
        "Tensor x, "
        "Tensor w, "
        "Tensor w2, "
        "Tensor sorted_token_ids, "
        "Tensor expert_ids, "
        "Tensor num_tokens_post_padded, "
        "Tensor topk_weights, "
        "Tensor(a!) out, "
        "int top_k, "
        "int block_m, "
        "int block_n, "
        "int warp_n, "
        "int stages"
        ") -> ()"
    );
    m.def("fused_rmsnorm_residual("
          "Tensor branch, "
          "Tensor residual, "
          "Tensor weight, "
          "float epsilon, "
          "Tensor(a!) vals, "
          "Tensor(b!) scales, "
          "Tensor(c!) rrms"
          ") -> ()");

    m.impl("fused_moe_bf16", torch::kCUDA, &pi::fused_moe_bf16_torch_stub);
    m.impl("fused_rmsnorm_residual", torch::kCUDA, &pi::fused_rmsnorm_residual_stub);
}

PYBIND11_MODULE(_C, m) {}
