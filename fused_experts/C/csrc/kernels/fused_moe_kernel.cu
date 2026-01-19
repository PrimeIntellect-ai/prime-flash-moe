#include "common.cuh"

#include <stdexcept>

#include <mma.h>
#include <cuda/ptx>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace pi {
    [[nodiscard]] static PFN_cuTensorMapEncodeTiled_v12000 getProcAddress_cuTensorMapEncodeTiled() {
        static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
        if (fn) return fn;
        cudaDriverEntryPointQueryResult stat;
        void *pfn = nullptr;
        auto res = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &pfn, 12000, cudaEnableDefault, &stat); 
        if (res != cudaSuccess || stat != cudaDriverEntryPointSuccess)
            throw std::runtime_error {"Failed to get address of cuTensorMapEncodeTiled: " + std::string{cudaGetErrorString(res)}};
        fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(pfn);
        return fn;
    }

    [[nodiscard]] static __device__ __forceinline__ float swiglu(float x, float w) {
        return w*(x/(1.f+__expf(-x)));
    }

    template <const int32_t STAGES, const int32_t WN, const int32_t BM, const int32_t BK, const int32_t BN>
    struct smem_up_pod {
        alignas(1024) __nv_bfloat16 w[STAGES*WN*BK*BN];
        alignas(1024) __nv_bfloat16 x[STAGES*BK*BM];
    };

    template <const int32_t STAGES, const int32_t WN, const int32_t BM, const int32_t BK, const int32_t BN>
    struct smem_down_pod {
        alignas(1024) __nv_bfloat16 w[STAGES*WN*BK*BN];
        alignas(1024) __nv_bfloat16 x[BM*WN*BN/2];
        alignas(16) __nv_bfloat16 out[BM*(BK*2 + 8)];
    };

    template <const int32_t BM, const int32_t BK, const int32_t BN, const int32_t WN, const int32_t STAGES, const int32_t PRODUCER_THREADS>
    static __global__ __launch_bounds__(WN*32 + PRODUCER_THREADS) void fused_moe_kernel(
        const __nv_bfloat16 *__restrict__ x,
        const __grid_constant__ CUtensorMap map_w,
        const __grid_constant__ CUtensorMap map_w2,
        __nv_bfloat16 *__restrict__ out,
        const int32_t *__restrict__ sorted_token_ids,
        const int32_t *__restrict__ expert_idxs,
        const int32_t *__restrict__ num_tokens_post_padded,
        const float *__restrict__ topk_weights,
        int32_t top_k,
        int32_t M,
        int32_t K,
        int32_t N
    ) {
        static constexpr int32_t CONSUMER_THREADS = WN<<5;
        static constexpr int32_t WARP_GROUPS = WN>>2;
        static constexpr int32_t BLOCK_SHAPE[2] = {128, 128};
        static constexpr uint32_t SWIZZLE_BITS_UP = 3;
        static constexpr uint32_t SWIZZLE_MODE_UP = 4-SWIZZLE_BITS_UP;
        static constexpr uint32_t SWIZZLE_BITS_DOWN = WN*BN == 256 ? 3 : WN*BN == 128 ? 2 : 1;
        static constexpr uint32_t SWIZZLE_MODE_DOWN = 4-SWIZZLE_BITS_DOWN;
        static constexpr int32_t BK2 = (WN*BN)>>1;
        static constexpr int32_t BN2 = BK<<1;
        static constexpr int32_t WS = WN*BK*BN;             // SMEM
        static constexpr int32_t XS = BK*BM;                // SMEM
        static constexpr int32_t TB = 16;                   // CP_ASYNC granularity
        static constexpr int32_t TO = TB/sizeof(__nv_bfloat16);    // Transfer thread offset
        static constexpr int32_t TN2 = BN2/(WARP_GROUPS<<6);
        static constexpr int32_t TM = BM>>3;
        static constexpr int32_t TN = BN>>4;
        static constexpr int32_t PAD = BN2+8;
        static constexpr int32_t X_IT = const_ceil(XS/static_cast<float>(PRODUCER_THREADS*TO));

        extern __shared__ __align__(1024) uint8_t smem_raw[];
        __shared__ __align__(8) uint64_t bar[STAGES<<1];
        __shared__ float topk_scales[BM];

        auto &smem_up = *reinterpret_cast<smem_up_pod<STAGES, WN, BM, BK, BN> *>(smem_raw);

        int32_t warp_m = blockIdx.y;
        int32_t expert_idx = expert_idxs[warp_m];
        if (BM*warp_m >= *num_tokens_post_padded) return;
        int32_t N2 = K;
        int32_t lane_id = threadIdx.x&31;
        bool is_prod = threadIdx.x < PRODUCER_THREADS;
        int32_t warp_id = is_prod ? threadIdx.x : threadIdx.x-PRODUCER_THREADS;
        warp_id >>= 5;
        if (!threadIdx.x) {
            for (int32_t i=0; i < STAGES; ++i) {
                barrier::init(bar+i, PRODUCER_THREADS+1, 0);
                barrier::init(bar+i+STAGES, CONSUMER_THREADS, 0);
            }
        }
        __syncthreads();
        auto con_sync = [=]() { asm volatile ("bar.sync 0, %0;\n" :: "n"(CONSUMER_THREADS)); };
        int32_t n_stages_up = K / *BLOCK_SHAPE;
        int32_t n_stages_down = N2/BN2;
        float2 facc[TN][TM][4] = {};
        int32_t phase = 0;
        const auto producer = [&] {
            int32_t tsrc[X_IT];
            int32_t i = threadIdx.x*TO;
            for (int32_t r=0; r < X_IT; ++r) {
                if (i < XS) {
                    int32_t tdest = sorted_token_ids[warp_m*BM + r*(PRODUCER_THREADS>>3) + (threadIdx.x>>3)];
                    tsrc[r] = tdest/top_k;
                    if (!(threadIdx.x&7) && tsrc[r] < M && i < XS) {
                        uint32_t ptr = __cvta_generic_to_shared((topk_scales+((r*(PRODUCER_THREADS>>3))+(threadIdx.x>>3))));
                        cp_async::cg64<4>(ptr, topk_weights+tdest);
                    }
                }
                i += PRODUCER_THREADS*TO;
            }
            int32_t smem_stage = 0;
            for (int32_t stage=0; stage < n_stages_up; ++stage) {
                if (smem_stage == STAGES) {
                    phase ^= 1;
                    smem_stage = 0;
                }
                int32_t offs = stage**BLOCK_SHAPE;
                int32_t i = threadIdx.x*TO;
                int32_t col = offs + i%BK;
                int32_t swi = swizzle<SWIZZLE_BITS_UP>(i);
                barrier::await(bar+STAGES+smem_stage, phase);
                for (int32_t r=0; r < X_IT; ++r) {
                    int32_t row = tsrc[r];
                    if (row < M && i < XS) {
                        uint32_t sm = __cvta_generic_to_shared(smem_up.x + smem_stage*XS + swi);
                        cp_async::cg256<TB>(sm, reinterpret_cast<const uint4 *>(x + row*K + col));
                    }
                    i += PRODUCER_THREADS*TO;
                    swi += PRODUCER_THREADS*TO;
                }
                barrier::arrive_cp_async_mem(bar+smem_stage);
                if (!threadIdx.x) {
                    barrier::expect_nb(bar+smem_stage, WS*sizeof(__nv_bfloat16));
                    cp_async::load(smem_up.w + smem_stage*WS, &map_w, bar+smem_stage, offs, expert_idx*N + blockIdx.x*WN*BN);
                }
                ++smem_stage;
            }
            auto &smem_down = *reinterpret_cast<smem_down_pod<STAGES, WN, BM, BK, BN> *>(smem_raw);
            for (int32_t stage=0; stage < n_stages_down; ++stage) {
                if (smem_stage == STAGES) {
                    phase ^= 1;
                    smem_stage = 0;
                }
                barrier::await(bar+STAGES+smem_stage, phase);
                if (!threadIdx.x) {
                    barrier::expect_nb(bar+smem_stage, WS*sizeof(__nv_bfloat16));
                    cp_async::load(smem_down.w + smem_stage*WS, &map_w2, bar+smem_stage, blockIdx.x*BK2, expert_idx*N2 + stage*BN2);
                }
                barrier::arrive_cp_async_mem(bar+smem_stage);
                ++smem_stage;
            }
        };
        const auto consumer = [&] {
            int32_t tok_src = M;
            if (threadIdx.x < PRODUCER_THREADS+BM)
                tok_src = sorted_token_ids[warp_m*BM + threadIdx.x-PRODUCER_THREADS]/top_k;
            for (int32_t i=0; i < STAGES; ++i) barrier::arrive(bar+STAGES+i);
            int32_t smem_stage = 0;
            for (int32_t stage=0; stage < n_stages_up; ++stage) {
                if (smem_stage == STAGES) {
                    phase ^= 1;
                    smem_stage = 0;
                }
                barrier::await(bar+smem_stage, phase);
                float tile[TN][TM][4] = {};
                #pragma unroll
                for (int i=0; i < TN; ++i)
                    for (int j=0; j < TM; ++j)
                        for (int k=0; k < 4; ++k)
                            asm volatile ("" : "+f"(tile[i][j][k]) :: "memory");
                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                warp_group::arrive();
                for (int32_t i=0; i < TN; ++i) {
                    __nv_bfloat16 *sw = smem_up.w + smem_stage*WS + (warp_id>>2)*(BN<<2)*BK + (i<<6)*BK;
                    __nv_bfloat16 *sx = smem_up.x + smem_stage*XS;
                    uint64_t desc_w = smem_desc_encode<64, 1, SWIZZLE_MODE_UP>(sw);
                    uint64_t desc_x = smem_desc_encode<64, 1, SWIZZLE_MODE_UP>(sx);
                    wgmma::wgmma<BM>(tile[i], desc_w, desc_x);
                    wgmma::wgmma<BM>(tile[i], desc_w+1*(32>>4), desc_x+1*(32>>4));
                    wgmma::wgmma<BM>(tile[i], desc_w+2*(32>>4), desc_x+2*(32>>4));
                    wgmma::wgmma<BM>(tile[i], desc_w+3*(32>>4), desc_x+3*(32>>4));
                }
                warp_group::commit_batch();
                warp_group::await();
                barrier::arrive(bar+STAGES+smem_stage);
                for (int32_t tm=0; tm < TM; ++tm)
                for (int32_t tn=0; tn < TN; ++tn) {
                    facc[tn][tm][0].x += tile[tn][tm][0];
                    facc[tn][tm][0].y += tile[tn][tm][1];
                    facc[tn][tm][1].x += tile[tn][tm][2];
                    facc[tn][tm][1].y += tile[tn][tm][3];
                }
                ++smem_stage;
            }
            con_sync();
            auto &smem_down = *reinterpret_cast<smem_down_pod<STAGES, WN, BM, BK, BN> *>(smem_raw);
            for (int32_t tn=0; tn < TN; ++tn)
            for (int32_t tm=0; tm < TM; ++tm) {
                facc[tn][tm][0].x = swiglu(facc[tn][tm][0].x, facc[tn][tm][1].x);
                facc[tn][tm][0].y = swiglu(facc[tn][tm][0].y, facc[tn][tm][1].y);
            }
            con_sync();
            for (int32_t tm=0; tm < TM; ++tm) {
                #pragma unroll
                for (int32_t t=0; t < 2; ++t)
                for (int32_t tn=0; tn < TN; ++tn) {
                    float xi = !t ? facc[tn][tm][0].x : facc[tn][tm][0].y;
                    int32_t x_row = (tm<<3) + ((lane_id&3)<<1) + t;
                    int32_t x_col = (warp_id>>1)*(TN<<5) + (tn<<5) + ((warp_id&3)<<3) + (lane_id>>2);
                    smem_down.x[swizzle<SWIZZLE_BITS_DOWN>(x_row*BK2 + x_col)] = __float2bfloat16(xi);
                }
            }
            con_sync();
            for (int32_t stage=0; stage < n_stages_down; ++stage) {
                if (smem_stage == STAGES) {
                    phase ^= 1;
                    smem_stage = 0;
                }
                float tile[TN2][TM][4] = {};
                __nv_bfloat16 *sx = smem_down.x;
                barrier::await(bar+smem_stage, phase);
                #pragma unroll
                for (int i=0; i < TN2; ++i)
                    for (int j=0; j < TM; ++j)
                        for (int k=0; k < 4; ++k)
                            asm volatile ("" : "+f"(tile[i][j][k]) :: "memory");
                warp_group::arrive();
                for (int32_t tn2=0; tn2 < TN2; ++tn2) {
                    __nv_bfloat16 *sw = smem_down.w + smem_stage*WS + (((warp_id>>2)*TN2 + tn2)<<6)*BK2;
                    uint64_t desc_w = smem_desc_encode<BK2/2, 1, SWIZZLE_MODE_DOWN>(sw);
                    uint64_t desc_x = smem_desc_encode<BK2/2, 1, SWIZZLE_MODE_DOWN>(sx);
                    #pragma unroll
                    for (int32_t tk=0; tk < BK2>>5; ++tk) {
                        wgmma::wgmma<BM>(tile[tn2], desc_w, desc_x);
                        desc_w += 32>>4;
                        desc_x += 32>>4;
                    }
                }
                warp_group::commit_batch();
                warp_group::await();
                barrier::arrive(bar+STAGES+smem_stage);
                __nv_bfloat16 out_tile[TN2>>1][TM][8];
                for (int32_t tn2=0; tn2 < TN2; tn2 += 2)
                for (int32_t tm=0; tm < TM; ++tm)
                for (int32_t t=0; t < 8; ++t)
                    out_tile[tn2>>1][tm][t] = __float2bfloat16(tile[tn2 + (t>>2)][tm][t&3]);
                asm volatile ("cp.async.bulk.wait_group 0;\n");
                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                con_sync();
                for (int32_t tn2=0; tn2 < TN2; tn2 += 2)
                for (int32_t tm=0; tm < TM; ++tm) {
                    int32_t o_row = (tm<<3) + (lane_id&7);
                    int32_t o_col = (warp_id>>2)*(TN2<<6) + ((warp_id&3)<<4) + (lane_id&8) + (tn2<<6) + ((lane_id>>4)<<6);
                    ls_matrix::store4<true>(reinterpret_cast<uint32_t *>(out_tile[tn2>>1][tm]), __cvta_generic_to_shared(smem_down.out + o_row*PAD + o_col));
                }
                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                con_sync();
                if (threadIdx.x < PRODUCER_THREADS+BM) {
                    if (tok_src < M) {
                        cuda::ptx::cp_reduce_async_bulk(
                            cuda::ptx::space_global,
                            cuda::ptx::space_shared,
                            cuda::ptx::op_add,
                            out + tok_src*N2 + stage*BN2,
                            smem_down.out + PAD*(threadIdx.x-PRODUCER_THREADS),
                            BN2*sizeof(__nv_bfloat16)
                        );
                    }
                    cuda::ptx::cp_async_bulk_commit_group();
                }
                ++smem_stage;
            }
         };
        if (is_prod) producer();
        else consumer();
    }

    template <const int32_t BM, const int32_t BN, const int32_t WN, const int32_t STAGES>
    static void launch_fused_moe_kernel(
        const __nv_bfloat16 *x,
        __nv_bfloat16 *w,
        __nv_bfloat16 *w2,
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
        cudaStream_t stream
    ) {
        static constexpr auto BK = 128;
        static constexpr auto PRODUCER_THREADS = 128;
        static constexpr size_t SMEM = std::max(sizeof(smem_up_pod<STAGES, WN, BM, BK, BN>), sizeof(smem_down_pod<STAGES, WN, BM, BK, BN>));
        dim3 block = {(WN<<5) + PRODUCER_THREADS, 1, 1};
        dim3 grid = {
            static_cast<uint32_t>(std::ceil(static_cast<double>(N)/static_cast<double>(BN*WN))),
            static_cast<uint32_t>(std::ceil(static_cast<double>(sorted_num)/static_cast<double>(block_m))),
            1
        };
        if (cudaFuncSetAttribute(&fused_moe_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess)
            throw std::runtime_error {"Failed to set max dynamic shared memory size for fused_moe_kernel: " + std::to_string(SMEM) + " bytes"};
        auto cuTensorMapEncodeTiled = getProcAddress_cuTensorMapEncodeTiled();
        CUtensorMap map_w = {};
        {
            constexpr auto rank = 2;
            uint64_t shape[rank] = {static_cast<uint64_t>(K), static_cast<uint64_t>(N*num_experts)};
            uint64_t strides[rank] = {K*sizeof(__nv_bfloat16)};
            uint32_t box[rank] = {BK, BN*WN};
            uint32_t elem_strides[rank] = {1, 1};
            auto res = cuTensorMapEncodeTiled(
                &map_w,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT16,
                rank,
                w,
                shape,
                strides,
                box,
                elem_strides,
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );
        }
        CUtensorMap map_w2 = {};
        {
            constexpr auto BK2 = (WN*BN)>>1;
            constexpr auto BN2 = BK<<1;
            auto K2 = N>>1;
            auto N2 = K;
            constexpr auto rank = 2;
            uint64_t shape[rank] = {static_cast<uint64_t>(K2), static_cast<uint64_t>(N2*num_experts)};
            uint64_t strides[rank] = {K2*sizeof(__nv_bfloat16)};
            uint32_t box[rank] = {BK2, BN2};
            uint32_t elem_strides[rank] = {1, 1};
            auto res = cuTensorMapEncodeTiled(
                &map_w2,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT16,
                rank,
                w2,
                shape,
                strides,
                box,
                elem_strides,
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                WN*BN == 256 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B : WN*BN == 128 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );
        }
        fused_moe_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS><<<grid, block, SMEM, stream>>>(
            x,
            map_w,
            map_w2,
            out,
            sorted_token_ids,
            expert_idxs,
            num_tokens_post_padded,
            topk_weights,
            top_k,
            M,
            K,
            N
        );
    }

    template <const int32_t BM, const int32_t BN, const int32_t WN>
    static void dispatch_stages(
        int32_t stages,
        const __nv_bfloat16 *x,
        __nv_bfloat16 *w,
        __nv_bfloat16 *w2,
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
        cudaStream_t stream
    ) {
        switch (stages) {
            case 1: launch_fused_moe_kernel<BM, BN, WN, 1>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case 2: launch_fused_moe_kernel<BM, BN, WN, 2>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case 3: launch_fused_moe_kernel<BM, BN, WN, 3>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case 4: launch_fused_moe_kernel<BM, BN, WN, 4>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case 5: launch_fused_moe_kernel<BM, BN, WN, 5>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            default: throw std::runtime_error {"Invalid number of stages: " + std::to_string(stages)};
        }
    }

    template <const int32_t BM>
    static void dispatch_bn_wn(
        int32_t bn,
        int32_t wn,
        int32_t stages,
        const __nv_bfloat16 *x,
        __nv_bfloat16 *w,
        __nv_bfloat16 *w2,
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
        cudaStream_t stream
    ) {
        switch (((bn&0xff)<<8)+(wn&0xff)) {
            case (32<<8)+8: dispatch_stages<BM, 32, 8>(stages, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case (64<<8)+4: dispatch_stages<BM, 64, 4>(stages, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            default: throw std::runtime_error {"Invalid BN and WN: " + std::to_string(bn) + " and " + std::to_string(wn)};
        }
    }

    void fused_moe_bf16(
        const __nv_bfloat16 *x,
        __nv_bfloat16 *w,
        __nv_bfloat16 *w2,
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
       switch (block_m) {
            case 8: dispatch_bn_wn<8>(block_n, warp_n, stages, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
       }
    }
}