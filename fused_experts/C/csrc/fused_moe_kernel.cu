#include "common.cuh"

#include <mma.h>
#include <cuda/ptx>

namespace pi {
    using bf16_t = __nv_bfloat16;

    [[nodiscard]] static __device__ __forceinline__ float swiglu(float x, float w) {
        return w*(x/(1.f+__expf(-x)));
    }

    template <const int32_t STAGES, const int32_t WN, const int32_t BM, const int32_t BK, const int32_t BN>
    struct smem_up_pod {
        alignas(1024) bf16_t w[STAGES*WN*BK*BN];
        alignas(1024) bf16_t x[STAGES*BK*BM];
    };

    template <const int32_t STAGES, const int32_t WN, const int32_t BM, const int32_t BK, const int32_t BN>
    struct smem_down_pod {
        alignas(1024) bf16_t w[STAGES*WN*BK*BN];
        alignas(1024) bf16_t x[BM*WN*BN/2];
        alignas(16) bf16_t out[BM*(BK*2 + 8)];
    };

    template <const int32_t BM, const int32_t BK, const int32_t BN, const int32_t WN, const int32_t STAGES, const int32_t PRODUCER_THREADS>
    static __global__ __launch_bounds__(WN*32 + PRODUCER_THREADS) void fused_moe_bf16(
        const bf16_t *__restrict__ x,
        const __grid_constant__ CUtensorMap map_w,
        const __grid_constant__ CUtensorMap map_w2,
        bf16_t *__restrict__ out,
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
        static constexpr int32_t TO = TB/sizeof(bf16_t);    // Transfer thread offset
        static constexpr int32_t TPT = BM>>2;               // Tokens per thread
        static constexpr int32_t TN2 = BN2/(WARP_GROUPS<<6);
        static constexpr int32_t SPSD = BN2 / *BLOCK_SHAPE;
        static constexpr int32_t TM = BM>>3;
        static constexpr int32_t TN = BN>>4;
        static constexpr int X_IT = const_ceil(XS/static_cast<float>(PRODUCER_THREADS*TO));

        extern __shared__ __align__(1024) uint8_t smem_raw[];
        __shared__ __align__(8) uint64_t bar[STAGES<<1];
        __shared__ float topk_scales[BM];

        auto &smem_up = *reinterpret_cast<smem_up_pod<STAGES, WN, BM, BK, BN> *>(smem_raw);

        int32_t warp_m = blockIdx.y;
        int32_t expert_idx = expert_idxs[warp_m];
        if (BM*warp_m >= *num_tokens_post_padded) return;
        int32_t K2 = N>>1;
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
                    barrier::expect_nb(bar+smem_stage, WS*sizeof(bf16_t));
                    cp_async::load(smem_up.w + smem_stage*WS, &map_w, bar+smem_stage, offs, expert_idx*N + blockIdx.x*WN*BN);
                }
                ++smem_stage;
            }
            auto &smem_down = *reinterpret_cast<smem_down_pod<STAGES, WN, BM, BK, BN> *>(smem_raw);
            for (int32_t stage=0; stage < n_stages_up; ++stage) {
                if (smem_stage == STAGES) {
                    phase ^= 1;
                    smem_stage = 0;
                }
                barrier::await(bar+STAGES+smem_stage, phase);
                if (!threadIdx.x) {
                    barrier::expect_nb(bar+smem_stage, WS*sizeof(bf16_t));
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
                            asm volatile ("f" : "+f"(tile[i][j][k]) :: "memory");
                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                warp_group::arrive();
                for (int32_t i=0; i < TN; ++i) {
                    bf16_t *sw = smem_up.w + smem_stage*WS + (warp_id>>2)*(BN<<2)*BK + (i<<6)*BK;
                    bf16_t *sx = smem_up.x + smem_stage*XS;
                    uint64_t desc_w = smem_desc_encode<64, 1, SWIZZLE_MODE_UP>(sw);
                    uint64_t desc_x = smem_desc_encode<64, 1, SWIZZLE_MODE_UP>(sx);
                    wgmma::wgmma<1, 1, 1, BM>(tile[i], desc_w, desc_x);
                    wgmma::wgmma<1, 1, 1, BM>(tile[i], desc_w+1*(32>>4), desc_x+1*(32>>4));
                    wgmma::wgmma<1, 1, 1, BM>(tile[i], desc_w+2*(32>>4), desc_x+2*(32>>4));
                    wgmma::wgmma<1, 1, 1, BM>(tile[i], desc_w+3*(32>>4), desc_x+3*(32>>4));
                }
                warp_group::commit_batch();
                warp_group::await();
                barrier::arrive(bar+STAGES+smem_stage);
                for (int32_t tm=0; tm < TM; ++tm)
                for (int32_t tn=0; tn < TN; ++tn) {
                    facc[tn][tm][0] += float2(tile[tn][tm][0], tile[tn][tm][1]);
                    facc[tn][tm][1] += float2(tile[tn][tm][2], tile[tn][tm][3]);
                }
                ++smem_stage;
            }
            con_sync();
            auto &smem_down = *reinterpret_cast<smem_down_pod<STAGES, WN, BM, BK, BN> *>(smem_raw);
            auto *block_max = reinterpret_cast<float4 *>(smem_down.out);
            static constexpr float EPS = 1e-10f;
            float tmax[TM] = {EPS, EPS};
            for (int32_t tn=0; tn < TN; ++tn)
            for (int32_t tm=0; tm < TM; ++tm) {
                facc[tn][tm][0].x = swiglu(facc[tn][tm][0].x, facc[tn][tm][1].x);
                facc[tn][tm][0].y = swiglu(facc[tn][tm][0].y, facc[tn][tm][1].y);
                tmax[tm] = __fmax2(tmax[tm], __fabs2(facc[tn][tm][0]));
            }
            for (int32_t tm=0; tm < TM; ++tm) {
                tmax[tm] = __fmax2(__shfl_xor_sync(~0u, tmax[tm], 16), tmax[tm]);
                tmax[tm] = __fmax2(__shfl_xor_sync(~0u, tmax[tm], 8), tmax[tm]);
                tmax[tm] = __fmax2(__shfl_xor_sync(~0u, tmax[tm], 4), tmax[tm]);
                if (lane_id < 4) {
                    auto offs = (tm<<3) + (lane_id<<1);
                    reinterpret_cast<float2 *>(block_max + offs*WARP_GROUPS)[warp_id] = tmax[tm];
                }
            }
            con_sync();
            
         };
        if (is_prod) producer();
        else consumer();
    }

}