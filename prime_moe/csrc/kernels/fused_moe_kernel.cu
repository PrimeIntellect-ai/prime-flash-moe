#include <stdexcept>
#include <array>

#include <mma.h>
#include <cuda/ptx>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

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

    [[nodiscard]] constexpr __device__ __forceinline__ int32_t const_ceil(float x) {
        return (static_cast<float>(static_cast<int32_t>(x)) == x) ? static_cast<int32_t>(x) : static_cast<int32_t>(x) + (x > 0);
    }

    template <const uint32_t bits, const uint32_t base=4, const uint32_t shift=3>
    [[nodiscard]] constexpr __device__ __forceinline__ int32_t swizzle(int32_t i) {
        if constexpr (!bits) return i;
        constexpr uint32_t mask = ((1u<<bits)-1)<<(base+shift);
        return i^((i&mask)>>shift);
    }

    template <const uint64_t sdo, const uint64_t ldo, uint64_t mode>
    [[nodiscard]] __device__ __forceinline__ uint64_t smem_desc_encode(void *p) {
        return ((static_cast<uint64_t>(__cvta_generic_to_shared(p)) & 0x3ffff)>>4) | (ldo<<16) | (sdo<<32) | (mode<<62);
    }

    namespace tcgen05 {
        [[nodiscard]] __device__ __forceinline__ uint64_t desc_enc(int addr) {
            return (static_cast<uint64_t>(addr)&0x3ffff)>>4;
        }

        /*
        __device__ __forceinline__ void mma_f16(int addr, uint64_t desc_a, uint64_t desc_b, uint64_t desc_i, int enable_input_d) {
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind.f16 [%0], %1, %2, %3m p;\n"
                "}\n"
                :
                : "r"(addr), "l"(desc_a), "l"(desc_b), "r"(desc_i), "r"(enable_input_d)
                : "memory"
            );
        }*/

        __device__ __forceinline__ void commit_mbarrier(uint64_t *mbar) {
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                :
                : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar)))
                : "memory"
            );
        }

        __device__ __forceinline__ void fence_after_thread_sync() {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        }

        __device__ __forceinline__ void await_ld() {
            asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
        }

        __device__ __forceinline__ void tmem_alloc(void *addr, int32_t cols) {
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                :
                :
                "r"(static_cast<uint32_t>(__cvta_generic_to_shared(addr))),
                "r"(cols)
            );
        }
        
        __device__ __forceinline__ void tmem_free(void *addr, int32_t cols) {
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 [%0], %1;\n"
                :
                :
                "r"(static_cast<uint32_t>(__cvta_generic_to_shared(addr))),
                "r"(cols)
            );
        }

        __device__ __forceinline__ void after_thread_sync() {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        }

        __device__ __forceinline__ void mbarrier_mma(uint64_t *bar) {
            asm volatile ("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                :
                :
                "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar)))
                : "memory"
            );
        }
    }

    namespace reg_alloc {
        template <const uint32_t regs>
        __device__ __forceinline__ void alloc() {
            asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(regs));
        }

        template <const uint32_t regs>
        __device__ __forceinline__ void dealloc() {
            asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(regs));
        }
    }

    namespace cp_async {
        __device__ __forceinline__ void load(void *dst, const void *tma_map, uint64_t *bar, int32_t row, int32_t col) {
            static_assert(sizeof(void *) == sizeof(uint64_t));
            auto ptma = reinterpret_cast<uint64_t>(tma_map);
            auto pmbar = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
            auto pdst = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
                :
                : "r"(pdst), "l"(ptma), "r"(pmbar), "r"(row), "r"(col)
                : "memory"
            );
        }

        template <const int32_t nb, typename t_dst, typename t_src>
        __device__ __forceinline__ void cg256(t_dst dst, t_src src) {
            asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" :: "r"(dst), "l"(src), "n"(nb));
        }

        template <const int32_t nb, typename t_dst, typename t_src>
        __device__ __forceinline__ void cg64(t_dst dst, t_src src) {
            asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" :: "r"(dst), "l"(src), "n"(nb));
        }

        __device__ __forceinline__ void commit_group() {
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        template <const int32_t n>
        __device__ __forceinline__ void await_group() {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
        }
    }

    namespace barrier {
        __device__ __forceinline__ void init(uint64_t *bar, int32_t threads, int32_t transactions) {
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0], %1;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(threads+transactions)
            );
        }

        __device__ __forceinline__ void arrive_cp_async_mem(uint64_t *bar) {
            asm volatile(
                "cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar)))
            );
        }

        __device__ __forceinline__ void arrive(uint64_t *bar, uint32_t num=1) {
            asm volatile(
                "mbarrier.arrive.shared.b64 _, [%0], %1;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(num)
                : "memory"
            );
        }

        __device__ __forceinline__ void expect_nb(uint64_t *bar, uint32_t nb) {
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(nb)
            );
        }

        __device__ __forceinline__ void await(uint64_t *bar, int32_t phase) {
            asm volatile(
                "{\n"
                ".reg .pred P1;\n"
                "AWAIT:\n"
                "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
                "@!P1 bra.uni AWAIT;\n"
                "}\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(phase)
            );
        }
    }

    namespace ls_matrix {
         __device__ __forceinline__ void load1(uint32_t *tile, uint32_t mat) {
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                : "=r"(tile[0]) : "r"(mat)
            );
        }

        __device__ __forceinline__ void load2(uint32_t *tile, uint32_t mat) {
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(tile[0]), "=r"(tile[1]) : "r"(mat)
            );
        }

        __device__ __forceinline__ void load4(uint32_t *tile, uint32_t mat) {
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(tile[0]), "=r"(tile[1]), "=r"(tile[2]), "=r"(tile[3]) : "r"(mat)
            );
        }

        template <const bool T>
        __device__ __forceinline__ void store1(uint32_t *tile, uint32_t mat) {
            if constexpr (T) asm volatile(
                "stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                :: "r"(mat), "r"(tile[0])
                : "memory"
            ); else asm volatile(
                "stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};\n"
                :: "r"(mat), "r"(tile[0])
                : "memory"
            );
        }

        template <const bool T>
        __device__ __forceinline__ void store2(uint32_t *tile, uint32_t mat) {
            if constexpr (T) asm volatile(
                "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                :: "r"(mat), "r"(tile[0]), "r"(tile[1])
                : "memory"
            ); else asm volatile(
                "stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                :: "r"(mat), "r"(tile[0]), "r"(tile[1])
                : "memory"
            );
        }

        template <const bool T>
        __device__ __forceinline__ void store4(uint32_t *tile, uint32_t mat) {
            if constexpr (T) asm volatile(
                "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
                : "memory"
            ); else asm volatile(
                "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
                : "memory"
            );
        }
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
            int32_t i = threadIdx.x * TO;
            #pragma unroll
            for (int32_t r = 0; r < X_IT; ++r) {
                const int32_t row_in_block = r * (PRODUCER_THREADS >> 3) + (threadIdx.x >> 3);
                const bool row_valid = row_in_block < BM;
                int32_t tdest = -1;
                int32_t tok = -1;
                if (i < XS && row_valid) {
                    tdest = sorted_token_ids[warp_m * BM + row_in_block];
                    if (tdest >= 0)
                        tok = tdest / top_k;
                }
                tsrc[r] = tok;
                if (!(threadIdx.x & 7) && i < XS && row_valid && tdest >= 0 && tok < M) {
                    uint32_t ptr = __cvta_generic_to_shared(topk_scales + row_in_block);
                    cp_async::cg64<4>(ptr, topk_weights + tdest);
                }
                i += PRODUCER_THREADS * TO;
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
                    if (row >= 0 && row < M && i < XS) {
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
            int32_t tok_src = -1;
            if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS + BM) {
                int32_t tdest = sorted_token_ids[warp_m*BM + (threadIdx.x - PRODUCER_THREADS)];
                if (tdest >= 0)
                    tok_src = tdest / top_k;
            }
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
                for (int32_t i=0; i < TN; ++i)
                    for (int32_t j=0; j < TM; ++j)
                        for (int32_t k=0; k < 4; ++k)
                            asm volatile ("" : "+f"(tile[i][j][k]) :: "memory");
                // debug vals
                #pragma unroll
                for (int32_t i=0; i < TN; ++i)
                    for (int32_t j=0; j < TM; ++j)
                        for (int32_t k=0; k < 4; ++k)
                            tile[i][j][k] = 2.5f;
                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                tcgen05::after_thread_sync();
                for (int32_t i=0; i < TN; ++i) {
                    __nv_bfloat16 *sw = smem_up.w + smem_stage*WS + (warp_id>>2)*(BN<<2)*BK + (i<<6)*BK;
                    __nv_bfloat16 *sx = smem_up.x + smem_stage*XS;
                    /*uint64_t desc_w = smem_desc_encode<64, 1, SWIZZLE_MODE_UP>(sw);
                    uint64_t desc_x = smem_desc_encode<64, 1, SWIZZLE_MODE_UP>(sx);
                    wgmma::wgmma<BM>(tile[i], desc_w, desc_x);
                    wgmma::wgmma<BM>(tile[i], desc_w+1*(32>>4), desc_x+1*(32>>4));
                    wgmma::wgmma<BM>(tile[i], desc_w+2*(32>>4), desc_x+2*(32>>4));
                    wgmma::wgmma<BM>(tile[i], desc_w+3*(32>>4), desc_x+3*(32>>4));*/
                }
                //warp_group::commit_batch();
                asm volatile("" ::: "memory");
                tcgen05::await_ld();
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
                for (int32_t i=0; i < TN2; ++i)
                    for (int32_t j=0; j < TM; ++j)
                        for (int32_t k=0; k < 4; ++k)
                            asm volatile ("" : "+f"(tile[i][j][k]) :: "memory");
                // debug vals
                #pragma unroll
                for (int32_t i=0; i < TN2; ++i)
                    for (int32_t j=0; j < TM; ++j)
                        for (int32_t k=0; k < 4; ++k)
                            tile[i][j][k] = 2.5f;
                tcgen05::after_thread_sync();
                for (int32_t tn2=0; tn2 < TN2; ++tn2) {
                    __nv_bfloat16 *sw = smem_down.w + smem_stage*WS + (((warp_id>>2)*TN2 + tn2)<<6)*BK2;
                    uint64_t desc_w = smem_desc_encode<BK2/2, 1, SWIZZLE_MODE_DOWN>(sw);
                    uint64_t desc_x = smem_desc_encode<BK2/2, 1, SWIZZLE_MODE_DOWN>(sx);
                    #pragma unroll
                    for (int32_t tk=0; tk < BK2>>5; ++tk) {
                        //wgmma::wgmma<BM>(tile[tn2], desc_w, desc_x);
                        //desc_w += 32>>4;
                        //desc_x += 32>>4;
                    }
                }
                //warp_group::commit_batch();
                asm volatile("" ::: "memory");
                tcgen05::await_ld();
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
                    if (tok_src >= 0 && tok_src < M) {
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

    template <const size_t rank>
    [[nodiscard]] static CUtensorMap init_tmap(
        const char *name,
        const void *ptr,
        const std::array<uint64_t, rank> &shape,
        const std::array<uint64_t, rank> &strides,
        const std::array<uint32_t, rank> &box,
        const std::array<uint32_t, rank> &elem_strides,
        CUtensorMapSwizzle swizzle
    ) {
        CUtensorMap map = {};
        auto *encode_fn =  getProcAddress_cuTensorMapEncodeTiled();
        CUresult res = (*encode_fn)(
            &map,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            rank,
            const_cast<void *>(ptr),
            shape.data(),
            strides.data(),
            box.data(),
            elem_strides.data(),
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (res != CUDA_SUCCESS) {
            /*const char *error_name = nullptr;
            cuGetErrorName(res, &error_name);
            error_name = error_name && *error_name ? error_name : "Unknown error";
            throw std::runtime_error {"Failed to encode tensor map: " + std::string{error_name}};*/
            throw std::runtime_error {"Failed to encode tensor map: " + std::string{name} + ": " + std::to_string(res)};
        }
        return map;
    }


    template <const int32_t BM, const int32_t BN, const int32_t WN, const int32_t STAGES>
    static void launch_fused_moe_kernel(
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
        auto *kernel = fused_moe_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS>;
        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess)
            throw std::runtime_error {"Failed to set max dynamic shared memory size for fused_moe_kernel: " + std::to_string(SMEM) + " bytes"};
        auto* cuTensorMapEncodeTiled = getProcAddress_cuTensorMapEncodeTiled();
        CUtensorMap map_w = init_tmap<2>(
            "w",
            w,
            {static_cast<uint64_t>(K), static_cast<uint64_t>(N*num_experts)},
            {K*sizeof(__nv_bfloat16)},
            {BK, BN*WN},
            {1, 1},
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
        );
        CUtensorMap map_w2 = init_tmap<2>(
            "w2",
            w2,
            {static_cast<uint64_t>(N>>1), static_cast<uint64_t>(K*num_experts)},
            {(N>>1)*sizeof(__nv_bfloat16)},
            {(WN*BN)>>1, BK<<1},
            {1, 1},
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
        );
        kernel<<<grid, block, SMEM, stream>>>(
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
        switch (block_m) {
            case 8: dispatch_bn_wn<8>(block_n, warp_n, stages, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case 16: dispatch_bn_wn<16>(block_n, warp_n, stages, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case 32: dispatch_bn_wn<32>(block_n, warp_n, stages, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case 64: dispatch_bn_wn<64>(block_n, warp_n, stages, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case 128: dispatch_bn_wn<128>(block_n, warp_n, stages, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            default:
                throw std::runtime_error {"Unsupported block_m for tcgen05 path: " + std::to_string(block_m)};
        }
    }
}