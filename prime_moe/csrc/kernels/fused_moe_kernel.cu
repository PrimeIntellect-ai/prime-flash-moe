#include <stdexcept>
#include <array>

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

    namespace tcgen05 {
        [[nodiscard]] __device__ __forceinline__ uint64_t smem_desc_kmajor(const void *ptr, int32_t height) {
            return ((static_cast<uint64_t>(static_cast<uint32_t>(__cvta_generic_to_shared(ptr)))&0x3ffffu)>>4)
                 |(((static_cast<uint64_t>(height*16)&0x3ffffu)>>4)<<16)
                 |(((static_cast<uint64_t>(8*16)&0x3ffffu)>>4)<<32)
                 |0x160400000000000;
        }

        [[nodiscard]] __device__ __forceinline__ uint32_t idesc_f32_bf16_bf16(int32_t m, int32_t n) {
            return (1u<<4)
                 |(1u<<7)
                 |(1u<<10)
                 |((static_cast<uint32_t>(n)>>3)<<17)
                 |((static_cast<uint32_t>(m)>>4)<<24);
        }

        [[nodiscard]] __device__ __forceinline__ uint32_t tmem_addr(uint32_t base, uint32_t row, uint32_t col) {
            return base + (row<<16) + col;
        }

        __device__ __forceinline__ void mma_f16(
            uint32_t d_tmem,
            uint64_t desc_a,
            uint64_t desc_b,
            uint32_t idesc,
            int32_t enable_input_d
        ) {
            uint32_t mask[4] = {0, 0, 0, 0};
            uint64_t idescE = static_cast<uint64_t>(idesc) << 32;
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
                "}\n"
                :
                : "r"(d_tmem), "l"(desc_a), "l"(desc_b),
                  "r"(static_cast<uint32_t>(idescE>>32)),
                  "r"(enable_input_d),
                  "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
                : "memory"
            );
        }

        __device__ __forceinline__ void commit_mbarrier(uint64_t *mbar) {
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                :
                : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar)))
                : "memory"
            );
        }

        __device__ __forceinline__ void after_thread_sync() {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        }

        __device__ __forceinline__ void await_ld() {
            asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
        }

        __device__ __forceinline__ void tmem_alloc(void *dst_smem, int32_t cols) {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                :
                : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem))), "r"(cols)
                : "memory"
            );
        }

        __device__ __forceinline__ void tmem_free(uint32_t taddr, int32_t cols) {
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                :
                : "r"(taddr), "r"(cols)
                : "memory"
            );
        }

        __device__ __forceinline__ void ld_32x32b_x8(float (&dst)[8], uint32_t addr) {
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
                  "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
                : "r"(addr)
                : "memory"
            );
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

        __device__ __forceinline__ void load3d(
            void *dst,
            const void *tma_map,
            uint64_t *bar,
            int32_t c0,
            int32_t c1,
            int32_t c2
        ) {
            auto ptma  = reinterpret_cast<uint64_t>(tma_map);
            auto pmbar = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
            auto pdst  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%3, %4, %5}], [%2];\n"
                :
                : "r"(pdst), "l"(ptma), "r"(pmbar), "r"(c0), "r"(c1), "r"(c2)
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
        static_assert(BM == 128, "tcgen05 path: BM=128 / Layout-D only.");
        static_assert(WN*BN == 256, "tcgen05 path assumes 256 output columns per MMA tile.");

        static constexpr int32_t CONSUMER_THREADS = WN<<5;
        static constexpr int32_t BLOCK_SHAPE_K = 128;
        static constexpr int32_t BK2 = (WN*BN)>>1;
        static constexpr int32_t BN2 = BK<<1;
        static constexpr int32_t WS = WN*BK*BN;
        static constexpr int32_t XS = BK*BM;
        static constexpr int32_t TB = 16;
        static constexpr int32_t PAD = BN2+8;
        static constexpr int32_t TC_N = WN*BN;

        extern __shared__ __align__(1024) uint8_t smem_raw[];
        __shared__ __align__(8) uint64_t bar_copy[STAGES];
        __shared__ __align__(8) uint64_t bar_recycle[STAGES];
        __shared__ __align__(8) uint64_t bar_mma[STAGES];
        __shared__ __align__(4) uint32_t tmem_acc;
        __shared__ float topk_scales[BM];
        __shared__ int32_t row_tok[BM];

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
            for (int32_t i = 0; i < STAGES; ++i) {
                barrier::init(bar_copy + i, PRODUCER_THREADS + 1, 0);
                barrier::init(bar_recycle + i, CONSUMER_THREADS, 0);
                barrier::init(bar_mma + i, 1, 0);
            }
            asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        }
        __syncthreads();
        if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS + 32) {
            tcgen05::tmem_alloc(&tmem_acc, 256);
        }
        __syncthreads();
        if (threadIdx.x < BM) {
            int32_t tdest = sorted_token_ids[warp_m*BM + threadIdx.x];
            int32_t tok = -1;
            if (tdest >= 0) tok = tdest / top_k;
            row_tok[threadIdx.x] = tok;

            if (tdest >= 0 && tok < M) {
                uint32_t ptr = __cvta_generic_to_shared(topk_scales + threadIdx.x);
                cp_async::cg64<4>(ptr, topk_weights + tdest);
            }
        }
        __syncthreads();
        auto con_sync = [=]() { asm volatile ("bar.sync 0, %0;\n" :: "n"(CONSUMER_THREADS)); };
        int32_t n_stages_up = K / BLOCK_SHAPE_K;
        int32_t n_stages_down = N2 / BN2;
        int32_t phase = 0;
        const auto producer = [&] {
            constexpr int32_t CHUNKS_PER_ROW = BK / 8;
            constexpr int32_t ROWS_PER_WAVE  = PRODUCER_THREADS / CHUNKS_PER_ROW;
            int32_t smem_stage = 0;
            for (int32_t stage = 0; stage < n_stages_up; ++stage) {
                if (smem_stage == STAGES) {
                    phase ^= 1;
                    smem_stage = 0;
                }
                const int32_t offs = stage*BK;
                barrier::await(bar_recycle + smem_stage, phase);
                for (int32_t wave=0; wave < BM / ROWS_PER_WAVE; ++wave) {
                    const int32_t row_in_block = wave*ROWS_PER_WAVE + (threadIdx.x / CHUNKS_PER_ROW);
                    const int32_t k8 = threadIdx.x % CHUNKS_PER_ROW;
                    const int32_t tok = row_tok[row_in_block];
                    if (row_in_block < BM && tok >= 0 && tok < M) {
                        __nv_bfloat16 *dst = smem_up.x + smem_stage*XS + (k8*BM + row_in_block)*8;
                        const uint4 *src = reinterpret_cast<const uint4 *>(x + tok*K + offs + (k8 << 3));
                        cp_async::cg256<TB>(static_cast<uint32_t>(__cvta_generic_to_shared(dst)), src);
                    }
                }
                barrier::arrive_cp_async_mem(bar_copy + smem_stage);
                if (!threadIdx.x) {
                    barrier::expect_nb(bar_copy + smem_stage, WS*sizeof(__nv_bfloat16));
                    cp_async::load3d(
                        smem_up.w + smem_stage*WS,
                        &map_w,
                        bar_copy + smem_stage,
                        0,
                        expert_idx*N + blockIdx.x*(WN*BN),
                        offs / 8
                    );
                }
                ++smem_stage;
            }
            auto &smem_down = *reinterpret_cast<smem_down_pod<STAGES, WN, BM, BK, BN> *>(smem_raw);
            for (int32_t stage = 0; stage < n_stages_down; ++stage) {
                if (smem_stage == STAGES) {
                    phase ^= 1;
                    smem_stage = 0;
                }
                barrier::await(bar_recycle + smem_stage, phase);
                if (!threadIdx.x) {
                    barrier::expect_nb(bar_copy + smem_stage, WS*sizeof(__nv_bfloat16));
                    cp_async::load3d(
                        smem_down.w + smem_stage*WS,
                        &map_w2,
                        bar_copy + smem_stage,
                        0,
                        expert_idx*N2 + stage*BN2,
                        (blockIdx.x*BK2) / 8
                    );
                }
                barrier::arrive_cp_async_mem(bar_copy + smem_stage);
                ++smem_stage;
            }
        };

        const auto consumer = [&] {
            int32_t tok_src = -1;
            if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS + BM) {
                int32_t tdest = sorted_token_ids[warp_m*BM + (threadIdx.x - PRODUCER_THREADS)];
                if (tdest >= 0) tok_src = tdest / top_k;
            }
            for (int32_t i=0; i < STAGES; ++i) barrier::arrive(bar_recycle + i);
            const uint32_t idesc = tcgen05::idesc_f32_bf16_bf16(BM, TC_N);
            int32_t smem_stage = 0;
            for (int32_t stage = 0; stage < n_stages_up; ++stage) {
                if (smem_stage == STAGES) {
                    phase ^= 1;
                    smem_stage = 0;
                }
                barrier::await(bar_copy + smem_stage, phase);
                if (warp_id == 0 && lane_id == 0) {
                    #pragma unroll
                    for (int32_t k16 = 0; k16 < BK / 16; ++k16) {
                        __nv_bfloat16 *a_ptr = smem_up.x + smem_stage*XS + k16*BM*16;
                        __nv_bfloat16 *b_ptr = smem_up.w + smem_stage*WS + k16*TC_N*16;

                        uint64_t a_desc = tcgen05::smem_desc_kmajor(a_ptr, BM);
                        uint64_t b_desc = tcgen05::smem_desc_kmajor(b_ptr, TC_N);

                        tcgen05::mma_f16(
                            tmem_acc,
                            a_desc,
                            b_desc,
                            idesc,
                            (stage != 0) || (k16 != 0)
                        );
                    }
                    tcgen05::commit_mbarrier(bar_mma + smem_stage);
                }
                barrier::await(bar_mma + smem_stage, phase);
                barrier::arrive(bar_recycle + smem_stage);
                ++smem_stage;
            }
            con_sync();
            tcgen05::after_thread_sync();
            auto &smem_down = *reinterpret_cast<smem_down_pod<STAGES, WN, BM, BK, BN> *>(smem_raw);
            if (warp_id < 4) {
                const int32_t row = (warp_id << 5) + lane_id;

                #pragma unroll
                for (int32_t c8 = 0; c8 < BK2 / 8; ++c8) {
                    float gate[8];
                    float up[8];

                    tcgen05::ld_32x32b_x8(gate, tcgen05::tmem_addr(tmem_acc, row, (c8 << 3)));
                    tcgen05::ld_32x32b_x8(up,   tcgen05::tmem_addr(tmem_acc, row, BK2 + (c8 << 3)));
                    tcgen05::await_ld();

                    __nv_bfloat16 *dst = smem_down.x + (c8*BM + row)*8;
                    #pragma unroll
                    for (int32_t e = 0; e < 8; ++e) {
                        dst[e] = __float2bfloat16(swiglu(gate[e], up[e]));
                    }
                }
            }

            con_sync();

            for (int32_t stage = 0; stage < n_stages_down; ++stage) {
                if (smem_stage == STAGES) {
                    phase ^= 1;
                    smem_stage = 0;
                }

                barrier::await(bar_copy + smem_stage, phase);
                if (warp_id == 0 && lane_id == 0) {
                    #pragma unroll
                    for (int32_t k16 = 0; k16 < BK2 / 16; ++k16) {
                        __nv_bfloat16 *a_ptr = smem_down.x + k16*BM*16;
                        __nv_bfloat16 *b_ptr = smem_down.w + smem_stage*WS + k16*BN2*16;
                        uint64_t a_desc = tcgen05::smem_desc_kmajor(a_ptr, BM);
                        uint64_t b_desc = tcgen05::smem_desc_kmajor(b_ptr, BN2);
                        tcgen05::mma_f16(
                            tmem_acc,
                            a_desc,
                            b_desc,
                            idesc,
                            k16 != 0
                        );
                    }
                    tcgen05::commit_mbarrier(bar_mma + smem_stage);
                }

                barrier::await(bar_mma + smem_stage, phase);
                barrier::arrive(bar_recycle + smem_stage);
                asm volatile("cp.async.bulk.wait_group 0;\n");
                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                con_sync();
                tcgen05::after_thread_sync();
                if (warp_id < 4) {
                    const int32_t row = (warp_id << 5) + lane_id;
                    #pragma unroll
                    for (int32_t c8 = 0; c8 < BN2 / 8; ++c8) {
                        float vals[8];
                        tcgen05::ld_32x32b_x8(vals, tcgen05::tmem_addr(tmem_acc, row, (c8 << 3)));
                        tcgen05::await_ld();
                        __nv_bfloat16 *dst = smem_down.out + row*PAD + (c8 << 3);
                        #pragma unroll
                        for (int32_t e = 0; e < 8; ++e) {
                            dst[e] = __float2bfloat16(vals[e]);
                        }
                    }
                }
                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                con_sync();
                if (threadIdx.x < PRODUCER_THREADS + BM) {
                    if (tok_src >= 0 && tok_src < M) {
                        cuda::ptx::cp_reduce_async_bulk(
                            cuda::ptx::space_global,
                            cuda::ptx::space_shared,
                            cuda::ptx::op_add,
                            out + tok_src*N2 + stage*BN2,
                            smem_down.out + PAD*(threadIdx.x - PRODUCER_THREADS),
                            BN2*sizeof(__nv_bfloat16)
                        );
                    }
                    cuda::ptx::cp_async_bulk_commit_group();
                }
                ++smem_stage;
            }

            asm volatile("cp.async.bulk.wait_group 0;\n");
            con_sync();
            if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS + 32) {
                tcgen05::tmem_free(tmem_acc, 256);
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
            throw std::runtime_error {"Failed to encode tensor map: " + std::string{name} + ": " + std::to_string(res)};
        }
        return map;
    }

    [[nodiscard]] static CUtensorMap init_tmap_kmajor_3d(
        const char *name,
        const void *ptr,
        uint64_t global_height,
        uint64_t global_width,
        uint32_t tile_height,
        uint32_t tile_width
    ) {
        if ((global_width & 7ull) || (tile_width & 7u))
            throw std::runtime_error{std::string{name} + ": K-like dim must be a multiple of 8 bf16"};

        return init_tmap<3>(
            name,
            ptr,
            {8ull, global_height, global_width / 8ull},
            {global_width*sizeof(__nv_bfloat16), 16ull, 0ull},
            {8u, tile_height, tile_width / 8u},
            {1u, 1u, 1u},
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
        );
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

        CUtensorMap map_w = init_tmap_kmajor_3d(
            "w",
            w,
            static_cast<uint64_t>(N*num_experts),
            static_cast<uint64_t>(K),
            BN*WN,
            BK
        );
        CUtensorMap map_w2 = init_tmap_kmajor_3d(
            "w2",
            w2,
            static_cast<uint64_t>(K*num_experts),
            static_cast<uint64_t>(N >> 1),
            BK << 1,
            (WN*BN) >> 1
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
            case 128:
                dispatch_bn_wn<128>(block_n, warp_n, stages, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream);
                break;
            default:
                throw std::runtime_error{
                    "tcgen05 Blackwell path currently supports block_m=128 only; "
                    "BM=64 needs Layout-F epilogue, BM=32 needs tcgen05.mma.ws, BM=8/16 are unsupported."
                };
        }
    }
}
