#pragma once

#include <cuda.h>

namespace pi {
    [[nodiscard]] constexpr __device__ __forceinline__ int32_t const_ceil(float x) {
        return (static_cast<float>(static_cast<int32_t>(x)) == x) ? static_cast<int32_t>(x) : static_cast<int32_t>(x) + (x > 0);
    }

    [[nodiscard]] constexpr __device__ __forceinline__ uint64_t matrix_desc_encode(uint64_t x) {
        return (x & 0x3ffff)>>4;
    }

    template <const uint32_t bits, const uint32_t base=4, const uint32_t shift=3>
    [[nodiscard]] constexpr __device__ __forceinline__ int32_t swizzle(int32_t i) {
        if constexpr (!bits) return i;
        constexpr uint32_t mask = ((1u<<bits)-1)<<(base+shift);
        return i^((i&mask)>>shift);
    }

    template <const uint64_t sdo, const uint64_t ldo, uint64_t mode>
    [[nodiscard]] __device__ __forceinline__ uint64_t smem_desc_encode(void *p) {
        auto addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
        uint64_t desc = (static_cast<uint64_t>(addr) & 0x3ffff)>>4;
        return desc | (ldo<<16) | (sdo<<32) | (mode<<62);
    }

    [[nodiscard]] __device__ __forceinline__ float swiglu(float x, float w) {
        return w*(x/(1.f+__expf(-x)));
    }

    namespace warp_group {
        __device__ __forceinline__ void arrive() {
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        }

        __device__ __forceinline__ void commit_batch() {
            asm volatile ("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        }

        __device__ __forceinline__ void await() {
            asm volatile ("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        }
    }

    namespace cp_async {
        __device__ __forceinline__ void load(void *dst, const void *tma_map, uint64_t *bar, int row, int col) {
            static_assert(sizeof(void *) == sizeof(uint64_t));
            auto ptma = reinterpret_cast<uint64_t>(tma_map);
            auto pmbar = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
            auto pdst = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0] [%1, {%3, %4}], [%2];\n"
                :
                : "r"(pdst), "l"(ptma), "r"(pmbar), "r"(row), "r"(col)
                : "memory"
            );
        }

        __device__ __forceinline__ void cg256(void *dst, const void *src, int nb) {
            asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" :: "r"(dst), "l"(src), "n"(nb));
        }

        __device__ __forceinline__ void cg64(void *dst, const void *src, int nb) {
            asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], %2;\n" :: "r"(dst), "l"(src), "n"(nb));
        }

        __device__ __forceinline__ void commit_group() {
            asm volatile("cp.async.commit_group;\n" :::);
        }

        __device__ __forceinline__ void await_group(int n) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
        }
    }
}

