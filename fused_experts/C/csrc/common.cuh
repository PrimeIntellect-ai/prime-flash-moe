#pragma once

#include <cuda.h>

namespace pi {
    [[nodiscard]] constexpr __device__ __forceinline__ int32_t const_ceil(float x) {
        return (static_cast<float>(static_cast<int32_t>(x)) == x) ? static_cast<int32_t>(x) : static_cast<int32_t>(x) + ((num > 0) ? 1 : 0);
    }

    [[nodiscard]] constexpr __device__ __forceinline__ uint64_t matrix_desc_encode(uint64_t x) {
        return (x & 0x3ffff)>>4;
    }

    template <const uint32_t bits, const uint32_t base=4, const uint32_t shift=3>
    [[nodiscard]] constexpr __device__ __forceinline__ int32_t swizzle(int32_t i) {
        if constexpr (!bits) return i;
        static constexpr uint32_t mask = ((1u<<bits)-1)<<(base+shift);
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

    namespace warpgroup {
        __device__ __forceinline__ void arrive() {
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        }

        __device__ __forceinline__ void commit_batch() {
            asm volatile ("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        }

        __device__ __forceinline__ void await() {
            asm volatile ("wgmma.wait_group.sync.aligned 0;\n" ::: memory);
        }
    }
}

