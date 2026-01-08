#pragma once

#include <cuda.h>

namespace pi {
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

    namespace wgmma {
        __device__ __forceinline__ void mm8(float r[1][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]), "+f"(r[0][1]), "+f"(r[0][2]), "+f"(r[0][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm16(float r[2][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]), "+f"(r[0][1]), "+f"(r[0][2]), "+f"(r[0][3]),
                  "+f"(r[1][0]), "+f"(r[1][1]), "+f"(r[1][2]), "+f"(r[1][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm24(float r[3][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, %12, %13, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]), "+f"(r[0][1]), "+f"(r[0][2]), "+f"(r[0][3]),
                  "+f"(r[1][0]), "+f"(r[1][1]), "+f"(r[1][2]), "+f"(r[1][3]),
                  "+f"(r[2][0]), "+f"(r[2][1]), "+f"(r[2][2]), "+f"(r[2][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm32(float r[4][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]), "+f"(r[0][1]), "+f"(r[0][2]), "+f"(r[0][3]),
                  "+f"(r[1][0]), "+f"(r[1][1]), "+f"(r[1][2]), "+f"(r[1][3]),
                  "+f"(r[2][0]), "+f"(r[2][1]), "+f"(r[2][2]), "+f"(r[2][3]),
                  "+f"(r[3][0]), "+f"(r[3][1]), "+f"(r[3][2]), "+f"(r[3][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm40(float r[5][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, %20, %21, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]), "+f"(r[0][1]), "+f"(r[0][2]), "+f"(r[0][3]),
                  "+f"(r[1][0]), "+f"(r[1][1]), "+f"(r[1][2]), "+f"(r[1][3]),
                  "+f"(r[2][0]), "+f"(r[2][1]), "+f"(r[2][2]), "+f"(r[2][3]),
                  "+f"(r[3][0]), "+f"(r[3][1]), "+f"(r[3][2]), "+f"(r[3][3]),
                  "+f"(r[4][0]), "+f"(r[4][1]), "+f"(r[4][2]), "+f"(r[4][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm48(float r[6][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, %24, %25, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]), "+f"(r[0][1]), "+f"(r[0][2]), "+f"(r[0][3]),
                  "+f"(r[1][0]), "+f"(r[1][1]), "+f"(r[1][2]), "+f"(r[1][3]),
                  "+f"(r[2][0]), "+f"(r[2][1]), "+f"(r[2][2]), "+f"(r[2][3]),
                  "+f"(r[3][0]), "+f"(r[3][1]), "+f"(r[3][2]), "+f"(r[3][3]),
                  "+f"(r[4][0]), "+f"(r[4][1]), "+f"(r[4][2]), "+f"(r[4][3]),
                  "+f"(r[5][0]), "+f"(r[5][1]), "+f"(r[5][2]), "+f"(r[5][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm56(float r[7][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, %28, %29, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]), "+f"(r[0][1]), "+f"(r[0][2]), "+f"(r[0][3]),
                  "+f"(r[1][0]), "+f"(r[1][1]), "+f"(r[1][2]), "+f"(r[1][3]),
                  "+f"(r[2][0]), "+f"(r[2][1]), "+f"(r[2][2]), "+f"(r[2][3]),
                  "+f"(r[3][0]), "+f"(r[3][1]), "+f"(r[3][2]), "+f"(r[3][3]),
                  "+f"(r[4][0]), "+f"(r[4][1]), "+f"(r[4][2]), "+f"(r[4][3]),
                  "+f"(r[5][0]), "+f"(r[5][1]), "+f"(r[5][2]), "+f"(r[5][3]),
                  "+f"(r[6][0]), "+f"(r[6][1]), "+f"(r[6][2]), "+f"(r[6][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm64(float r[8][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {"
                    "%0 ,  %1,  %2,  %3, "
                    "%4 ,  %5,  %6,  %7, "
                    "%8 ,  %9, %10, %11, "
                    "%12, %13, %14, %15, "
                    "%16, %17, %18, %19, "
                    "%20, %21, %22, %23, "
                    "%24, %25, %26, %27, "
                    "%28, %29, %30, %31  "
                "}, %32, %33, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]), "+f"(r[0][1]), "+f"(r[0][2]), "+f"(r[0][3]),
                  "+f"(r[1][0]), "+f"(r[1][1]), "+f"(r[1][2]), "+f"(r[1][3]),
                  "+f"(r[2][0]), "+f"(r[2][1]), "+f"(r[2][2]), "+f"(r[2][3]),
                  "+f"(r[3][0]), "+f"(r[3][1]), "+f"(r[3][2]), "+f"(r[3][3]),
                  "+f"(r[4][0]), "+f"(r[4][1]), "+f"(r[4][2]), "+f"(r[4][3]),
                  "+f"(r[5][0]), "+f"(r[5][1]), "+f"(r[5][2]), "+f"(r[5][3]),
                  "+f"(r[6][0]), "+f"(r[6][1]), "+f"(r[6][2]), "+f"(r[6][3]),
                  "+f"(r[7][0]), "+f"(r[7][1]), "+f"(r[7][2]), "+f"(r[7][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm72(float r[9][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {"
                    "%0 ,  %1,  %2,  %3, "
                    "%4 ,  %5,  %6,  %7, "
                    "%8 ,  %9, %10, %11, "
                    "%12, %13, %14, %15, "
                    "%16, %17, %18, %19, "
                    "%20, %21, %22, %23, "
                    "%24, %25, %26, %27, "
                    "%28, %29, %30, %31, "
                    "%32, %33, %34, %35  "
                "}, %36, %37, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]), "+f"(r[0][1]), "+f"(r[0][2]), "+f"(r[0][3]),
                  "+f"(r[1][0]), "+f"(r[1][1]), "+f"(r[1][2]), "+f"(r[1][3]),
                  "+f"(r[2][0]), "+f"(r[2][1]), "+f"(r[2][2]), "+f"(r[2][3]),
                  "+f"(r[3][0]), "+f"(r[3][1]), "+f"(r[3][2]), "+f"(r[3][3]),
                  "+f"(r[4][0]), "+f"(r[4][1]), "+f"(r[4][2]), "+f"(r[4][3]),
                  "+f"(r[5][0]), "+f"(r[5][1]), "+f"(r[5][2]), "+f"(r[5][3]),
                  "+f"(r[6][0]), "+f"(r[6][1]), "+f"(r[6][2]), "+f"(r[6][3]),
                  "+f"(r[7][0]), "+f"(r[7][1]), "+f"(r[7][2]), "+f"(r[7][3]),
                  "+f"(r[8][0]), "+f"(r[8][1]), "+f"(r[8][2]), "+f"(r[8][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm80(float r[10][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 {"
                    "%0 ,  %1,  %2,  %3, "
                    "%4 ,  %5,  %6,  %7, "
                    "%8 ,  %9, %10, %11, "
                    "%12, %13, %14, %15, "
                    "%16, %17, %18, %19, "
                    "%20, %21, %22, %23, "
                    "%24, %25, %26, %27, "
                    "%28, %29, %30, %31, "
                    "%32, %33, %34, %35, "
                    "%36, %37, %38, %39  "
                "}, %40, %41, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]), "+f"(r[0][1]), "+f"(r[0][2]), "+f"(r[0][3]),
                  "+f"(r[1][0]), "+f"(r[1][1]), "+f"(r[1][2]), "+f"(r[1][3]),
                  "+f"(r[2][0]), "+f"(r[2][1]), "+f"(r[2][2]), "+f"(r[2][3]),
                  "+f"(r[3][0]), "+f"(r[3][1]), "+f"(r[3][2]), "+f"(r[3][3]),
                  "+f"(r[4][0]), "+f"(r[4][1]), "+f"(r[4][2]), "+f"(r[4][3]),
                  "+f"(r[5][0]), "+f"(r[5][1]), "+f"(r[5][2]), "+f"(r[5][3]),
                  "+f"(r[6][0]), "+f"(r[6][1]), "+f"(r[6][2]), "+f"(r[6][3]),
                  "+f"(r[7][0]), "+f"(r[7][1]), "+f"(r[7][2]), "+f"(r[7][3]),
                  "+f"(r[8][0]), "+f"(r[8][1]), "+f"(r[8][2]), "+f"(r[8][3]),
                  "+f"(r[9][0]), "+f"(r[9][1]), "+f"(r[9][2]), "+f"(r[9][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm88(float r[11][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n88k16.f32.bf16.bf16 {"
                    "%0 ,  %1,  %2,  %3, "
                    "%4 ,  %5,  %6,  %7, "
                    "%8 ,  %9, %10, %11, "
                    "%12, %13, %14, %15, "
                    "%16, %17, %18, %19, "
                    "%20, %21, %22, %23, "
                    "%24, %25, %26, %27, "
                    "%28, %29, %30, %31, "
                    "%32, %33, %34, %35, "
                    "%36, %37, %38, %39, "
                    "%40, %41, %42, %43  "
                "}, %44, %45, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]), "+f"(r[0][1]), "+f"(r[0][2]), "+f"(r[0][3]),
                  "+f"(r[1][0]), "+f"(r[1][1]), "+f"(r[1][2]), "+f"(r[1][3]),
                  "+f"(r[2][0]), "+f"(r[2][1]), "+f"(r[2][2]), "+f"(r[2][3]),
                  "+f"(r[3][0]), "+f"(r[3][1]), "+f"(r[3][2]), "+f"(r[3][3]),
                  "+f"(r[4][0]), "+f"(r[4][1]), "+f"(r[4][2]), "+f"(r[4][3]),
                  "+f"(r[5][0]), "+f"(r[5][1]), "+f"(r[5][2]), "+f"(r[5][3]),
                  "+f"(r[6][0]), "+f"(r[6][1]), "+f"(r[6][2]), "+f"(r[6][3]),
                  "+f"(r[7][0]), "+f"(r[7][1]), "+f"(r[7][2]), "+f"(r[7][3]),
                  "+f"(r[8][0]), "+f"(r[8][1]), "+f"(r[8][2]), "+f"(r[8][3]),
                  "+f"(r[9][0]), "+f"(r[9][1]), "+f"(r[9][2]), "+f"(r[9][3]),
                  "+f"(r[10][0]), "+f"(r[10][1]), "+f"(r[10][2]), "+f"(r[10][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm96(float r[12][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {"
                    "%0 ,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                    "%8 ,  %9, %10, %11, %12, %13, %14, %15, "
                    "%16, %17, %18, %19, %20, %21, %22, %23, "
                    "%24, %25, %26, %27, %28, %29, %30, %31, "
                    "%32, %33, %34, %35, %36, %37, %38, %39, "
                    "%40, %41, %42, %43, %44, %45, %46, %47  "
                "}, %48, %49, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]),  "+f"(r[0][1]),  "+f"(r[0][2]),  "+f"(r[0][3]),
                  "+f"(r[1][0]),  "+f"(r[1][1]),  "+f"(r[1][2]),  "+f"(r[1][3]),
                  "+f"(r[2][0]),  "+f"(r[2][1]),  "+f"(r[2][2]),  "+f"(r[2][3]),
                  "+f"(r[3][0]),  "+f"(r[3][1]),  "+f"(r[3][2]),  "+f"(r[3][3]),
                  "+f"(r[4][0]),  "+f"(r[4][1]),  "+f"(r[4][2]),  "+f"(r[4][3]),
                  "+f"(r[5][0]),  "+f"(r[5][1]),  "+f"(r[5][2]),  "+f"(r[5][3]),
                  "+f"(r[6][0]),  "+f"(r[6][1]),  "+f"(r[6][2]),  "+f"(r[6][3]),
                  "+f"(r[7][0]),  "+f"(r[7][1]),  "+f"(r[7][2]),  "+f"(r[7][3]),
                  "+f"(r[8][0]),  "+f"(r[8][1]),  "+f"(r[8][2]),  "+f"(r[8][3]),
                  "+f"(r[9][0]),  "+f"(r[9][1]),  "+f"(r[9][2]),  "+f"(r[9][3]),
                  "+f"(r[10][0]), "+f"(r[10][1]), "+f"(r[10][2]), "+f"(r[10][3]),
                  "+f"(r[11][0]), "+f"(r[11][1]), "+f"(r[11][2]), "+f"(r[11][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm104(float r[13][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {"
                    "%0 ,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                    "%8 ,  %9, %10, %11, %12, %13, %14, %15, "
                    "%16, %17, %18, %19, %20, %21, %22, %23, "
                    "%24, %25, %26, %27, %28, %29, %30, %31, "
                    "%32, %33, %34, %35, %36, %37, %38, %39, "
                    "%40, %41, %42, %43, %44, %45, %46, %47,  "
                    "%48, %49, %50, %51 "
                "}, %52, %53, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]),  "+f"(r[0][1]),  "+f"(r[0][2]),  "+f"(r[0][3]),
                  "+f"(r[1][0]),  "+f"(r[1][1]),  "+f"(r[1][2]),  "+f"(r[1][3]),
                  "+f"(r[2][0]),  "+f"(r[2][1]),  "+f"(r[2][2]),  "+f"(r[2][3]),
                  "+f"(r[3][0]),  "+f"(r[3][1]),  "+f"(r[3][2]),  "+f"(r[3][3]),
                  "+f"(r[4][0]),  "+f"(r[4][1]),  "+f"(r[4][2]),  "+f"(r[4][3]),
                  "+f"(r[5][0]),  "+f"(r[5][1]),  "+f"(r[5][2]),  "+f"(r[5][3]),
                  "+f"(r[6][0]),  "+f"(r[6][1]),  "+f"(r[6][2]),  "+f"(r[6][3]),
                  "+f"(r[7][0]),  "+f"(r[7][1]),  "+f"(r[7][2]),  "+f"(r[7][3]),
                  "+f"(r[8][0]),  "+f"(r[8][1]),  "+f"(r[8][2]),  "+f"(r[8][3]),
                  "+f"(r[9][0]),  "+f"(r[9][1]),  "+f"(r[9][2]),  "+f"(r[9][3]),
                  "+f"(r[10][0]), "+f"(r[10][1]), "+f"(r[10][2]), "+f"(r[10][3]),
                  "+f"(r[11][0]), "+f"(r[11][1]), "+f"(r[11][2]), "+f"(r[11][3]),
                  "+f"(r[12][0]), "+f"(r[12][1]), "+f"(r[12][2]), "+f"(r[12][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm112(float r[14][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {"
                    "%0 ,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                    "%8 ,  %9, %10, %11, %12, %13, %14, %15, "
                    "%16, %17, %18, %19, %20, %21, %22, %23, "
                    "%24, %25, %26, %27, %28, %29, %30, %31, "
                    "%32, %33, %34, %35, %36, %37, %38, %39, "
                    "%40, %41, %42, %43, %44, %45, %46, %47, "
                    "%48, %49, %50, %51, %52, %53, %54, %55  "
                "}, %56, %57, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]),  "+f"(r[0][1]),  "+f"(r[0][2]),  "+f"(r[0][3]),
                  "+f"(r[1][0]),  "+f"(r[1][1]),  "+f"(r[1][2]),  "+f"(r[1][3]),
                  "+f"(r[2][0]),  "+f"(r[2][1]),  "+f"(r[2][2]),  "+f"(r[2][3]),
                  "+f"(r[3][0]),  "+f"(r[3][1]),  "+f"(r[3][2]),  "+f"(r[3][3]),
                  "+f"(r[4][0]),  "+f"(r[4][1]),  "+f"(r[4][2]),  "+f"(r[4][3]),
                  "+f"(r[5][0]),  "+f"(r[5][1]),  "+f"(r[5][2]),  "+f"(r[5][3]),
                  "+f"(r[6][0]),  "+f"(r[6][1]),  "+f"(r[6][2]),  "+f"(r[6][3]),
                  "+f"(r[7][0]),  "+f"(r[7][1]),  "+f"(r[7][2]),  "+f"(r[7][3]),
                  "+f"(r[8][0]),  "+f"(r[8][1]),  "+f"(r[8][2]),  "+f"(r[8][3]),
                  "+f"(r[9][0]),  "+f"(r[9][1]),  "+f"(r[9][2]),  "+f"(r[9][3]),
                  "+f"(r[10][0]), "+f"(r[10][1]), "+f"(r[10][2]), "+f"(r[10][3]),
                  "+f"(r[11][0]), "+f"(r[11][1]), "+f"(r[11][2]), "+f"(r[11][3]),
                  "+f"(r[12][0]), "+f"(r[12][1]), "+f"(r[12][2]), "+f"(r[12][3]),
                  "+f"(r[13][0]), "+f"(r[13][1]), "+f"(r[13][2]), "+f"(r[13][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm120(float r[15][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16 {"
                    "%0 ,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                    "%8 ,  %9, %10, %11, %12, %13, %14, %15, "
                    "%16, %17, %18, %19, %20, %21, %22, %23, "
                    "%24, %25, %26, %27, %28, %29, %30, %31, "
                    "%32, %33, %34, %35, %36, %37, %38, %39, "
                    "%40, %41, %42, %43, %44, %45, %46, %47, "
                    "%48, %49, %50, %51, %52, %53, %54, %55,  "
                    "%56, %57, %58, %59 "
                "}, %60, %61, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]),  "+f"(r[0][1]),  "+f"(r[0][2]),  "+f"(r[0][3]),
                  "+f"(r[1][0]),  "+f"(r[1][1]),  "+f"(r[1][2]),  "+f"(r[1][3]),
                  "+f"(r[2][0]),  "+f"(r[2][1]),  "+f"(r[2][2]),  "+f"(r[2][3]),
                  "+f"(r[3][0]),  "+f"(r[3][1]),  "+f"(r[3][2]),  "+f"(r[3][3]),
                  "+f"(r[4][0]),  "+f"(r[4][1]),  "+f"(r[4][2]),  "+f"(r[4][3]),
                  "+f"(r[5][0]),  "+f"(r[5][1]),  "+f"(r[5][2]),  "+f"(r[5][3]),
                  "+f"(r[6][0]),  "+f"(r[6][1]),  "+f"(r[6][2]),  "+f"(r[6][3]),
                  "+f"(r[7][0]),  "+f"(r[7][1]),  "+f"(r[7][2]),  "+f"(r[7][3]),
                  "+f"(r[8][0]),  "+f"(r[8][1]),  "+f"(r[8][2]),  "+f"(r[8][3]),
                  "+f"(r[9][0]),  "+f"(r[9][1]),  "+f"(r[9][2]),  "+f"(r[9][3]),
                  "+f"(r[10][0]), "+f"(r[10][1]), "+f"(r[10][2]), "+f"(r[10][3]),
                  "+f"(r[11][0]), "+f"(r[11][1]), "+f"(r[11][2]), "+f"(r[11][3]),
                  "+f"(r[12][0]), "+f"(r[12][1]), "+f"(r[12][2]), "+f"(r[12][3]),
                  "+f"(r[13][0]), "+f"(r[13][1]), "+f"(r[13][2]), "+f"(r[13][3]),
                  "+f"(r[14][0]), "+f"(r[14][1]), "+f"(r[14][2]), "+f"(r[14][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        __device__ __forceinline__ void mm128(float r[16][4], uint64_t desc_a, uint64_t desc_b) {
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {"
                    "%0 ,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                    "%8 ,  %9, %10, %11, %12, %13, %14, %15, "
                    "%16, %17, %18, %19, %20, %21, %22, %23, "
                    "%24, %25, %26, %27, %28, %29, %30, %31, "
                    "%32, %33, %34, %35, %36, %37, %38, %39, "
                    "%40, %41, %42, %43, %44, %45, %46, %47, "
                    "%48, %49, %50, %51, %52, %53, %54, %55, "
                    "%56, %57, %58, %59, %60, %61, %62, %63  "
                "}, %64, %65, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(r[0][0]),  "+f"(r[0][1]),  "+f"(r[0][2]),  "+f"(r[0][3]),
                  "+f"(r[1][0]),  "+f"(r[1][1]),  "+f"(r[1][2]),  "+f"(r[1][3]),
                  "+f"(r[2][0]),  "+f"(r[2][1]),  "+f"(r[2][2]),  "+f"(r[2][3]),
                  "+f"(r[3][0]),  "+f"(r[3][1]),  "+f"(r[3][2]),  "+f"(r[3][3]),
                  "+f"(r[4][0]),  "+f"(r[4][1]),  "+f"(r[4][2]),  "+f"(r[4][3]),
                  "+f"(r[5][0]),  "+f"(r[5][1]),  "+f"(r[5][2]),  "+f"(r[5][3]),
                  "+f"(r[6][0]),  "+f"(r[6][1]),  "+f"(r[6][2]),  "+f"(r[6][3]),
                  "+f"(r[7][0]),  "+f"(r[7][1]),  "+f"(r[7][2]),  "+f"(r[7][3]),
                  "+f"(r[8][0]),  "+f"(r[8][1]),  "+f"(r[8][2]),  "+f"(r[8][3]),
                  "+f"(r[9][0]),  "+f"(r[9][1]),  "+f"(r[9][2]),  "+f"(r[9][3]),
                  "+f"(r[10][0]), "+f"(r[10][1]), "+f"(r[10][2]), "+f"(r[10][3]),
                  "+f"(r[11][0]), "+f"(r[11][1]), "+f"(r[11][2]), "+f"(r[11][3]),
                  "+f"(r[12][0]), "+f"(r[12][1]), "+f"(r[12][2]), "+f"(r[12][3]),
                  "+f"(r[13][0]), "+f"(r[13][1]), "+f"(r[13][2]), "+f"(r[13][3]),
                  "+f"(r[14][0]), "+f"(r[14][1]), "+f"(r[14][2]), "+f"(r[14][3]),
                  "+f"(r[15][0]), "+f"(r[15][1]), "+f"(r[15][2]), "+f"(r[15][3])
                : "l"(desc_a), "l"(desc_b)
            );
        }

        template <const int bm>
        __device__ __forceinline__ void wgmma(float r[bm>>3][4], uint64_t desc_a, uint64_t desc_b) {
            static_assert(bm >= 8 && bm <= 128 && !(bm & 7));
            if constexpr (bm == 8)   mm8  (r, desc_a, desc_b);
            if constexpr (bm == 16)  mm16 (r, desc_a, desc_b);
            if constexpr (bm == 24)  mm24 (r, desc_a, desc_b);
            if constexpr (bm == 32)  mm32 (r, desc_a, desc_b);
            if constexpr (bm == 40)  mm40 (r, desc_a, desc_b);
            if constexpr (bm == 48)  mm48 (r, desc_a, desc_b);
            if constexpr (bm == 56)  mm56 (r, desc_a, desc_b);
            if constexpr (bm == 64)  mm64 (r, desc_a, desc_b);
            if constexpr (bm == 72)  mm72 (r, desc_a, desc_b);
            if constexpr (bm == 80)  mm80 (r, desc_a, desc_b);
            if constexpr (bm == 88)  mm88 (r, desc_a, desc_b);
            if constexpr (bm == 96)  mm96 (r, desc_a, desc_b);
            if constexpr (bm == 104) mm104(r, desc_a, desc_b);
            if constexpr (bm == 112) mm112(r, desc_a, desc_b);
            if constexpr (bm == 120) mm120(r, desc_a, desc_b);
            if constexpr (bm == 128) mm128(r, desc_a, desc_b);
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

    namespace warp_group {
        __device__ __forceinline__ void arrive() {
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        }

        __device__ __forceinline__ void commit_batch() {
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        }

        __device__ __forceinline__ void await() {
            asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
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

        template <const int nb>
        __device__ __forceinline__ void cg256(void *dst, const void *src) {
            asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" :: "r"(dst), "l"(src), "n"(nb));
        }

        template <const int nb>
        __device__ __forceinline__ void cg64(void *dst, const void *src) {
            asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], %2;\n" :: "r"(dst), "l"(src), "n"(nb));
        }

        __device__ __forceinline__ void commit_group() {
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        template <const int n>
        __device__ __forceinline__ void await_group() {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
        }
    }

    namespace barrier {
        __device__ __forceinline__ void init(uint64_t *bar, int threads, int transactions) {
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
                "mbarrier.arrive.expect_tx.release.cta.shared::ctx.b64 _, [%0], %1;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(nb)
            );
        }

        __device__ __forceinline__ void await(uint64_t *bar, int phase) {
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
                "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1 %2};\n"
                :: "r"(mat), "r"(tile[0]), "r"(tile[1])
                : "memory"
            ); else asm volatile(
                "stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1 %2};\n"
                :: "r"(mat), "r"(tile[0]), "r"(tile[1])
                : "memory"
            );
        }

        template <const bool T>
        __device__ __forceinline__ void store4(uint32_t *tile, uint32_t mat) {
            if constexpr (T) asm volatile(
                "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1 %2 %3 %4};\n"
                :: "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
                : "memory"
            ); else asm volatile(
                "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1 %2 %3 %4};\n"
                :: "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
                : "memory"
            );
        }
    }
}

