#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "vec.cuh"

namespace cg = cooperative_groups;

// https://github.com/LopezCastroRoberto/vllm/blob/028ce6772826a2d227f5761990b52d4fd8425a34/csrc/quantization/fp4/nvfp4_utils.cuh#L326
// Compute SF output offset for swizzled tensor core layout.
// SF layout: [numMTiles, numKTiles, 32, 4, 4]
// Caller must precompute: numKTiles = (numCols + 63) / 64
__device__ __forceinline__ int64_t get_sf_out_offset(
        int mIdx, int kIdx, int32_t numKTiles) {
    // Decompose indices using bitwise ops (all divisors are powers of 2).
    // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    int32_t mTileIdx = mIdx >> 7;         // mIdx / 128
    int32_t outerMIdx = mIdx & 31;        // mIdx % 32
    int32_t innerMIdx = (mIdx >> 5) & 3;  // (mIdx / 32) % 4
    int32_t kTileIdx = kIdx >> 2;         // kIdx / 4
    int32_t innerKIdx = kIdx & 3;         // kIdx % 4

    // Compute global SF offset: mTileIdx * (numKTiles * 512) + kTileIdx * 512 +
    //                           outerMIdx * 16 + innerMIdx * 4 + innerKIdx
    // Use bitwise OR for non-overlapping lower bits.
    int64_t SFOffset = (static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx)
                               << 9 |
                       (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx;

    return SFOffset;
}

// mixed-precision FMA
__device__ __forceinline__ float fma(nv_bfloat16 a, nv_bfloat16 b, float c) {
    __asm ("fma.rn.f32.bf16 %0, %1, %2, %0;\n"
            :"+f"(c)
            :"h"(__bfloat16_as_short(a)), "h"(__bfloat16_as_short(b)));
    return c;
}

__device__ __forceinline__ float fmul(nv_bfloat16 a, nv_bfloat16 b) {
    return fma(a, b, 0.f);
}


__global__ void fused_rmsnorm_residual_forward_kernel(
        __nv_fp8_e4m3* __restrict__ vals, __nv_fp8_e8m0* __restrict__ scales, float* __restrict__ rrms,
        const nv_bfloat16* __restrict__ branch, const nv_bfloat16* __restrict__ residual,
        const nv_bfloat16* __restrict__ weight, float epsilon,
        int N, int C, float inv_c) {
    using bf16x16 = GenericVector<nv_bfloat16, 16>;
    using fp8x16 = GenericVector<__nv_fp8_e4m3, 16>;
    constexpr const int WARP_SIZE = 32;
    assert(blockDim.x == WARP_SIZE);

    const cg::thread_block block = cg::this_thread_block();
    const cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    const cg::thread_block_tile<2> quant_group = cg::tiled_partition<2>(block);
    long idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx >= N) return;

    // adjust pointers to current token
    vals += C * idx;
    branch += C * idx;
    residual += C * idx;

    // rmsnorm is applied to branch; residual is added *after* normalization,
    // i.e. out = rmsnorm(branch) * weight + residual.
    float sum_squared = 0.f;

    for(int c = threadIdx.x * bf16x16::size; c < C; c += WARP_SIZE * bf16x16::size) {
        const bf16x16 in1 = bf16x16::load_cs(branch + c);
        for(int k = 0; k < bf16x16::size; ++k) {
            sum_squared = fma(in1[k], in1[k], sum_squared);
        }
    }

    sum_squared = cg::reduce(warp, sum_squared, cg::plus<float>{}) * inv_c;
    float s = rsqrtf(sum_squared + epsilon);

    for(int c = threadIdx.x * bf16x16::size; c < C; c += WARP_SIZE * bf16x16::size) {
        const bf16x16 in1 = bf16x16::load_cs(branch + c);
        const bf16x16 w = bf16x16::load(weight + c);
        const bf16x16 in2 = bf16x16::load_cs(residual + c);
        bf16x16 res_out;
        nv_bfloat16 local_max = __float2bfloat16(0.f);
        for(int k = 0; k < bf16x16::size; ++k) {
            float n = s * __bfloat162float(in1[k]); // normalized output
            // scale in bf16 (not fp32) to match the reference transformer implementation
            float nw = __bfloat162float(__float2bfloat16(n) * w[k]);
            // the residual add happens after the norm, in the input precision.
            nv_bfloat16 res = __float2bfloat16(nw + __bfloat162float(in2[k]));
            local_max = __hmax(local_max, __habs(res));
            res_out[k] = res;
        }

        nv_bfloat16 group_amax = cg::reduce(quant_group, local_max, cg::greater<nv_bfloat16>{});
        constexpr float one_over_448 = 1.f / 448.f;
        float scale_factor = __bfloat162float(group_amax) * one_over_448;
        __nv_fp8_storage_t sf = __nv_cvt_float_to_e8m0(scale_factor, __NV_SATFINITE, cudaRoundPosInf);
        // by construction, E8M0 can't be zero
        // V = 2^(SF - 127) => for 1.0 / V: SF' = 127 - SF; add 127 for bias
        nv_bfloat16 inv_sf = nv_bfloat16(__nv_cvt_e8m0_to_bf16raw(254-sf));

        fp8x16 quantized;
        for(int k = 0; k < bf16x16::size; ++k) {
            auto fp8_raw = __nv_cvt_bfloat16raw_to_fp8(res_out[k] * inv_sf, __NV_SATFINITE, __NV_E4M3);
            quantized[k] = reinterpret_cast<__nv_fp8_e4m3&>(fp8_raw);
        }

        quantized.store(vals + c);
        if( quant_group.thread_rank() == 0 ) {
            scales[get_sf_out_offset(idx, c / 32, C/128)] = reinterpret_cast<__nv_fp8_e8m0&>(sf);
        }
    }

    // cache the rrms for the backward pass later
    if(warp.thread_rank() == 0) {
        rrms[idx] = s;
    }
}

void fused_rmsnorm_residual_forward(
        __nv_fp8_e4m3* __restrict__ vals, __nv_fp8_e8m0* __restrict__ scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    assert(C % 128 == 0);
    assert(N % 128 == 0);
    const int block_size = 256;
    int block_y = block_size / 32;
    const int grid_size = (N + block_y - 1) / block_y;

    fused_rmsnorm_residual_forward_kernel<<<grid_size, dim3(32, block_y), 0, stream>>>(
            vals, scales, rrms, branch, residual, weight, epsilon, N, C, 1.f/C);
}
