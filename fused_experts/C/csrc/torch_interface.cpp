#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

namespace pi {
    extern void fused_moe_bf16(
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
    );
}
