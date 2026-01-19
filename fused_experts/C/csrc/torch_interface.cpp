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
        TORCH_CHECK(block_m > 0 && block_m <= 128 && (block_m&7) == 0);
        TORCH_CHECK(x.is_contiguous());
        TORCH_CHECK(w.is_contiguous());
        TORCH_CHECK(w2.is_contiguous());
        TORCH_CHECK(sorted_token_ids.is_contiguous());
        TORCH_CHECK(expert_ids.is_contiguous());
        TORCH_CHECK(num_tokens_post_padded.is_contiguous());
        TORCH_CHECK(topk_weights.is_contiguous());
        TORCH_CHECK(out.is_contiguous());
        TORCH_CHECK(x.dtype() == torch::kFloat16);
        TORCH_CHECK(w.dtype() == torch::kFloat16);
        TORCH_CHECK(w2.dtype() == torch::kFloat16);
        TORCH_CHECK(sorted_token_ids.dtype() == torch::kInt32);
        TORCH_CHECK(expert_ids.dtype() == torch::kInt32);
        TORCH_CHECK(num_tokens_post_padded.dtype() == torch::kInt32);
        TORCH_CHECK(out.dtype() == torch::kFloat16);
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
}

TORCH_LIBRARY_FRAGMENT(prime_fused_moe, m) {
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
    m.impl("fused_moe_bf16", torch::kCUDA, &pi::fused_moe_bf16_torch_stub);
}

PYBIND11_MODULE(prime_fused_moe, m) {}
