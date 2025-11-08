#include <torch/torch.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

namespace py = pybind11;

using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float
>;

static void gemm_bf16(const cutlass::bfloat16_t* A, int M, int K, int lda, const cutlass::bfloat16_t* B, int N, int ldb, cutlass::bfloat16_t* C, int ldc) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        problem_size,
        {A, lda},
        {B, ldb},
        {C, ldc},
        {C, ldc},
        {1.0f, 0.0f}
    );
    auto status = gemm_op(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM failed");
}


[[nodiscard]] static torch::Tensor fused_experts_kernel(
    const torch::Tensor& x,
    const torch::Tensor& wgt,
    const torch::Tensor& wut,
    const torch::Tensor& wdt,
    const torch::Tensor& offsets
) {
    TORCH_CHECK(x.dtype() == at::ScalarType::BFloat16, "x must be BFloat16");
    TORCH_CHECK(wgt.dtype() == at::ScalarType::BFloat16, "wgt must be BFloat16");
    TORCH_CHECK(wut.dtype() == at::ScalarType::BFloat16, "wut must be BFloat16");
    TORCH_CHECK(wdt.dtype() == at::ScalarType::BFloat16, "wdt must be BFloat16");
    TORCH_CHECK(offsets.dtype() == at::ScalarType::Int, "offsets must be Int32");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(wgt.is_cuda(), "wgt must be a CUDA tensor");
    TORCH_CHECK(wut.is_cuda(), "wut must be a CUDA tensor");
    TORCH_CHECK(wdt.is_cuda(), "wdt must be a CUDA tensor");
    TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(wgt.is_contiguous(), "wgt must be contiguous");
    TORCH_CHECK(wut.is_contiguous(), "wut must be contiguous");
    TORCH_CHECK(wdt.is_contiguous(), "wdt must be contiguous");
    TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(wgt.dim() == 3, "wgt must be 3D");
    TORCH_CHECK(wut.dim() == 3, "wut must be 3D");
    TORCH_CHECK(wdt.dim() == 3, "wdt must be 3D");
    TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D");

    int64_t N = x.size(0);
    int64_t D = x.size(1);
    int64_t E = wgt.size(0);
    int64_t H = wgt.size(2);
    int64_t M = wdt.size(2);

    TORCH_CHECK(wgt.size(1) == D, "wgt size mismatch");
    TORCH_CHECK(wut.size(0) == E, "wut size mismatch");
    TORCH_CHECK(wut.size(1) == D, "wut size mismatch");
    TORCH_CHECK(wut.size(2) == H, "wut size mismatch");
    TORCH_CHECK(wdt.size(0) == E, "wdt size mismatch");
    TORCH_CHECK(wdt.size(1) == H, "wdt size mismatch");
    TORCH_CHECK(offsets.size(0) == E, "offsets size mismatch");
    auto out = torch::empty({N, H}, x.options());
    auto offsets_cpu = offsets.to(at::kCPU);
    const int32_t* off = offsets_cpu.data_ptr<int32_t>();
    auto* x_ptr   = reinterpret_cast<const cutlass::bfloat16_t*>(x.data_ptr<at::BFloat16>());
    auto* wgt_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(wgt.data_ptr<at::BFloat16>());
    auto* out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    for (int64_t e = 0; e < E; ++e) {
        int64_t row_start = off[e];
        int64_t row_end = off[e+1];
        int64_t Ne = row_end - row_start;
        if (Ne <= 0) continue;
        const cutlass::bfloat16_t* A = x_ptr + row_start * D;
        const cutlass::bfloat16_t* B = wgt_ptr + e * (D * H);
        cutlass::bfloat16_t* C = out_ptr + row_start * H;
        gemm_bf16(
            A, Ne, D, D,
            B, H,  H,
            C, H
        );
    }

    return out;

/*
    REF

    at::Tensor gate_proj = at::_grouped_mm(
        x,
        wgt,
        offsets,
        std::nullopt,
        at::ScalarType::BFloat16
    );
    at::Tensor up_proj = at::_grouped_mm(
        x,
        wut,
        offsets,
        std::nullopt,
        at::ScalarType::BFloat16
    );
    at::Tensor activated = at::mul(at::silu(gate_proj), up_proj);
    at::Tensor down_proj = at::_grouped_mm(
        activated,
        wdt,
        offsets,
        std::nullopt,
        at::ScalarType::BFloat16
    );
    return down_proj;
    */
}

PYBIND11_MODULE(kernel, m) {
    m.def(
        "fused_experts_kernel", &fused_experts_kernel,
        "Compute the MoE up-projection, activation and down-projection fused.",
        py::arg("x"),       // hidden states             (N,D)
        py::arg("wgt"),     // weight gate transposed    (E,D,H)
        py::arg("wut"),     // weight up transposed      (E,D,H)
        py::arg("wdt"),     // weight down transposed    (E,H,M)
        py::arg("offsets")  // expert offsets            (E)
    );
}
