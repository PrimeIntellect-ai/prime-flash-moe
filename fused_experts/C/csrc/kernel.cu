#include <torch/torch.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

namespace py = pybind11;

static constexpr auto AT_DTYPE = at::ScalarType::BFloat16;

[[nodiscard]] static torch::Tensor fused_experts_kernel(
    const torch::Tensor& x,
    const torch::Tensor& wgt,
    const torch::Tensor& wut,
    const torch::Tensor& wdt,
    const torch::Tensor& offsets
) {
    at::Tensor gate_proj = at::_grouped_mm(
        x,
        wgt,
        offsets,
        std::nullopt,
        AT_DTYPE
    );
    at::Tensor up_proj = at::_grouped_mm(
        x,
        wut,
        offsets,
        std::nullopt,
        AT_DTYPE
    );
    at::Tensor activated = at::mul(at::silu(gate_proj), up_proj);
    at::Tensor down_proj = at::_grouped_mm(
        activated,
        wdt,
        offsets,
        std::nullopt,
        AT_DTYPE
    );
    return down_proj;
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
