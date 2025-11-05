#include <torch/torch.h>

namespace py = pybind11;

[[nodiscard]] static torch::Tensor fused_experts_kernel(
    const torch::Tensor& hidden_states,
    const torch::Tensor& w1,
    const torch::Tensor& w2
) {
    return torch::relu(torch::matmul(torch::matmul(hidden_states, w1), w2));
}

PYBIND11_MODULE(kernel, m) {
    m.def(
        "fused_experts_kernel", &fused_experts_kernel,
        "Compute the MoE up-projection, activation and down-projection fused.",
        py::arg("hidden_states"),
        py::arg("w1"),
        py::arg("w2")
    );
}
