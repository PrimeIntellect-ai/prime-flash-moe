#include <torch/torch.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

namespace py = pybind11;

/*
    gate = x @ wgt
    up = x @ wut
    act = silu(gate) * up
    out = act @ wdt
*/

constexpr int TILE_N = 64;   // tokens per block tile
constexpr int TILE_H = 64;   // hidden per tile
constexpr int TILE_D = 32;   // D chunk
constexpr int TILE_M = 64;   // output per tile
constexpr int THREADS = 256;
constexpr size_t SMEM = (TILE_N*TILE_H*2 + TILE_H*TILE_M) * sizeof(float);

static __global__ void fused_experts_megakernel(
    const __nv_bfloat16* __restrict__ x,    // [N, D]
    const __nv_bfloat16* __restrict__ wgt,  // [E, D, H]
    const __nv_bfloat16* __restrict__ wut,  // [E, D, H]
    const __nv_bfloat16* __restrict__ wdt,  // [E, H, M]
    const int32_t* __restrict__ offsets,    // [E+1]
    __nv_bfloat16* __restrict__ out,        // [N, M]
    int E,
    int D,
    int H,
    int N,
    int M
) {
    // zero out for now
    int idx = blockIdx.x * TILE_N + threadIdx.x;
    int idy = threadIdx.y;
    for (int m = idy; m < M; m += blockDim.y)
        for (int n = threadIdx.x; n < TILE_N; n += blockDim.x) {
            if (idx + n < N)
                out[(idx + n) * M + m] = __float2bfloat16(0.0f);
        }
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

    auto out = torch::zeros({N, M}, x.options());
    int blocks = static_cast<int>(E);
    fused_experts_megakernel<<<blocks, THREADS, SMEM>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(wgt.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(wut.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(wdt.data_ptr<at::BFloat16>()),
        offsets.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        static_cast<int>(E),
        static_cast<int>(D),
        static_cast<int>(H),
        static_cast<int>(N),
        static_cast<int>(M)
    );
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
