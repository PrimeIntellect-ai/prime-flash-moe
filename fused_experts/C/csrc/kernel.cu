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

constexpr int TN = 128;
constexpr int TH = 32;
constexpr int TD = 32;
constexpr int TM = 64;

[[nodiscard]] __device__ __forceinline__ static float silu(float x) {
    return x*(1.f/(1.f+__expf(-x)));
}

static __global__ void fused_experts_megakernel(
    const at::BFloat16* __restrict__ x,    // [N, D]
    const at::BFloat16* __restrict__ wgt,  // [E, D, H]
    const at::BFloat16* __restrict__ wut,  // [E, D, H]
    const at::BFloat16* __restrict__ wdt,  // [E, H, M]
    const int32_t* __restrict__ offsets,    // [E+1]
    at::BFloat16* __restrict__ out,        // [N, M]
    int E,
    int D,
    int H,
    int N,
    int M
) {
    int e = blockIdx.x;
    if (e >= E) return;
    int start = e ? offsets[e-1] : 0;
    int end = offsets[e];
    int numel = end - start;
    int tid = threadIdx.x;
    if (tid >= numel) return;
    int ni = start + tid;
    const auto* wg = wgt + e*D*H;
    const auto* wu = wut + e*D*H;
    const auto* wd = wdt + e*H*M;
    for (int m=0; m < M; m += TM) {
        int vm = min(TM, M-m);
        float acc[TM] = {};
        for (int h=0; h < H; h += TH) {
            int vh = min(TH, H-h);
            float gf[TH] = {}, uf[TH] = {};
            for (int d=0; d < D; d += TD) {
                int vd = min(TD, D-d);
                for (int dd=0; dd < TD; ++dd) {
                    if (dd >= vd) break;
                    float xv = __bfloat162float(static_cast<__nv_bfloat16>(x[ni*D + (d+dd)]));
                    for (int hh=0; hh < TH; ++hh) {
                        if (hh >= vh) break;
                        float wgv = __bfloat162float(static_cast<__nv_bfloat16>(wg[(d+dd)*H + (h+hh)]));
                        float wuv = __bfloat162float(static_cast<__nv_bfloat16>(wu[(d+dd)*H + (h+hh)]));
                        gf[hh] += xv*wgv;
                        uf[hh] += xv*wuv;
                    }
                }
            }
            float act[TH];
            #pragma unroll
            for (int hh=0; hh < TH; ++hh) {
                if (hh >= vh) break;
                act[hh] = silu(gf[hh])*uf[hh];
            }
            for (int hh=0; hh < TH; ++hh) {
                if (hh >= vh) break;
                float a = act[hh];
                const auto* wd_row = wd + (h+hh)*M + m;
                #pragma unroll
                for (int mm=0; mm < TM; ++mm) {
                    if (mm >= vm) break;
                    float w = __bfloat162float(static_cast<__nv_bfloat16>(wd_row[mm]));
                    acc[mm] += a*w;
                }
            }
        }
        auto* o_row = out + ni*M + m;
        #pragma unroll
        for (int mm=0; mm < TM; ++mm) {
            if (mm >= vm) break;
            o_row[mm] = static_cast<at::BFloat16>(__float2bfloat16(acc[mm]));
        }
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
    dim3 block {TN, 1, 1};
    dim3 grid {static_cast<unsigned>(E), 1};
    fused_experts_megakernel<<<grid, block>>>(
        x.const_data_ptr<at::BFloat16>(),
        wgt.const_data_ptr<at::BFloat16>(),
        wut.const_data_ptr<at::BFloat16>(),
        wdt.const_data_ptr<at::BFloat16>(),
        offsets.const_data_ptr<int32_t>(),
        out.data_ptr<at::BFloat16>(),
        static_cast<int>(E),
        static_cast<int>(D),
        static_cast<int>(H),
        static_cast<int>(N),
        static_cast<int>(M)
    );
    return out;
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
