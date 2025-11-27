#include <torch/torch.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

#include <mma.h>
using namespace nvcuda;

namespace py = pybind11;

static_assert(sizeof(__nv_bfloat16) == sizeof(at::BFloat16));
static_assert(alignof(__nv_bfloat16) == alignof(at::BFloat16));

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "CUDA too old, BF16 WMMA requires a recent CUDA version"
#endif

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

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_TILE = WMMA_M*WMMA_N;

[[nodiscard]] __device__ __forceinline__ static float silu(float x) {
    return x*(1.f/(1.f+__expf(-x)));
}

using wmma_acc = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
using wmma_mat_a = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>;
using wmma_mat_b = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>;

static __global__ void fused_experts_megakernel(
    const __nv_bfloat16 *__restrict__ x,    // [N, D]
    const __nv_bfloat16 *__restrict__ wgt,  // [E, D, H]
    const __nv_bfloat16 *__restrict__ wut,  // [E, D, H]
    const __nv_bfloat16 *__restrict__ wdt,  // [E, H, M]
    const int32_t *__restrict__ offsets,    // [E+1]
    __nv_bfloat16 *__restrict__ out,        // [N, M]
    int E,
    int D,
    int H,
    int N,
    int M
) {
    int e = blockIdx.x;
    if (e >= E) return;
    int startE = !e ? 0 : offsets[e-1];
    int endE = offsets[e];
    int nE = endE-startE;
    if (nE <= 0) return;
    int warp_id = threadIdx.x / 32;
    int lane  = threadIdx.x & 31;
    int tileN = blockIdx.y*WMMA_M;
    if (tileN >= nE) return;
    int tileM_base = blockIdx.z*WMMA_N*4;
    int tileM = tileM_base + warp_id*WMMA_N;
    int globalN0 = startE + tileN;

    bool warp_active = tileM < M;

    const auto *wg_e = wgt + e*D*H;
    const auto *wu_e = wut + e*D*H;
    const auto *wd_e = wdt + e*H*M;

    wmma_acc out_frag;
    wmma::fill_fragment(out_frag, .0f);
    if (warp_active) {
        for (int h0=0; h0 < H; h0 += WMMA_N) {
            wmma_acc gate_frag, up_frag;
            wmma::fill_fragment(gate_frag, .0f);
            wmma::fill_fragment(up_frag, .0f);
            for (int d0=0; d0 < D; d0 += WMMA_K) {
                wmma_mat_a a_frag;
                wmma_mat_b bg_frag, bu_frag;
                const __nv_bfloat16 *a_ptr = x + (globalN0*D + d0);
                const __nv_bfloat16 *bg_ptr = wg_e + (d0*H + h0);
                const __nv_bfloat16 *bu_ptr = wu_e + (d0*H + h0);
                wmma::load_matrix_sync(a_frag, a_ptr, D);
                wmma::load_matrix_sync(bg_frag, bg_ptr, H);
                wmma::load_matrix_sync(bu_frag, bu_ptr, H);
                wmma::mma_sync(gate_frag, a_frag, bg_frag, gate_frag);
                wmma::mma_sync(up_frag, a_frag, bu_frag, up_frag);
            }
            wmma_mat_a act_frag;
            #pragma unroll
            for (int i=0; i < gate_frag.num_elements; ++i) {
                float g = gate_frag.x[i];
                float u = up_frag.x[i];
                float a = silu(g)*u;
                act_frag.x[i] = __float2bfloat16(a);
            }
            wmma_mat_b wd_frag;
            const auto *wd_ptr = wd_e + (h0*M + tileM);
            wmma::load_matrix_sync(wd_frag, wd_ptr, M);
            wmma::mma_sync(out_frag, act_frag, wd_frag, out_frag);
        }
    }
    __shared__ float sh_out[4][WMMA_TILE];
    auto *sho = sh_out[warp_id];
    if (warp_active)
        wmma::store_matrix_sync(sho, out_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();
    if (warp_active) {
        for (int mi = lane; mi < WMMA_TILE; mi += 32) {
            int i = mi / WMMA_N;
            int j = mi % WMMA_N;
            int n_idx = globalN0 + i;
            int m_idx = tileM    + j;
            if (n_idx < endE && n_idx < N && m_idx < M) {
                out[n_idx * M + m_idx] = __float2bfloat16(sho[mi]);
            }
        }
    }
}

[[nodiscard]] static torch::Tensor fused_experts_kernel(
    const torch::Tensor &x,
    const torch::Tensor &wgt,
    const torch::Tensor &wut,
    const torch::Tensor &wdt,
    const torch::Tensor &offsets
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

    TORCH_CHECK(D%WMMA_K == 0, "D must be multiple of 16 for WMMA");
    TORCH_CHECK(H%WMMA_N == 0, "H must be multiple of 16 for WMMA");
    TORCH_CHECK(M%WMMA_N == 0, "M must be multiple of 16 for WMMA");
    TORCH_CHECK(N%WMMA_M == 0, "N must be multiple of 16 for WMMA");

    auto offsets_cpu = offsets.to(at::kCPU);
    const int32_t *off = offsets_cpu.data_ptr<int32_t>();

    /*
    int32_t prev=0;
    for (int64_t e=0; e < E; ++e) { // TODO: for debug only, add masking
        int32_t end = off[e];
        int32_t nE  = end - prev;
        TORCH_CHECK(nE%WMMA_M == 0, "per-expert token count must be multiple of 16 for WMMA (expert ", e, " has ", nE, ")");
        prev = end;
    }
    */

    auto out = torch::zeros({N, M}, x.options());

    dim3 block(128, 1, 1); // 4 warps
    dim3 grid(
        E,
        (N + WMMA_M-1) / WMMA_M,
        (M + (WMMA_N*4) - 1) / (WMMA_N*4)
    );
    fused_experts_megakernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(wgt.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(wut.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(wdt.const_data_ptr<at::BFloat16>()),
        offsets.const_data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
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
