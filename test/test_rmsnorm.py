# test_fused_rmsnorm_residual.py
import pytest
import torch

from your_pkg.ops import fused_rmsnorm_residual  # allocating wrapper

_op = torch.ops.your_ns.fused_rmsnorm_residual

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E4M3_MAX = 448.0
EPS = 1e-6
# (N, C), both % 128 == 0
SHAPES = [(128, 128), (128, 256), (256, 128), (256, 512), (512, 1024), (384, 768)]


# ============================================================================
# reference: unfused rmsnorm -> residual add -> mxfp8 quant.
# numerics mirror the kernel: fp32 rmsnorm accumulate, bf16 normalize before
# the weight multiply, fp32 residual add rounded to bf16, bf16 block amax,
# e8m0 dequant scale with ceil rounding, pow2 inv_sf.
# ============================================================================
def to_blocked(input_matrix) -> torch.Tensor:
    rows, cols = input_matrix.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    padded = input_matrix
    assert (rows, cols) == (padded_rows, padded_cols)
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def unswizzle(scales_flat, N, C):
    """Inverse of to_blocked: swizzled flat e8m0 -> linear (N, C//32)."""
    G = C // 32
    n_row = N // 128
    n_col = G // 4
    x = scales_flat.view(torch.uint8).view(n_row * n_col, 32, 16)
    x = x.view(n_row * n_col, 32, 4, 4).transpose(1, 2).reshape(n_row, n_col, 128, 4)
    x = x.permute(0, 2, 1, 3).reshape(N, G)
    return x.view(torch.float8_e8m0fnu)


def e8m0_ceil_encode(x):
    """fp32 (>=0) -> e8m0 byte, round toward +inf (ceil to next pow2).
    Matches __nv_cvt_float_to_e8m0(x, SATFINITE, cudaRoundPosInf)."""
    m, exp = torch.frexp(x)                       # x = m*2^exp, m in [0.5,1)
    e = torch.where(m == 0.5, exp - 1, exp)       # ceil(log2 x)
    e = torch.where(x == 0, torch.full_like(e, -127), e)
    return (e + 127).clamp(0, 254).to(torch.uint8)


def e8m0_decode(byte):
    return torch.exp2(byte.to(torch.uint8).to(torch.float32) - 127.0)


def ref_rmsnorm(branch, weight, eps):
    """branch (N,C) bf16, weight (C,) bf16 -> (nw bf16 (N,C), rrms fp32 (N,))."""
    assert branch.dtype == torch.bfloat16
    N, C = branch.shape
    b = branch.to(torch.float32)
    ms = (b * b).sum(dim=-1) / C
    s = torch.rsqrt(ms + eps)
    n = (s[:, None] * b).to(torch.bfloat16)       # normalize fp32, cast bf16
    nw = n * weight[None, :]                       # bf16 * bf16 -> bf16
    return nw, s


def ref_residual_add(nw, residual):
    """Add in fp32, round to bf16."""
    return (nw.to(torch.float32) + residual.to(torch.float32)).to(torch.bfloat16)


def ref_quantize(res):
    """res (N,C) bf16 -> (vals e4m3 (N,C), scales_linear e8m0 (N, C//32))."""
    N, C = res.shape
    G = C // 32
    res_g = res.view(N, G, 32)
    amax = res_g.abs().amax(dim=-1).to(torch.float32)
    sf = e8m0_ceil_encode(amax / E4M3_MAX)
    inv_sf = (1.0 / e8m0_decode(sf)).to(torch.bfloat16)   # exact pow2
    vals = (res_g * inv_sf[:, :, None]).to(torch.float8_e4m3fn).view(N, C)
    return vals, sf.view(torch.float8_e8m0fnu)


def ref_pipeline(branch, residual, weight, eps):
    nw, rrms = ref_rmsnorm(branch, weight, eps)
    res = ref_residual_add(nw, residual)
    vals, scales_linear = ref_quantize(res)
    return vals, scales_linear, rrms


def dequant(vals, scales_linear):
    N, C = vals.shape
    G = C // 32
    v = vals.to(torch.float32).view(N, G, 32)
    sc = e8m0_decode(scales_linear.view(torch.uint8))
    return (v * sc[:, :, None]).view(N, C)


# ============================================================================
# helpers
# ============================================================================
def make_inputs(N, C, *, seed=0, scale=1.0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    branch = (torch.randn(N, C, generator=g, device=device, dtype=torch.float32)
              * scale).to(torch.bfloat16)
    residual = (torch.randn(N, C, generator=g, device=device, dtype=torch.float32)
                * scale).to(torch.bfloat16)
    weight = torch.randn(C, generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    return branch, residual, weight


def run_kernel(branch, residual, weight, eps):
    return fused_rmsnorm_residual(branch, residual, weight, eps)


# ============================================================================
# reference internal-consistency (CPU, no kernel). guards the reference
# against drift if someone edits the cast points.
# ============================================================================
@pytest.mark.parametrize("N,C", SHAPES)
def test_ref_rrms_matches_closed_form(N, C):
    branch, _, weight = make_inputs(N, C, device="cpu")
    _, rrms = ref_rmsnorm(branch, weight, EPS)
    b = branch.to(torch.float32)
    expected = torch.rsqrt((b * b).mean(dim=-1) + EPS)
    torch.testing.assert_close(rrms, expected, atol=0, rtol=0)


def test_ref_quant_no_overflow_and_roundtrip():
    torch.manual_seed(1)
    res = (torch.randn(128, 256) * 5).to(torch.bfloat16)
    vals, scales = ref_quantize(res)
    assert vals.to(torch.float32).abs().max() <= E4M3_MAX
    deq = dequant(vals, scales)
    rel = (deq - res.to(torch.float32)).abs() / (res.to(torch.float32).abs() + 1e-30)
    assert rel.median() < 0.07          # e4m3 ~3 mantissa bits


def test_ref_quant_zero_block():
    res = torch.zeros(128, 128, dtype=torch.bfloat16)
    vals, scales = ref_quantize(res)
    assert (vals.to(torch.float32) == 0).all()
    assert torch.isfinite(e8m0_decode(scales.view(torch.uint8))).all()


def test_unswizzle_inverts_to_blocked():
    N, C = 256, 512
    G = C // 32
    x = torch.randint(0, 255, (N, G), dtype=torch.uint8)
    rt = unswizzle(to_blocked(x), N, C).view(torch.uint8)
    torch.testing.assert_close(rt, x, atol=0, rtol=0)


# ============================================================================
# kernel vs reference
# ============================================================================
@requires_cuda
@pytest.mark.parametrize("N,C", SHAPES)
def test_rrms(N, C):
    branch, residual, weight = make_inputs(N, C)
    _, _, rrms = run_kernel(branch, residual, weight, EPS)
    _, ref = ref_rmsnorm(branch, weight, EPS)
    torch.testing.assert_close(rrms, ref, atol=0, rtol=2e-3)   # reduction-order ULP


@requires_cuda
@pytest.mark.parametrize("N,C", SHAPES)
def test_dequant_close(N, C):
    # primary numerical check: reconstructed values agree regardless of
    # occasional e4m3-step / e8m0-boundary flips from ULP-level rrms drift.
    branch, residual, weight = make_inputs(N, C)
    vals, scales, _ = run_kernel(branch, residual, weight, EPS)
    ref_vals, ref_scales, _ = ref_pipeline(branch, residual, weight, EPS)
    deq_k = dequant(vals, unswizzle(scales, N, C))
    deq_r = dequant(ref_vals, ref_scales)
    torch.testing.assert_close(deq_k, deq_r, atol=4e-2, rtol=4e-2)


@requires_cuda
@pytest.mark.parametrize("N,C", SHAPES)
def test_exact_bytes_mostly(N, C):
    branch, residual, weight = make_inputs(N, C)
    vals, scales, _ = run_kernel(branch, residual, weight, EPS)
    ref_vals, ref_scales, _ = ref_pipeline(branch, residual, weight, EPS)

    val_mismatch = (vals.view(torch.uint8) != ref_vals.view(torch.uint8)).float().mean().item()
    assert val_mismatch < 1e-3, f"val byte mismatch {val_mismatch:.2e}"

    sc_r = to_blocked(ref_scales.view(torch.uint8).view(N, C // 32))
    sf_mismatch = (scales.view(torch.uint8).flatten() != sc_r).float().mean().item()
    assert sf_mismatch < 1e-3, f"scale byte mismatch {sf_mismatch:.2e}"


@requires_cuda
def test_scale_swizzle_layout():
    N, C = 256, 512
    branch, residual, weight = make_inputs(N, C, scale=2.0, seed=7)
    vals, scales, _ = run_kernel(branch, residual, weight, EPS)
    _, ref_scales, _ = ref_pipeline(branch, residual, weight, EPS)
    expected = to_blocked(ref_scales.view(torch.uint8).view(N, C // 32))
    mismatch = (scales.view(torch.uint8).flatten() != expected).float().mean().item()
    assert mismatch < 1e-3


@requires_cuda
def test_zero_input():
    N, C = 128, 256
    z = torch.zeros(N, C, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(C, dtype=torch.bfloat16, device="cuda")
    vals, scales, rrms = run_kernel(z, z, weight, EPS)
    assert torch.isfinite(rrms).all()
    assert (vals.to(torch.float32) == 0).all()
    assert torch.isfinite(e8m0_decode(scales.view(torch.uint8))).all()


# ============================================================================
# out-param contract + registration
# ============================================================================
@requires_cuda
def test_preallocated_outputs_written():
    N, C = 256, 512
    branch, residual, weight = make_inputs(N, C)
    vals = torch.empty(N, C, dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty(N * C // 32, dtype=torch.float8_e8m0fnu, device="cuda")
    rrms = torch.empty(N, dtype=torch.float32, device="cuda")
    out = fused_rmsnorm_residual(branch, residual, weight, EPS,
                                 vals=vals, scales=scales, rrms=rrms)
    assert out[0].data_ptr() == vals.data_ptr()
    assert out[1].data_ptr() == scales.data_ptr()
    assert out[2].data_ptr() == rrms.data_ptr()


@requires_cuda
def test_reject_non_bf16():
    branch, residual, weight = make_inputs(128, 128)
    with pytest.raises(RuntimeError, match="bf16"):
        run_kernel(branch.float(), residual, weight, EPS)


@requires_cuda
def test_reject_mismatched_shape():
    branch, residual, weight = make_inputs(128, 128)
    with pytest.raises(RuntimeError):
        run_kernel(branch, residual[:, :64], weight, EPS)


@requires_cuda
def test_reject_noncontiguous():
    branch, residual, weight = make_inputs(128, 128)
    nc = branch.t().contiguous().t()
    with pytest.raises(RuntimeError, match="contiguous"):
        run_kernel(nc, residual, weight, EPS)


@requires_cuda
def test_reject_bad_divisibility():
    branch, residual, weight = make_inputs(128, 128)
    with pytest.raises(RuntimeError):
        run_kernel(branch[:, :120], residual[:, :120], weight[:120], EPS)


@requires_cuda
def test_opcheck():
    N, C = 128, 256
    branch, residual, weight = make_inputs(N, C)
    vals = torch.empty(N, C, dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty(N * C // 32, dtype=torch.float8_e8m0fnu, device="cuda")
    rrms = torch.empty(N, dtype=torch.float32, device="cuda")
    torch.library.opcheck(_op, (branch, residual, weight, EPS, vals, scales, rrms))
