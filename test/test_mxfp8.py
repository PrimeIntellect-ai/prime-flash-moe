import os
import sys

import prime_moe
import torch

from prime_moe import BLOCK, pack_scales_blocked

# Accuracy gates. Both sides run on the same dequantized values, so the residual is
# accumulation order plus the activation requantization before the down projection.
# Calibrated 2026-08-07 on B200: worst observed across the sweep was rel_clamp_10
# 3.3e-4 and rel_clamp_1 3.3e-3; gates leave ~6x headroom. Override without editing
# via PRIME_MOE_REL_CLAMP_10=...
REL_CLAMP_10_MAX = float(os.environ.get('PRIME_MOE_REL_CLAMP_10', 0.002))
REL_CLAMP_1_MAX = float(os.environ.get('PRIME_MOE_REL_CLAMP_1', 0.02))

E4M3_MAX = 448.0
E4M3_EMAX = 8  # 448 == 1.75 * 2**8


def _shared_exponent(amax: torch.Tensor) -> torch.Tensor:
    """Biased e8m0 exponent of the OCP MX shared scale for a block whose absmax is ``amax``."""
    bits = amax.float().contiguous().view(torch.int32)
    return (((bits >> 23) & 0xFF) - E4M3_EMAX).clamp_(0, 254)


def quantize_mx(t: torch.Tensor, block: int = BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension of ``t`` to MXFP8.

    Returns ``(e4m3 values, e8m0 scales)`` where the scales have shape ``t.shape[:-1] + (K // block,)``.
    """
    k = t.shape[-1]
    assert k % block == 0, f'last dim {k} must be a multiple of {block}'
    g = t.float().reshape(*t.shape[:-1], k // block, block)
    e = _shared_exponent(g.abs().amax(-1))
    inv = torch.exp2((127 - e).float())
    q = (g * inv.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.reshape(*t.shape[:-1], k).contiguous(), e.to(torch.uint8).view(torch.float8_e8m0fnu)


def dequantize_mx(q: torch.Tensor, scales: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Inverse of :func:`quantize_mx`, in float32."""
    k = q.shape[-1]
    e = scales.view(torch.uint8).to(torch.int32)
    scale = torch.exp2((e - 127).float())
    g = q.float().reshape(*q.shape[:-1], k // block, block) * scale.unsqueeze(-1)
    return g.reshape(*q.shape[:-1], k)


def moe_naive_from_topk(x, w1, w2, topk_ids, topk_weights):
    """bf16 reference over already dequantized weights/activations."""
    T, K = x.shape
    E, N, K2 = w1.shape
    assert K == K2
    assert w2.shape == (E, K, N // 2)
    top_k = topk_ids.shape[1]
    pair_ids = torch.arange(T * top_k, device=x.device, dtype=torch.int32)
    expert_flat = topk_ids.reshape(-1).to(torch.int32)
    sort_idx = torch.argsort(expert_flat)
    expert_sorted = expert_flat[sort_idx]
    pair_sorted = pair_ids[sort_idx]
    counts = torch.bincount(expert_sorted, minlength=E).to(torch.int64)
    offs = torch.cumsum(counts, dim=0).to(torch.int32)
    tok = pair_sorted // top_k
    x_sorted = x[tok]
    scale_sorted = topk_weights.reshape(-1)[pair_sorted].float()
    h = torch._grouped_mm(x_sorted, w1.transpose(1, 2).contiguous(), offs, out_dtype=torch.bfloat16)
    gate, up = h.chunk(2, dim=-1)
    act = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
    # the kernel quantizes the activations to mxfp8 before the down projection
    act_q, act_sf = quantize_mx(act)
    act = dequantize_mx(act_q, act_sf).to(torch.bfloat16)
    y_sorted = torch._grouped_mm(act, w2.transpose(1, 2).contiguous(), offs, out_dtype=torch.bfloat16)
    y_sorted = (y_sorted.float() * scale_sorted[:, None]).to(torch.bfloat16)
    out = torch.zeros((T, K), device=x.device, dtype=torch.bfloat16)
    out.index_add_(0, tok.long(), y_sorted)
    return out


def moe_align_block_size_simple(topk_ids: torch.Tensor, block_size_m: int, num_experts: int):
    device = topk_ids.device
    T, top_k = topk_ids.shape
    pair_ids = torch.arange(T * top_k, device=device, dtype=torch.int32)
    expert_flat = topk_ids.reshape(-1).to(torch.int32)
    sort_idx = torch.argsort(expert_flat)
    expert_sorted = expert_flat[sort_idx]
    pair_sorted = pair_ids[sort_idx]
    counts = torch.bincount(expert_sorted, minlength=num_experts)
    padded_counts = ((counts + block_size_m - 1) // block_size_m) * block_size_m
    total_padded = int(padded_counts.sum().item())
    sorted_token_ids = torch.empty((total_padded,), device=device, dtype=torch.int32)
    num_blocks = int((padded_counts // block_size_m).sum().item())
    expert_ids = torch.empty((num_blocks,), device=device, dtype=torch.int32)
    write_ptr = block_ptr = read_ptr = 0
    for e in range(num_experts):
        c = int(counts[e].item())
        pc = int(padded_counts[e].item())
        if c > 0:
            sorted_token_ids[write_ptr : write_ptr + c] = pair_sorted[read_ptr : read_ptr + c]
        if pc > c:
            sorted_token_ids[write_ptr + c : write_ptr + pc] = -1
        nb = pc // block_size_m
        if nb > 0:
            expert_ids[block_ptr : block_ptr + nb] = e
            block_ptr += nb
        write_ptr += pc
        read_ptr += c
    num_tokens_post_padded = torch.tensor([total_padded], device=device, dtype=torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def generate_topk_ids(num_experts, num_tokens, top_k, balancedness=1.0):
    uniform = torch.ones(num_experts, device='cuda:0') / num_experts
    skewed = torch.zeros(num_experts, device='cuda:0')
    skewed[0] = 1.0
    probs = balancedness * uniform + (1 - balancedness) * skewed
    topk_ids = torch.multinomial(probs, num_tokens * top_k, replacement=True)
    return topk_ids.view(num_tokens, top_k)


def test_cfg(num_tokens, K, E, N, top_k, block_m, bn, wn, stages):
    x = torch.empty((num_tokens, K), dtype=torch.bfloat16, device='cuda:0').normal_(mean=0, std=0.05)
    # realistic init scale: with std-1 weights the outputs reach rms ~100, where mxfp8's
    # ~1% inherent noise makes the rel gates unachievable for ANY implementation (two
    # pure-torch references fail them against each other)
    w1 = torch.randn((E, N, K), dtype=torch.bfloat16, device='cuda:0') * 0.05
    w2 = torch.randn((E, K, N // 2), dtype=torch.bfloat16, device='cuda:0') * 0.05

    x_q, x_sf = quantize_mx(x)                              # row major (M, K/32) scales
    w1_q, w1_sf = quantize_mx(w1)
    w2_q, w2_sf = quantize_mx(w2)

    topk_ids = generate_topk_ids(E - 1, num_tokens, top_k - 1)
    last = torch.full((num_tokens, 1), E - 1, device='cuda:0', dtype=topk_ids.dtype)
    topk_ids = torch.hstack((topk_ids, last))
    topk_weights = torch.nn.functional.softmax(
        torch.randn((num_tokens, top_k), dtype=torch.float32, device='cuda:0'), dim=-1
    )
    out = torch.zeros((num_tokens, K), dtype=torch.bfloat16, device='cuda:0')
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_simple(topk_ids, block_m, E)

    torch.ops.prime_moe.fused_moe_mxfp8(
        x_q,
        x_sf,
        w1_q,
        pack_scales_blocked(w1_sf),
        w2_q,
        pack_scales_blocked(w2_sf),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        top_k,
        block_m,
        bn,
        wn,
        stages,
        1,      # bpc
        False,  # split: the single fused kernel, `out` pre-zeroed above
    )
    torch.cuda.synchronize()

    # reference runs on the same quantized values, dequantized back to bf16
    out_naive = moe_naive_from_topk(
        dequantize_mx(x_q, x_sf).to(torch.bfloat16),
        dequantize_mx(w1_q, w1_sf).to(torch.bfloat16),
        dequantize_mx(w2_q, w2_sf).to(torch.bfloat16),
        topk_ids,
        topk_weights,
    )
    torch.cuda.synchronize()

    diff = (out.float() - out_naive.float()).abs()
    ref = out_naive.float().abs()
    fused_finite = torch.isfinite(out.float()).all().item()
    naive_finite = torch.isfinite(out_naive.float()).all().item()
    rel_1 = (diff / ref.clamp_min(1.0)).max().item()
    rel_10 = (diff / ref.clamp_min(10.0)).max().item()

    cfg = f'T={num_tokens} K={K} E={E} N={N} bn={bn} wn={wn} stages={stages}'
    print(cfg)
    print('  finite fused/naive:', fused_finite, naive_finite)
    print('  max_abs', diff.max().item(), 'mean_abs', diff.mean().item())
    print('  rel_clamp_1', rel_1)
    print('  rel_clamp_10', rel_10)

    failures = []
    if not fused_finite:
        failures.append('fused output has non finite values')
    if not naive_finite:
        failures.append('reference output has non finite values')
    if rel_10 > REL_CLAMP_10_MAX:
        failures.append(f'rel_clamp_10 {rel_10:.4g} > {REL_CLAMP_10_MAX:.4g}')
    if rel_1 > REL_CLAMP_1_MAX:
        failures.append(f'rel_clamp_1 {rel_1:.4g} > {REL_CLAMP_1_MAX:.4g}')
    if failures:
        print('  FAIL:', '; '.join(failures))
        return [f'{cfg}: {f}' for f in failures]
    return []


def test_public_wrapper():
    """The prime_moe.fused_moe_mxfp8 wrapper must agree with the raw op and return `out`."""
    T, K, E, N, top_k, block_m = 128, 4096, 8, 256, 4, 128
    x = torch.empty((T, K), dtype=torch.bfloat16, device='cuda:0').normal_(mean=0, std=0.05)
    w1 = torch.randn((E, N, K), dtype=torch.bfloat16, device='cuda:0') * 0.05
    w2 = torch.randn((E, K, N // 2), dtype=torch.bfloat16, device='cuda:0') * 0.05
    x_q, x_sf = quantize_mx(x)
    w1_q, w1_sf = quantize_mx(w1)
    w2_q, w2_sf = quantize_mx(w2)
    topk_ids = generate_topk_ids(E, T, top_k)
    topk_weights = torch.nn.functional.softmax(
        torch.randn((T, top_k), dtype=torch.float32, device='cuda:0'), dim=-1
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_simple(topk_ids, block_m, E)

    args = (
        x_q, x_sf, w1_q, pack_scales_blocked(w1_sf), w2_q, pack_scales_blocked(w2_sf),
        sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
    )
    tail = (top_k, block_m, 32, 8, 1, 1, False)   # ..., stages, bpc, split

    raw = torch.zeros((T, K), dtype=torch.bfloat16, device='cuda:0')
    torch.ops.prime_moe.fused_moe_mxfp8(*args, raw, *tail)
    wrapped_out = torch.zeros((T, K), dtype=torch.bfloat16, device='cuda:0')
    returned = prime_moe.fused_moe_mxfp8(*args, wrapped_out, *tail)
    torch.cuda.synchronize()

    failures = []
    if returned is not wrapped_out:
        failures.append(f'wrapper returned {type(returned).__name__}, expected the `out` tensor')
    # the split-K epilogue accumulates with atomics, so two runs of the same op are not
    # bitwise identical -- compare with a tolerance instead of torch.equal
    wrap_diff = (raw.float() - wrapped_out.float()).abs().max().item()
    if not torch.allclose(raw.float(), wrapped_out.float(), rtol=1e-2, atol=2e-3):
        failures.append(f'wrapper result differs from the raw op (max_abs {wrap_diff:.4g})')
    print('wrapper:', 'ok' if not failures else 'FAIL: ' + '; '.join(failures))
    return [f'wrapper: {f}' for f in failures]


if __name__ == '__main__':
    # mxfp8 path: N and K must be multiples of 256, block_m is 128, bpc is 1
    batch_sizes = [8, 32, 128, 256]
    n_vals = [256, 512]
    k_vals = [7168, 4096]
    e_vals = [257, 8, 37]
    top_k = 9
    block_m = 128
    configs = [(block_m, bn, wn, stages) for bn, wn in [(32, 8), (64, 4)] for stages in (1, 2)]

    failures, ncfg, nbad = [], 0, 0
    for E in e_vals:
        for N in n_vals:
            for K in k_vals:
                for num_tokens in batch_sizes:
                    for block_m, bn, wn, stages in configs:
                        f = test_cfg(num_tokens, K, E, N, top_k, block_m, bn, wn, stages)
                        failures += f
                        ncfg += 1
                        nbad += bool(f)
    f = test_public_wrapper()
    failures += f
    ncfg += 1
    nbad += bool(f)

    print(f'\n{ncfg - nbad}/{ncfg} configs passed')
    if failures:
        print(f'{len(failures)} failure(s):')
        for msg in failures:
            print('  -', msg)
        sys.exit(1)
