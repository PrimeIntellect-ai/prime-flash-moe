"""Benchmark the fused MoE kernels against torch's grouped GEMM path and a naive loop.

The primary baseline is the realistic "no fused kernel" option: a single grouped GEMM call
per projection (``torch.nn.functional.grouped_mm`` for bf16, ``scaled_grouped_mm`` for
mxfp8), not a per-expert python loop. That is what you would actually reach for without a
custom kernel, so it is the fair ruler for "is the fused kernel worth it":

    gate_up = grouped_mm(x, w1, offs)         # one launch across every expert
    gate, up = gate_up.chunk(2, dim=-1)
    act = F.silu(gate) * up
    expert_out = grouped_mm(act, w2, offs)    # one launch across every expert
    out = routing_weight * expert_out

A second, naive baseline is also timed: the textbook per-expert python loop, one plain
GEMM pair per expert (``x[rows] @ w1[e].T`` → SwiGLU → ``act @ w2[e].T``; ``scaled_mm``
per expert for mxfp8). It shows what the fused kernel buys over the simplest possible
implementation, not just over torch's best grouped path.

Routing, sorting, the gather and the final scatter all happen once during setup and sit
outside the timed region for both sides, so neither implementation is charged for
bookkeeping that isn't part of its own kernel. Both sides are timed with CUDA graph
replays by default so the numbers are GPU time, not python dispatch time.

Two dtypes are reported. bf16 compares against ``torch.nn.functional.grouped_mm``; mxfp8
compares against ``torch.nn.functional.scaled_grouped_mm`` with OCP MX (1x32 e8m0) block
scales on both operands, and the baseline requantizes the intermediate activation exactly
like the kernel does. mxfp8 additionally emits a third, "no requant" comparison where the
torch baseline's intermediate requantize/pack is moved out of the timed region (the second
GEMM reads a pre-quantized activation) while the fused kernel still requantizes on chip —
deliberately unfair, but it isolates the raw GEMM-side speedup from the fusion win of not
round-tripping the requant through global memory. mxfp8 groups must start on a 128 row
boundary (the MX swizzle tile),
so each expert's slice of the grouped call is padded up to a multiple of 128 rows; bf16
has no such constraint and runs on the raw, unpadded per-expert counts.

Run it as-is on any CUDA box with prime_moe installed to get both plots for your machine:

    python benchmark/benchmark.py
"""

import argparse
import statistics
import sys
from collections.abc import Callable

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker
import torch
import torch.nn.functional as F
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

try:
    import prime_moe
    from prime_moe import pack_scales_blocked

    HAVE_KERNEL = True
    KERNEL_IMPORT_ERROR = None
except Exception as exc:  # the extension may not be built for this box
    prime_moe = None
    HAVE_KERNEL = False
    KERNEL_IMPORT_ERROR = exc

    def pack_scales_blocked(scales: torch.Tensor) -> torch.Tensor:
        u = scales.view(torch.uint8)
        lead = u.shape[:-2]
        mn, kb = u.shape[-2:]
        assert mn % 128 == 0, f'MN ({mn}) must be a multiple of 128'
        assert kb % 4 == 0, f'K ({kb * 32}) must be a multiple of 128'
        t = u.reshape(-1, mn // 128, 4, 32, kb // 4, 4)
        t = t.permute(0, 1, 4, 3, 2, 5).contiguous()
        return t.reshape(*lead, -1).view(torch.float8_e8m0fnu)


BLOCK = 32
MX_ALIGN = 128  # scaled_grouped_mm groups must start on a swizzle tile boundary
E4M3_MAX = 448.0
E4M3_EMAX = 8  # 448 == 1.75 * 2**8
MX = ScalingType.BlockWise1x32
SWZ = SwizzleType.SWIZZLE_32_4_4

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SOFT = '#52514e'
GRID = '#e4e3de'
C_TORCH = '#2a78d6'
C_FUSED = '#eb6834'
C_NAIVE = '#1baf7a'

def _shared_exponent(amax: torch.Tensor) -> torch.Tensor:
    bits = amax.float().contiguous().view(torch.int32)
    return (((bits >> 23) & 0xFF) - E4M3_EMAX).clamp_(0, 254)


def quantize_mx(t: torch.Tensor, block: int = BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    k = t.shape[-1]
    assert k % block == 0, f'last dim {k} must be a multiple of {block}'
    g = t.float().reshape(*t.shape[:-1], k // block, block)
    e = _shared_exponent(g.abs().amax(-1))
    inv = torch.exp2((127 - e).float())
    q = (g * inv.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.reshape(*t.shape[:-1], k).contiguous(), e.to(torch.uint8).view(torch.float8_e8m0fnu)


def dequantize_mx(q: torch.Tensor, scales: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    k = q.shape[-1]
    e = scales.view(torch.uint8).to(torch.int32)
    scale = torch.exp2((e - 127).float())
    g = q.float().reshape(*q.shape[:-1], k // block, block) * scale.unsqueeze(-1)
    return g.reshape(*q.shape[:-1], k)


def generate_topk_ids(num_experts: int, num_tokens: int, top_k: int, balancedness: float = 1.0) -> torch.Tensor:
    uniform = torch.ones(num_experts, device='cuda') / num_experts
    skewed = torch.zeros(num_experts, device='cuda')
    skewed[0] = 1.0
    probs = balancedness * uniform + (1.0 - balancedness) * skewed
    return torch.multinomial(probs, num_tokens * top_k, replacement=True).view(num_tokens, top_k).to(torch.int32)


def make_routing(topk_ids: torch.Tensor, block_m: int, num_experts: int, bpc: int = 1) -> dict:
    """Sort the (token, slot) pairs by expert and build the kernel's block schedule.

    Padded to a multiple of ``block_m * bpc`` per expert, matching ``torch.ops.prime_moe.moe_align``.
    With bpc>1 a CTA covers ``bpc`` consecutive blocks under one ``expert_idx`` lookup; padding to
    only ``block_m`` lets an expert's last (odd) block pair up with the next expert's first block
    in the same CTA, silently computing that half with the wrong expert's weights.
    """
    T, top_k = topk_ids.shape
    pair_ids = torch.arange(T * top_k, device='cuda', dtype=torch.int32)
    expert_flat = topk_ids.reshape(-1).to(torch.int32)
    sort_idx = torch.argsort(expert_flat)
    expert_sorted = expert_flat[sort_idx]
    pair_sorted = pair_ids[sort_idx]
    counts = torch.bincount(expert_sorted, minlength=num_experts).to(torch.int32)
    pad_to = block_m * bpc
    padded_counts = ((counts + pad_to - 1) // pad_to) * pad_to
    counts_cpu = counts.tolist()
    padded_cpu = padded_counts.tolist()
    total_padded = sum(padded_cpu)
    num_blocks = sum(pc // block_m for pc in padded_cpu)
    sorted_token_ids = torch.empty((total_padded,), device='cuda', dtype=torch.int32)
    expert_ids = torch.empty((num_blocks,), device='cuda', dtype=torch.int32)
    read_ptr = write_ptr = block_ptr = 0
    for e in range(num_experts):
        c, pc = counts_cpu[e], padded_cpu[e]
        if c:
            sorted_token_ids[write_ptr : write_ptr + c] = pair_sorted[read_ptr : read_ptr + c]
        if pc > c:
            sorted_token_ids[write_ptr + c : write_ptr + pc] = -1
        nb = pc // block_m
        if nb:
            expert_ids[block_ptr : block_ptr + nb] = e
            block_ptr += nb
        read_ptr += c
        write_ptr += pc
    return {
        'pair_sorted': pair_sorted,
        'counts': counts_cpu,
        'sorted_token_ids': sorted_token_ids,
        'expert_ids': expert_ids,
        'num_tokens_post_padded': torch.tensor([total_padded], device='cuda', dtype=torch.int32),
    }


def build_grouped_bf16_inputs(x: torch.Tensor, topk_weights: torch.Tensor, routing: dict, top_k: int) -> dict:
    pair_sorted = routing['pair_sorted']
    tok = (pair_sorted // top_k).long()
    route = pair_sorted % top_k
    offs = torch.cumsum(torch.tensor(routing['counts'], device='cuda', dtype=torch.int32), dim=0).to(torch.int32)
    return {
        'xg': x[tok],
        'wg': topk_weights[tok, route.long()].to(torch.bfloat16),
        'offs': offs,
        'tok': tok,
    }


def build_grouped_mxfp8_inputs(x: torch.Tensor, topk_weights: torch.Tensor, routing: dict, top_k: int) -> dict:
    pair_sorted = routing['pair_sorted']
    counts = routing['counts']
    padded_counts = [((c + MX_ALIGN - 1) // MX_ALIGN) * MX_ALIGN for c in counts]
    total_padded = sum(padded_counts)
    idx_pair = torch.full((total_padded,), -1, device='cuda', dtype=torch.int32)
    read_ptr = write_ptr = 0
    for c, pc in zip(counts, padded_counts):
        if c:
            idx_pair[write_ptr : write_ptr + c] = pair_sorted[read_ptr : read_ptr + c]
        read_ptr += c
        write_ptr += pc
    valid = idx_pair >= 0
    safe_pair = torch.where(valid, idx_pair, torch.zeros_like(idx_pair)).long()
    tok = safe_pair // top_k
    route = safe_pair % top_k
    offs = torch.cumsum(torch.tensor(padded_counts, device='cuda', dtype=torch.int32), dim=0).to(torch.int32)
    return {
        'xg': x[tok],
        'wg': (topk_weights[tok, route] * valid.to(torch.float32)).to(torch.bfloat16),
        'offs': offs,
        'tok': tok,
        'valid': valid,
        'padded_counts': padded_counts,
    }


def expert_bounds(counts: list[int]) -> list[tuple[int, int]]:
    bounds, ptr = [], 0
    for c in counts:
        bounds.append((ptr, ptr + c))
        ptr += c
    return bounds

def torch_naive_bf16(xg: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, bounds: list[tuple[int, int]], wg: torch.Tensor) -> torch.Tensor:
    out = torch.empty((xg.shape[0], w2.shape[1]), dtype=torch.bfloat16, device=xg.device)
    for e, (s, t) in enumerate(bounds):
        if s == t:
            continue
        gate_up = xg[s:t] @ w1[e].T
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        out[s:t] = wg[s:t].unsqueeze(1) * (act @ w2[e].T)
    return out


def torch_naive_mxfp8(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    w1q: torch.Tensor,
    w1sf: torch.Tensor,
    w2q: torch.Tensor,
    w2sf: torch.Tensor,
    bounds: list[tuple[int, int]],
    wg: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty((xq.shape[0], w2q.shape[1]), dtype=torch.bfloat16, device=xq.device)
    for e, (s, t) in enumerate(bounds):
        if s == t:
            continue
        gate_up = scaled_mm(xq[s:t], w1q[e].t(), xsf[s:t], MX, w1sf[e], MX, swizzle_a=SWZ, swizzle_b=SWZ, output_dtype=torch.bfloat16)
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        act_q, act_sf = quantize_mx(act)
        act_sf_packed = pack_scales_blocked(act_sf).view(act.shape[0], -1)
        expert_out = scaled_mm(act_q, w2q[e].t(), act_sf_packed, MX, w2sf[e], MX, swizzle_a=SWZ, swizzle_b=SWZ, output_dtype=torch.bfloat16)
        out[s:t] = wg[s:t].unsqueeze(1) * expert_out
    return out


def torch_grouped_bf16(xg: torch.Tensor, w1t: torch.Tensor, w2t: torch.Tensor, offs: torch.Tensor, wg: torch.Tensor) -> torch.Tensor:
    gate_up = F.grouped_mm(xg, w1t, offs=offs, out_dtype=torch.bfloat16)
    gate, up = gate_up.chunk(2, dim=-1)
    act = F.silu(gate) * up
    expert_out = F.grouped_mm(act, w2t, offs=offs, out_dtype=torch.bfloat16)
    return wg.unsqueeze(1) * expert_out


def torch_grouped_mxfp8(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    w1qt: torch.Tensor,
    w1sf: torch.Tensor,
    w2qt: torch.Tensor,
    w2sf: torch.Tensor,
    offs: torch.Tensor,
    wg: torch.Tensor,
) -> torch.Tensor:
    gate_up = F.scaled_grouped_mm(xq, w1qt, xsf, MX, w1sf, MX, swizzle_a=SWZ, swizzle_b=SWZ, offs=offs, output_dtype=torch.bfloat16)
    gate, up = gate_up.chunk(2, dim=-1)
    act = F.silu(gate) * up
    act_q, act_sf = quantize_mx(act)
    act_sf_packed = pack_scales_blocked(act_sf).view(act.shape[0], -1)
    expert_out = F.scaled_grouped_mm(act_q, w2qt, act_sf_packed, MX, w2sf, MX, swizzle_a=SWZ, swizzle_b=SWZ, offs=offs, output_dtype=torch.bfloat16)
    return wg.unsqueeze(1) * expert_out


def torch_grouped_mxfp8_noreq(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    w1qt: torch.Tensor,
    w1sf: torch.Tensor,
    act_q: torch.Tensor,
    act_sf: torch.Tensor,
    w2qt: torch.Tensor,
    w2sf: torch.Tensor,
    offs: torch.Tensor,
    wg: torch.Tensor,
) -> torch.Tensor:
    """The grouped mxfp8 baseline with the intermediate requant taken out of the timed path.

    The second GEMM reads a pre-quantized activation prepared during setup, so the timed
    region covers the two grouped GEMMs and the SwiGLU but not ``quantize_mx``/``pack``.
    The fused kernel still requantizes on chip, so this comparison is deliberately unfair
    to it — it isolates the raw GEMM-side speedup.
    """
    gate_up = F.scaled_grouped_mm(xq, w1qt, xsf, MX, w1sf, MX, swizzle_a=SWZ, swizzle_b=SWZ, offs=offs, output_dtype=torch.bfloat16)
    gate, up = gate_up.chunk(2, dim=-1)
    F.silu(gate) * up  # noqa: B018 — the activation math stays in the timing, only its requant is skipped
    expert_out = F.scaled_grouped_mm(act_q, w2qt, act_sf, MX, w2sf, MX, swizzle_a=SWZ, swizzle_b=SWZ, offs=offs, output_dtype=torch.bfloat16)
    return wg.unsqueeze(1) * expert_out


def fused_bf16(state: dict) -> torch.Tensor:
    if not state['split']:
        state['out'].zero_()
    torch.ops.prime_moe.fused_moe_bf16(
        state['x'],
        state['w1'],
        state['w2'],
        state['sorted_token_ids'],
        state['expert_ids'],
        state['num_tokens_post_padded'],
        state['topk_weights'],
        state['out'],
        state['top_k'],
        state['block_m'],
        state['bn'],
        state['wn'],
        state['stages'],
        state['bpc'],
        state['cpc'],
        state['split'],
    )
    return state['out']


def fused_mxfp8(state: dict) -> torch.Tensor:
    if not state['split']:
        state['out'].zero_()
    torch.ops.prime_moe.fused_moe_mxfp8(
        state['x'],
        state['x_sf'],
        state['w1'],
        state['w1_sf'],
        state['w2'],
        state['w2_sf'],
        state['sorted_token_ids'],
        state['expert_ids'],
        state['num_tokens_post_padded'],
        state['topk_weights'],
        state['out'],
        state['top_k'],
        state['block_m'],
        state['bn'],
        state['wn'],
        state['stages'],
        state['bpc'],
        state['split'],
    )
    return state['out']


def graphed(fn: Callable[[], object], warmup: int = 3) -> Callable[[], object] | None:
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g.replay
    except Exception:
        torch.cuda.synchronize()
        return None


def bench_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def approx_flops(num_tokens: int, top_k: int, K: int, N: int) -> float:
    pairs = num_tokens * top_k
    return 2.0 * pairs * K * N + 2.0 * pairs * (N // 2) * K

def run_one(dtype: str, num_tokens: int, args: argparse.Namespace) -> dict:
    E, K, N, top_k = args.experts, args.hidden, 2 * args.inter, args.top_k
    block_m = args.block_m
    split = args.split
    # mxfp8 runs 1 CTA/SM (tmem-capped), so it profits from a deeper ring; bf16 needs
    # the shallow ring to keep 2 CTAs/SM
    stages = args.stages if args.stages is not None else (4 if dtype == 'mxfp8' and split else 2)
    torch.manual_seed(args.seed)

    x = torch.empty((num_tokens, K), dtype=torch.bfloat16, device='cuda').normal_(0, 0.05)
    w1 = (torch.randn((E, N, K), dtype=torch.bfloat16, device='cuda') * 0.05).contiguous()
    w2 = (torch.randn((E, K, N // 2), dtype=torch.bfloat16, device='cuda') * 0.05).contiguous()
    topk_ids = generate_topk_ids(E, num_tokens, top_k, args.balancedness)
    topk_weights = F.softmax(torch.randn((num_tokens, top_k), dtype=torch.float32, device='cuda'), dim=-1)

    # every block_m*bpc row group has to be single-expert, so each expert's run is padded
    # up to that boundary
    routing = make_routing(topk_ids, block_m, E, args.bpc)
    out = torch.zeros_like(x)
    state = {
        'sorted_token_ids': routing['sorted_token_ids'],
        'expert_ids': routing['expert_ids'],
        'num_tokens_post_padded': routing['num_tokens_post_padded'],
        'topk_weights': topk_weights,
        'out': out,
        'top_k': top_k,
        'block_m': block_m,
        'bn': args.bn,
        'wn': args.wn,
        'stages': stages,
        'bpc': args.bpc,
        'cpc': args.cpc,
        'split': split,
    }

    if dtype == 'bf16':
        state |= {'x': x, 'w1': w1, 'w2': w2}
        g = build_grouped_bf16_inputs(x, topk_weights, routing, top_k)
        w1t, w2t = w1.transpose(1, 2), w2.transpose(1, 2)
        bounds = expert_bounds(routing['counts'])
        baseline_fn = lambda: torch_grouped_bf16(g['xg'], w1t, w2t, g['offs'], g['wg'])
        naive_fn = lambda: torch_naive_bf16(g['xg'], w1, w2, bounds, g['wg'])
        fused_fn = lambda: fused_bf16(state)
        noreq_fn = None  # bf16 has no intermediate quantization to strip
        scatter_tok, scatter_mask = g['tok'], None
    else:
        x_q, x_sf = quantize_mx(x)
        w1_q, w1_sf = quantize_mx(w1)
        w2_q, w2_sf = quantize_mx(w2)
        w1_sf_e = pack_scales_blocked(w1_sf).view(E, N, K // BLOCK)
        w2_sf_e = pack_scales_blocked(w2_sf).view(E, K, (N // 2) // BLOCK)
        state |= {'x': x_q, 'x_sf': x_sf, 'w1': w1_q, 'w1_sf': w1_sf_e, 'w2': w2_q, 'w2_sf': w2_sf_e}

        g = build_grouped_mxfp8_inputs(x, topk_weights, routing, top_k)
        gxq, gxsf = quantize_mx(g['xg'])
        gxsf_packed = pack_scales_blocked(gxsf).view(gxq.shape[0], -1)
        w1_sf_flat = w1_sf_e.reshape(E, -1)
        w2_sf_flat = w2_sf_e.reshape(E, -1)
        w1_qt, w2_qt = w1_q.transpose(1, 2), w2_q.transpose(1, 2)
        bounds = expert_bounds(g['padded_counts'])
        baseline_fn = lambda: torch_grouped_mxfp8(gxq, gxsf_packed, w1_qt, w1_sf_flat, w2_qt, w2_sf_flat, g['offs'], g['wg'])
        naive_fn = lambda: torch_naive_mxfp8(gxq, gxsf_packed, w1_q, w1_sf_flat, w2_q, w2_sf_flat, bounds, g['wg'])
        fused_fn = lambda: fused_mxfp8(state)
        # prepare a quantized activation once so the no-requant baseline can skip the
        # quantize_mx/pack step inside the timed loop (values are static, timing-only)
        gate_up = F.scaled_grouped_mm(
            gxq, w1_qt, gxsf_packed, MX, w1_sf_flat, MX, swizzle_a=SWZ, swizzle_b=SWZ, offs=g['offs'], output_dtype=torch.bfloat16
        )
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        act_q_pre, act_sf_pre = quantize_mx(act)
        act_sf_pre_packed = pack_scales_blocked(act_sf_pre).view(act.shape[0], -1)
        del gate_up, gate, up, act, act_sf_pre
        noreq_fn = lambda: torch_grouped_mxfp8_noreq(
            gxq, gxsf_packed, w1_qt, w1_sf_flat, act_q_pre, act_sf_pre_packed, w2_qt, w2_sf_flat, g['offs'], g['wg']
        )
        scatter_tok, scatter_mask = g['tok'], g['valid']

    max_abs = rel = naive_max_abs = float('nan')
    if args.check:
        base_out = baseline_fn()
        if not args.skip_naive:
            # naive and grouped share the same row layout, so they compare directly
            naive_out = naive_fn()
            ndiff = (base_out.float() - naive_out.float()).abs()
            naive_max_abs = (ndiff[scatter_mask] if scatter_mask is not None else ndiff).max().item()
            del naive_out
        if not args.skip_fused:
            ref = torch.zeros_like(x)
            if scatter_mask is None:
                ref.index_add_(0, scatter_tok, base_out.to(x.dtype))
            else:
                ref.index_add_(0, scatter_tok[scatter_mask], base_out[scatter_mask].to(x.dtype))
            got = fused_fn().clone()
            torch.cuda.synchronize()
            diff = (ref.float() - got.float()).abs()
            max_abs = diff.max().item()
            rel = (diff / ref.float().abs().clamp_min(1.0)).max().item()

    baseline_run = baseline_fn
    naive_run = naive_fn
    fused_run = fused_fn
    noreq_run = noreq_fn
    if args.cuda_graphs:
        g_ = graphed(baseline_fn)
        baseline_run = g_ or baseline_fn
        if g_ is None:
            print('  note: CUDA graph capture failed for the torch baseline, timing it eagerly')
        if noreq_fn is not None:
            g_ = graphed(noreq_fn)
            noreq_run = g_ or noreq_fn
            if g_ is None:
                print('  note: CUDA graph capture failed for the no-requant baseline, timing it eagerly')
        if not args.skip_naive:
            g_ = graphed(naive_fn)
            naive_run = g_ or naive_fn
            if g_ is None:
                print('  note: CUDA graph capture failed for the naive loop, timing it eagerly')
        if not args.skip_fused:
            g_ = graphed(fused_fn)
            fused_run = g_ or fused_fn
            if g_ is None:
                print('  note: CUDA graph capture failed for the fused kernel, timing it eagerly')

    baseline_ms = bench_cuda(baseline_run, args.warmup, args.iters)
    noreq_ms = float('nan') if noreq_run is None else bench_cuda(noreq_run, args.warmup, args.iters)
    naive_ms = float('nan') if args.skip_naive else bench_cuda(naive_run, args.warmup, args.iters)
    fused_ms = float('nan') if args.skip_fused else bench_cuda(fused_run, args.warmup, args.iters)

    flops = approx_flops(num_tokens, top_k, K, N)
    row = {
        'dtype': dtype,
        'tokens': num_tokens,
        'E': E,
        'K': K,
        'N': N,
        'top_k': top_k,
        'naive_ms': naive_ms,
        'baseline_ms': baseline_ms,
        'noreq_ms': noreq_ms,
        'fused_ms': fused_ms,
        'speedup': baseline_ms / fused_ms,
        'speedup_noreq': noreq_ms / fused_ms,
        'speedup_naive': naive_ms / fused_ms,
        'naive_tflops': flops / (naive_ms * 1e-3) / 1e12,
        'baseline_tflops': flops / (baseline_ms * 1e-3) / 1e12,
        'fused_tflops': flops / (fused_ms * 1e-3) / 1e12,
        'max_abs': max_abs,
        'rel': rel,
        'naive_max_abs': naive_max_abs,
    }
    del state, baseline_run, naive_run, fused_run, noreq_run, baseline_fn, naive_fn, fused_fn, noreq_fn
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return row


def fmt_tokens(t: int) -> str:
    return f'{t // 1024}k' if t >= 1024 and t % 1024 == 0 else str(t)


BASELINE_LABEL = {
    'bf16': 'torch grouped_mm (bf16)',
    'mxfp8': 'torch scaled_grouped_mm (mxfp8)',
    'mxfp8_noreq': 'torch scaled_grouped_mm (mxfp8, requant untimed)',
}
NAIVE_LABEL = {'bf16': 'torch naive expert loop (bf16)', 'mxfp8': 'naive expert loop (mxfp8)', 'mxfp8_noreq': ''}
FUSED_LABEL = {
    'bf16': 'fused MoE (bf16)',
    'mxfp8': 'fused MoE (mxfp8)',
    'mxfp8_noreq': 'fused MoE (mxfp8, requant on-chip)',
}


def _fmt_ms(v: float) -> str:
    return 'n/a' if v != v else f'{v:.3f} ms'


def _fmt_x(v: float) -> str:
    return 'n/a' if v != v else f'{v:.2f}x'


def print_table(dtype: str, rows: list[dict]) -> None:
    has_noreq = any(r['noreq_ms'] == r['noreq_ms'] for r in rows)
    head = ('tokens', NAIVE_LABEL[dtype], BASELINE_LABEL[dtype], FUSED_LABEL[dtype], 'vs grouped', 'vs naive')
    if has_noreq:
        head += ('grouped w/o requant', 'vs grouped w/o requant')
    body = []
    for r in rows:
        b = (
            fmt_tokens(r['tokens']),
            _fmt_ms(r['naive_ms']),
            _fmt_ms(r['baseline_ms']),
            _fmt_ms(r['fused_ms']),
            _fmt_x(r['speedup']),
            _fmt_x(r['speedup_naive']),
        )
        if has_noreq:
            b += (_fmt_ms(r['noreq_ms']), _fmt_x(r['speedup_noreq']))
        body.append(b)
    widths = [max(len(head[i]), max((len(b[i]) for b in body), default=0)) for i in range(len(head))]
    print(f'\n{dtype}')
    print('  '.join(h.ljust(w) for h, w in zip(head, widths)).rstrip())
    for b in body:
        print('  '.join(c.ljust(w) for c, w in zip(b, widths)).rstrip())
    print('\nmarkdown:')
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '|'.join('---' for _ in head) + '|')
    for b in body:
        print('| ' + ' | '.join(b) + ' |')


def plot_dtype(dtype: str, rows: list[dict], path: str, title: str | None = None) -> None:
    labels = [fmt_tokens(r['tokens']) for r in rows]
    xs = list(range(len(rows)))
    naive = [r['naive_ms'] for r in rows]
    base = [r['baseline_ms'] for r in rows]
    fused = [r['fused_ms'] for r in rows]
    speedup = [r['speedup'] for r in rows]
    speedup_naive = [r['speedup_naive'] for r in rows]
    have_naive = all(v == v for v in naive)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=SURFACE)
    fig.suptitle(title or f'Fused MoE vs torch baselines — {dtype}', x=0.012, y=0.97, ha='left', fontsize=15, color=INK, weight='bold')

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis='y', color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, length=0, labelsize=9.5)
        ax.set_xticks(xs, labels)
        ax.set_xlabel('tokens', fontsize=9.5, color=INK_SOFT)

    if have_naive:
        ax1.plot(xs, naive, color=C_NAIVE, linewidth=2, marker='o', markersize=6, label=NAIVE_LABEL[dtype], zorder=3)
    ax1.plot(xs, base, color=C_TORCH, linewidth=2, marker='o', markersize=6, label=BASELINE_LABEL[dtype], zorder=3)
    ax1.plot(xs, fused, color=C_FUSED, linewidth=2, marker='o', markersize=6, label=FUSED_LABEL[dtype], zorder=3)
    ax1.set_yscale('log')
    ax1.minorticks_off()
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax1.set_xlim(-0.35, len(xs) - 1 + 0.75)
    ax1.set_title('median latency (ms, log scale)', fontsize=10.5, color=INK, loc='left', pad=10)
    # selective direct labels: the value at the last point, identity comes from the legend;
    # the three series converge at 64k, so dodge the labels vertically to keep them apart
    if have_naive:
        ax1.annotate(
            f'{naive[-1]:.2f} ms', (xs[-1], naive[-1]), textcoords='offset points', xytext=(7, 14),
            ha='left', fontsize=9, color=INK,
        )
    ax1.annotate(
        f'{base[-1]:.2f} ms', (xs[-1], base[-1]), textcoords='offset points', xytext=(7, 2),
        ha='left', fontsize=9, color=INK,
    )
    ax1.annotate(
        f'{fused[-1]:.2f} ms', (xs[-1], fused[-1]), textcoords='offset points', xytext=(7, -12),
        ha='left', fontsize=9, color=INK,
    )
    leg = ax1.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc='upper left')
    leg.set_zorder(4)

    # speedup bars are colored by the comparator so they match its latency line
    ax2.axhline(1.0, color=INK_SOFT, linewidth=1, linestyle=(0, (4, 4)), zorder=2)
    if have_naive:
        ax2.bar([x - 0.19 for x in xs], speedup, width=0.36, color=C_TORCH, label='vs grouped GEMM', zorder=3)
        ax2.bar([x + 0.19 for x in xs], speedup_naive, width=0.36, color=C_NAIVE, label='vs naive loop', zorder=3)
        tops = speedup + speedup_naive
        for xi, s in zip(xs, speedup):
            ax2.annotate(f'{s:.2f}x', (xi - 0.19, s), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8, color=INK)
        for xi, s in zip(xs, speedup_naive):
            ax2.annotate(f'{s:.2f}x', (xi + 0.19, s), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8, color=INK)
        leg2 = ax2.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc='upper left')
        leg2.set_zorder(4)
    else:
        ax2.bar(xs, speedup, width=0.62, color=C_TORCH, zorder=3)
        tops = speedup
        for xi, s in zip(xs, speedup):
            ax2.annotate(f'{s:.2f}x', (xi, s), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=9, color=INK)
    ax2.set_ylim(0, max(tops) * 1.22)
    ax2.set_title('fused speedup over each baseline', fontsize=10.5, color=INK, loc='left', pad=10)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def write_csv(rows: list[dict], path: str) -> None:
    keys = list(rows[0].keys())
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(str(r[k]) for k in keys) + '\n')


# ---------------------------------------------------------------- main


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tokens', type=int, nargs='+', default=[4096, 8192, 16384, 32768, 65536, 2*65536])
    p.add_argument('--dtypes', nargs='+', choices=['bf16', 'mxfp8'], default=['bf16', 'mxfp8'])
    p.add_argument('--experts', type=int, default=32)
    p.add_argument('--top-k', type=int, default=4)
    p.add_argument('--hidden', type=int, default=2048, help='K, the model dim')
    p.add_argument('--inter', type=int, default=1024, help='per expert intermediate dim H; w1 has N = 2H rows')
    p.add_argument('--balancedness', type=float, default=1.0, help='1.0 = uniform routing, 0.0 = everything to expert 0')
    p.add_argument('--block-m', type=int, default=128)
    p.add_argument('--bn', type=int, default=64)
    p.add_argument('--wn', type=int, default=4)
    p.add_argument('--stages', type=int, default=None, help='pipeline ring depth; default: 2 for bf16, 4 for mxfp8 with --split')
    p.add_argument('--bpc', type=int, default=1)
    p.add_argument('--cpc', type=int, default=1, help='bf16 only: intermediate 128-col groups per CTA, cuts the down-projection split-K factor by this much')
    p.add_argument(
        '--split', action=argparse.BooleanOptionalAction, default=True,
        help='--split (default) runs the split pipeline: the activation is materialized once and the '
             'down-projection split-K disappears. --no-split runs the single fused kernel, which keeps '
             'the activation on chip and pays a split-K reduction that grows with token count.',
    )
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--check', action='store_true', help='compare the fused output against the torch baseline')
    p.add_argument('--cuda-graphs', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--skip-fused', action='store_true', help='time the torch baselines only')
    p.add_argument('--skip-naive', action='store_true', help='skip the naive per-expert loop baseline')
    p.add_argument('--outdir', type=str, default='.')
    args = p.parse_args()

    if not HAVE_KERNEL and not args.skip_fused:
        print(f'prime_moe could not be imported ({KERNEL_IMPORT_ERROR}); rerun with --skip-fused for the torch side only')
        sys.exit(1)

    torch.cuda.set_device('cuda:0')
    gpu = torch.cuda.get_device_name(0)
    K, N = args.hidden, 2 * args.inter
    subtitle = (
        f'{gpu} · E={args.experts} top_k={args.top_k} K={K} N={N} (H={args.inter}) · '
        f'median of {args.iters} runs, {"CUDA graph replay" if args.cuda_graphs else "eager"}'
    )
    print(subtitle)
    print('baseline timing (grouped and naive) covers the GEMMs, SwiGLU and the routing scale only; gather/scatter are excluded')

    all_rows = []
    for dtype in args.dtypes:
        if dtype == 'mxfp8' and (K % 256 or N % 256):
            print(f'skipping mxfp8: K={K} and N={N} must both be multiples of 256')
            continue
        rows = []
        for t in args.tokens:
            print(f'bench {dtype} tokens={t}')
            try:
                row = run_one(dtype, t, args)
            except Exception as exc:
                print(f'  FAILED: {type(exc).__name__}: {exc}')
                torch.cuda.empty_cache()
                continue
            rows.append(row)
            msg = (
                f'  naive={row["naive_ms"]:.3f} ms  grouped={row["baseline_ms"]:.3f} ms  fused={row["fused_ms"]:.3f} ms'
                f'  vs_grouped={row["speedup"]:.2f}x  vs_naive={row["speedup_naive"]:.2f}x'
            )
            if row['noreq_ms'] == row['noreq_ms']:
                msg += f'  grouped_noreq={row["noreq_ms"]:.3f} ms  vs_grouped_noreq={row["speedup_noreq"]:.2f}x'
            if args.check:
                msg += f'  max_abs={row["max_abs"]:.4g}  rel={row["rel"]:.4g}  naive_vs_grouped={row["naive_max_abs"]:.4g}'
            print(msg)
        if not rows:
            continue
        all_rows += rows
        print_table(dtype, rows)
        if not args.skip_fused:
            png = f'{args.outdir.rstrip("/")}/moe_bench_{dtype}.png'
            plot_dtype(dtype, rows, png)
            print(f'wrote {png}')
            if any(r['noreq_ms'] == r['noreq_ms'] for r in rows):
                # third plot: same fused numbers against the baseline with the intermediate
                # requant untimed — unfair to the kernel (it still requantizes on chip), but
                # shows the raw GEMM-side speedup
                noreq_rows = [
                    {**r, 'baseline_ms': r['noreq_ms'], 'speedup': r['speedup_noreq'], 'naive_ms': float('nan')}
                    for r in rows
                ]
                png = f'{args.outdir.rstrip("/")}/moe_bench_{dtype}_noreq.png'
                plot_dtype(f'{dtype}_noreq', noreq_rows, png, title=f'Fused MoE vs torch baseline — {dtype}, requant untimed for torch')
                print(f'wrote {png}')

    if all_rows:
        csv = f'{args.outdir.rstrip("/")}/moe_bench.csv'
        write_csv(all_rows, csv)
        print(f'wrote {csv}')


if __name__ == '__main__':
    main()
