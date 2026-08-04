"""Benchmark the fused MoE kernels against a naive torch expert loop.

The torch baseline is deliberately reduced to the arithmetic only:

    gate_up = x @ w1[expert].T
    gate, up = gate_up.chunk(2, dim=-1)
    act = F.silu(gate) * up
    expert_out = act @ w2[expert].T
    out = routing_weight * expert_out

Routing, sorting, the per expert gather and the final scatter all happen once during
setup and sit outside the timed region, so the baseline is not charged for bookkeeping
that the fused kernel folds into its prologue. Both sides are timed with CUDA graph
replays by default so the numbers are GPU time, not python dispatch time.

Two dtypes are reported. bf16 compares against ``x @ w.T``; mxfp8 compares against
``torch.nn.functional.scaled_mm`` with OCP MX (1x32 e8m0) block scales on both operands,
and the baseline requantizes the intermediate activation exactly like the kernel does.
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
    from prime_moe.mxfp8 import pack_scales_blocked

    HAVE_KERNEL = True
    KERNEL_IMPORT_ERROR = None
except Exception as exc:  # the extension may not be built for this box
    prime_moe = None
    HAVE_KERNEL = False
    KERNEL_IMPORT_ERROR = exc

    def pack_scales_blocked(scales: torch.Tensor) -> torch.Tensor:
        """Local copy of the blocked scale packer, so the torch side runs without the extension."""
        u = scales.view(torch.uint8)
        lead = u.shape[:-2]
        mn, kb = u.shape[-2:]
        assert mn % 128 == 0, f'MN ({mn}) must be a multiple of 128'
        assert kb % 4 == 0, f'K ({kb * 32}) must be a multiple of 128'
        t = u.reshape(-1, mn // 128, 4, 32, kb // 4, 4)
        t = t.permute(0, 1, 4, 3, 2, 5).contiguous()
        return t.reshape(*lead, -1).view(torch.float8_e8m0fnu)


BLOCK = 32
E4M3_MAX = 448.0
E4M3_EMAX = 8  # 448 == 1.75 * 2**8
MX = ScalingType.BlockWise1x32
SWZ = SwizzleType.SWIZZLE_32_4_4

# validated categorical slots 1/2/3 (see the dataviz palette), light surface
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SOFT = '#52514e'
GRID = '#e4e3de'
C_TORCH = '#2a78d6'
C_FUSED = '#eb6834'
C_SPEEDUP = '#1baf7a'


# ---------------------------------------------------------------- mxfp8 helpers


def _shared_exponent(amax: torch.Tensor) -> torch.Tensor:
    bits = amax.float().contiguous().view(torch.int32)
    return (((bits >> 23) & 0xFF) - E4M3_EMAX).clamp_(0, 254)


def quantize_mx(t: torch.Tensor, block: int = BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dim of ``t`` to MXFP8, returning ``(e4m3 values, e8m0 scales)``."""
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


def mx_mm(a: torch.Tensor, a_sf: torch.Tensor, bt: torch.Tensor, bt_sf: torch.Tensor) -> torch.Tensor:
    """``a @ bt.T`` in mxfp8. Both scale tensors are already in the swizzled blocked layout."""
    return scaled_mm(a, bt.t(), a_sf, MX, bt_sf, MX, swizzle_a=SWZ, swizzle_b=SWZ, output_dtype=torch.bfloat16)


# ---------------------------------------------------------------- routing


def generate_topk_ids(num_experts: int, num_tokens: int, top_k: int, balancedness: float = 1.0) -> torch.Tensor:
    uniform = torch.ones(num_experts, device='cuda') / num_experts
    skewed = torch.zeros(num_experts, device='cuda')
    skewed[0] = 1.0
    probs = balancedness * uniform + (1.0 - balancedness) * skewed
    return torch.multinomial(probs, num_tokens * top_k, replacement=True).view(num_tokens, top_k).to(torch.int32)


def make_routing(topk_ids: torch.Tensor, block_m: int, num_experts: int) -> dict:
    """Sort the (token, slot) pairs by expert and build the kernel's block schedule."""
    T, top_k = topk_ids.shape
    pair_ids = torch.arange(T * top_k, device='cuda', dtype=torch.int32)
    expert_flat = topk_ids.reshape(-1).to(torch.int32)
    sort_idx = torch.argsort(expert_flat)
    expert_sorted = expert_flat[sort_idx]
    pair_sorted = pair_ids[sort_idx]
    counts = torch.bincount(expert_sorted, minlength=num_experts).to(torch.int32)
    padded_counts = ((counts + block_m - 1) // block_m) * block_m
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


def gather_per_expert(x: torch.Tensor, topk_weights: torch.Tensor, routing: dict, top_k: int, pad_to: int) -> dict:
    """Split the sorted pairs into one dense tensor per expert, padded up to ``pad_to`` rows.

    This is the work the fused kernel does inline and the baseline is not charged for.
    Padding rows are zero, so they contribute nothing but keep every expert tile a legal
    shape for the mxfp8 scale layout (which needs a multiple of 128 rows).
    """
    pair_sorted = routing['pair_sorted']
    tok_all = (pair_sorted // top_k).long()
    route_all = (pair_sorted % top_k).long()
    xs, ws, toks, rows = [], [], [], []
    off = 0
    for c in routing['counts']:
        tok = tok_all[off : off + c]
        route = route_all[off : off + c]
        padded = (c + pad_to - 1) // pad_to * pad_to
        xe = x.new_zeros((padded, x.shape[1]))
        xe[:c] = x[tok]
        we = torch.zeros((padded, 1), device=x.device, dtype=torch.bfloat16)
        we[:c, 0] = topk_weights[tok, route].to(torch.bfloat16)
        xs.append(xe)
        ws.append(we)
        toks.append(tok)
        rows.append(c)
        off += c
    return {'x': xs, 'w': ws, 'tok': toks, 'rows': rows}


# ---------------------------------------------------------------- the two implementations


def torch_naive_bf16(parts: dict, w1: torch.Tensor, w2: torch.Tensor) -> list[torch.Tensor]:
    outs = []
    for e, x in enumerate(parts['x']):
        gate_up = x @ w1[e].T
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        expert_out = act @ w2[e].T
        outs.append(parts['w'][e] * expert_out)
    return outs


def torch_naive_mxfp8(parts: dict, w1q: torch.Tensor, w1sf: torch.Tensor, w2q: torch.Tensor, w2sf: torch.Tensor) -> list[torch.Tensor]:
    outs = []
    for e, x in enumerate(parts['x']):
        gate_up = mx_mm(x, parts['x_sf'][e], w1q[e], w1sf[e])
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        # the kernel requantizes the activation to mxfp8 before the down projection, so the
        # baseline has to pay for it too or the comparison is not like for like
        act_q, act_sf = quantize_mx(act)
        expert_out = mx_mm(act_q, pack_scales_blocked(act_sf).view(act.shape[0], -1), w2q[e], w2sf[e])
        outs.append(parts['w'][e] * expert_out)
    return outs


def scatter_parts(outs: list[torch.Tensor], parts: dict, out: torch.Tensor) -> torch.Tensor:
    """Fold the per expert results back into a (T, K) tensor. Setup/checking only, never timed."""
    out.zero_()
    for e, y in enumerate(outs):
        c = parts['rows'][e]
        if c:
            out.index_add_(0, parts['tok'][e], y[:c].to(out.dtype))
    return out


def fused_bf16(state: dict) -> torch.Tensor:
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
    )
    return state['out']


def fused_mxfp8(state: dict) -> torch.Tensor:
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
    )
    return state['out']


# ---------------------------------------------------------------- timing


def graphed(fn: Callable[[], object], warmup: int = 3) -> Callable[[], object] | None:
    """Capture ``fn`` into a CUDA graph and return a replay callable, or ``None`` if capture fails."""
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


# ---------------------------------------------------------------- one measurement


def run_one(dtype: str, num_tokens: int, args: argparse.Namespace) -> dict:
    E, K, N, top_k = args.experts, args.hidden, 2 * args.inter, args.top_k
    block_m = args.block_m
    torch.manual_seed(args.seed)

    x = torch.empty((num_tokens, K), dtype=torch.bfloat16, device='cuda').normal_(0, 0.05)
    w1 = (torch.randn((E, N, K), dtype=torch.bfloat16, device='cuda') * 0.05).contiguous()
    w2 = (torch.randn((E, K, N // 2), dtype=torch.bfloat16, device='cuda') * 0.05).contiguous()
    topk_ids = generate_topk_ids(E, num_tokens, top_k, args.balancedness)
    topk_weights = F.softmax(torch.randn((num_tokens, top_k), dtype=torch.float32, device='cuda'), dim=-1)

    routing = make_routing(topk_ids, block_m, E)
    parts = gather_per_expert(x, topk_weights, routing, top_k, pad_to=max(block_m, 128))
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
        'stages': args.stages,
        'bpc': args.bpc,
    }

    if dtype == 'bf16':
        state |= {'x': x, 'w1': w1, 'w2': w2}
        baseline_fn = lambda: torch_naive_bf16(parts, w1, w2)
        fused_fn = lambda: fused_bf16(state)
    else:
        x_q, x_sf = quantize_mx(x)
        w1_q, w1_sf = quantize_mx(w1)
        w2_q, w2_sf = quantize_mx(w2)
        # per expert activation tiles, quantized and swizzled once during setup, replacing the
        # bf16 tiles in place so the padded bf16 copies can be freed
        parts['x_sf'] = [None] * len(parts['x'])
        for i, xe in enumerate(parts['x']):
            q, sf = quantize_mx(xe)
            parts['x'][i] = q
            parts['x_sf'][i] = pack_scales_blocked(sf).view(q.shape[0], -1)
        del xe
        w1_sf_e = pack_scales_blocked(w1_sf).view(E, N, K // BLOCK)
        w2_sf_e = pack_scales_blocked(w2_sf).view(E, K, (N // 2) // BLOCK)
        state |= {'x': x_q, 'x_sf': x_sf, 'w1': w1_q, 'w1_sf': w1_sf_e, 'w2': w2_q, 'w2_sf': w2_sf_e}
        baseline_fn = lambda: torch_naive_mxfp8(parts, w1_q, w1_sf_e, w2_q, w2_sf_e)
        fused_fn = lambda: fused_mxfp8(state)

    max_abs = rel = float('nan')
    if args.check and not args.skip_fused:
        ref = scatter_parts(baseline_fn(), parts, torch.zeros_like(x))
        got = fused_fn().clone()
        torch.cuda.synchronize()
        diff = (ref.float() - got.float()).abs()
        max_abs = diff.max().item()
        rel = (diff / ref.float().abs().clamp_min(1.0)).max().item()

    baseline_run = baseline_fn
    fused_run = fused_fn
    if args.cuda_graphs:
        g = graphed(baseline_fn)
        baseline_run = g or baseline_fn
        if g is None:
            print('  note: CUDA graph capture failed for the torch baseline, timing it eagerly')
        if not args.skip_fused:
            g = graphed(fused_fn)
            fused_run = g or fused_fn
            if g is None:
                print('  note: CUDA graph capture failed for the fused kernel, timing it eagerly')

    baseline_ms = bench_cuda(baseline_run, args.warmup, args.iters)
    fused_ms = float('nan') if args.skip_fused else bench_cuda(fused_run, args.warmup, args.iters)

    flops = approx_flops(num_tokens, top_k, K, N)
    row = {
        'dtype': dtype,
        'tokens': num_tokens,
        'E': E,
        'K': K,
        'N': N,
        'top_k': top_k,
        'baseline_ms': baseline_ms,
        'fused_ms': fused_ms,
        'speedup': baseline_ms / fused_ms,
        'baseline_tflops': flops / (baseline_ms * 1e-3) / 1e12,
        'fused_tflops': flops / (fused_ms * 1e-3) / 1e12,
        'max_abs': max_abs,
        'rel': rel,
    }
    del state, parts, baseline_run, fused_run, baseline_fn, fused_fn
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return row


# ---------------------------------------------------------------- reporting


def fmt_tokens(t: int) -> str:
    return f'{t // 1024}k' if t >= 1024 and t % 1024 == 0 else str(t)


BASELINE_LABEL = {'bf16': 'torch naive (bf16)', 'mxfp8': 'torch naive (mxfp8 scaled_mm)'}
FUSED_LABEL = {'bf16': 'fused MoE (bf16)', 'mxfp8': 'fused MoE (mxfp8)'}


def print_table(dtype: str, rows: list[dict]) -> None:
    base, fused = BASELINE_LABEL[dtype], FUSED_LABEL[dtype]
    head = ('tokens', base, fused, 'speedup')
    body = [
        (fmt_tokens(r['tokens']), f'{r["baseline_ms"]:.3f} ms', f'{r["fused_ms"]:.3f} ms', f'{r["speedup"]:.2f}x')
        for r in rows
    ]
    widths = [max(len(head[i]), max((len(b[i]) for b in body), default=0)) for i in range(4)]
    print(f'\n{dtype}')
    print('  '.join(h.ljust(w) for h, w in zip(head, widths)).rstrip())
    for b in body:
        print('  '.join(c.ljust(w) for c, w in zip(b, widths)).rstrip())
    print('\nmarkdown:')
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '|'.join('---' for _ in head) + '|')
    for b in body:
        print('| ' + ' | '.join(b) + ' |')


def plot_dtype(dtype: str, rows: list[dict], path: str, subtitle: str) -> None:
    labels = [fmt_tokens(r['tokens']) for r in rows]
    xs = list(range(len(rows)))
    base = [r['baseline_ms'] for r in rows]
    fused = [r['fused_ms'] for r in rows]
    speedup = [r['speedup'] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=SURFACE)
    fig.suptitle(f'Fused MoE vs naive torch — {dtype}', x=0.012, y=0.965, ha='left', fontsize=15, color=INK, weight='bold')
    fig.text(0.012, 0.912, subtitle, ha='left', fontsize=9.5, color=INK_SOFT)

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

    ax1.plot(xs, base, color=C_TORCH, linewidth=2, marker='o', markersize=6, label=BASELINE_LABEL[dtype], zorder=3)
    ax1.plot(xs, fused, color=C_FUSED, linewidth=2, marker='o', markersize=6, label=FUSED_LABEL[dtype], zorder=3)
    ax1.set_yscale('log')
    ax1.minorticks_off()
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax1.set_xlim(-0.35, len(xs) - 1 + 0.75)
    ax1.set_title('median latency (ms, log scale)', fontsize=10.5, color=INK, loc='left', pad=10)
    # selective direct labels: the value at the last point, identity comes from the legend
    ax1.annotate(
        f'{base[-1]:.2f} ms', (xs[-1], base[-1]), textcoords='offset points', xytext=(7, 3),
        ha='left', fontsize=9, color=INK,
    )
    ax1.annotate(
        f'{fused[-1]:.2f} ms', (xs[-1], fused[-1]), textcoords='offset points', xytext=(7, -10),
        ha='left', fontsize=9, color=INK,
    )
    leg = ax1.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc='upper left')
    leg.set_zorder(4)

    ax2.axhline(1.0, color=INK_SOFT, linewidth=1, linestyle=(0, (4, 4)), zorder=2)
    ax2.bar(xs, speedup, width=0.62, color=C_SPEEDUP, zorder=3)
    ax2.set_ylim(0, max(speedup) * 1.22)
    ax2.set_title('speedup over the torch baseline', fontsize=10.5, color=INK, loc='left', pad=10)
    for xi, s in zip(xs, speedup):
        ax2.annotate(f'{s:.2f}x', (xi, s), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=9, color=INK)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
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
    p.add_argument('--tokens', type=int, nargs='+', default=[1024, 2048, 4096, 8192, 16384, 32768, 65536])
    p.add_argument('--dtypes', nargs='+', choices=['bf16', 'mxfp8'], default=['bf16', 'mxfp8'])
    p.add_argument('--experts', type=int, default=32)
    p.add_argument('--top-k', type=int, default=4)
    p.add_argument('--hidden', type=int, default=2048, help='K, the model dim')
    p.add_argument('--inter', type=int, default=1024, help='per expert intermediate dim H; w1 has N = 2H rows')
    p.add_argument('--balancedness', type=float, default=1.0, help='1.0 = uniform routing, 0.0 = everything to expert 0')
    p.add_argument('--block-m', type=int, default=128)
    p.add_argument('--bn', type=int, default=32)
    p.add_argument('--wn', type=int, default=8)
    p.add_argument('--stages', type=int, default=4)
    p.add_argument('--bpc', type=int, default=1)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--check', action='store_true', help='compare the fused output against the torch baseline')
    p.add_argument('--cuda-graphs', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--skip-fused', action='store_true', help='time the torch baseline only')
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
    print('baseline timing covers the expert matmuls, SwiGLU and the routing scale only; gather/scatter are excluded')

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
            msg = f'  baseline={row["baseline_ms"]:.3f} ms  fused={row["fused_ms"]:.3f} ms  speedup={row["speedup"]:.2f}x'
            if args.check:
                msg += f'  max_abs={row["max_abs"]:.4g}  rel={row["rel"]:.4g}'
            print(msg)
        if not rows:
            continue
        all_rows += rows
        print_table(dtype, rows)
        if not args.skip_fused:
            png = f'{args.outdir.rstrip("/")}/moe_bench_{dtype}.png'
            plot_dtype(dtype, rows, png, subtitle)
            print(f'wrote {png}')

    if all_rows:
        csv = f'{args.outdir.rstrip("/")}/moe_bench.csv'
        write_csv(all_rows, csv)
        print(f'wrote {csv}')


if __name__ == '__main__':
    main()
