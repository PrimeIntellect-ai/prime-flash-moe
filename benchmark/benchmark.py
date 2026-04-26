import argparse
import itertools
import math
import statistics
import time

import prime_moe
import torch
import matplotlib.pyplot as plt


def generate_topk_ids(num_experts, num_tokens, top_k, balancedness=1.0):
    uniform = torch.ones(num_experts, device="cuda") / num_experts
    skewed = torch.zeros(num_experts, device="cuda")
    skewed[0] = 1.0
    probs = balancedness * uniform + (1.0 - balancedness) * skewed
    return torch.multinomial(probs, num_tokens * top_k, replacement=True).view(num_tokens, top_k)


def make_routing(topk_ids, block_m, num_experts):
    T, top_k = topk_ids.shape
    pair_ids = torch.arange(T * top_k, device="cuda", dtype=torch.int32)
    expert_flat = topk_ids.reshape(-1).to(torch.int32)

    sort_idx = torch.argsort(expert_flat)
    expert_sorted = expert_flat[sort_idx]
    pair_sorted = pair_ids[sort_idx]

    counts = torch.bincount(expert_sorted, minlength=num_experts).to(torch.int32)
    offs = torch.cumsum(counts, dim=0).to(torch.int32)

    padded_counts = ((counts + block_m - 1) // block_m) * block_m
    total_padded = int(padded_counts.sum().item())
    num_blocks = int((padded_counts // block_m).sum().item())

    sorted_token_ids = torch.empty((total_padded,), device="cuda", dtype=torch.int32)
    expert_ids = torch.empty((num_blocks,), device="cuda", dtype=torch.int32)

    read_ptr = 0
    write_ptr = 0
    block_ptr = 0

    for e in range(num_experts):
        c = int(counts[e].item())
        pc = int(padded_counts[e].item())

        if c:
            sorted_token_ids[write_ptr:write_ptr + c] = pair_sorted[read_ptr:read_ptr + c]
        if pc > c:
            sorted_token_ids[write_ptr + c:write_ptr + pc] = -1

        nb = pc // block_m
        if nb:
            expert_ids[block_ptr:block_ptr + nb] = e
            block_ptr += nb

        read_ptr += c
        write_ptr += pc

    num_tokens_post_padded = torch.tensor([total_padded], device="cuda", dtype=torch.int32)

    return {
        "pair_sorted": pair_sorted,
        "counts": counts,
        "offs": offs,
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "num_tokens_post_padded": num_tokens_post_padded,
    }


def moe_native_grouped(x, w1, w2, topk_weights, top_k, routing, weighted=False):
    pair_sorted = routing["pair_sorted"]
    offs = routing["offs"]

    tok = pair_sorted // top_k
    route = pair_sorted % top_k

    xg = x[tok]

    # Kernel uses w1 as [E, N, K], equivalent grouped_mm weight [E, K, N].
    h = torch._grouped_mm(xg, w1.transpose(1, 2), offs, out_dtype=torch.bfloat16)

    gate, up = h.chunk(2, dim=1)
    act = torch.nn.functional.silu(gate.float()).to(torch.bfloat16) * up

    # w2 is [E, K, N/2], grouped_mm wants [E, N/2, K].
    yg = torch._grouped_mm(act, w2.transpose(1, 2), offs, out_dtype=torch.bfloat16)

    if weighted:
        yg = yg * topk_weights[tok, route].unsqueeze(1).to(torch.bfloat16)

    out = torch.zeros_like(x)
    out.index_add_(0, tok.long(), yg)
    return out


def moe_kernel(x, w1, w2, topk_weights, out, top_k, block_m, bn, wn, stages, routing):
    out.zero_()
    torch.ops.prime_moe.fused_moe_bf16(
        x,
        w1,
        w2,
        routing["sorted_token_ids"],
        routing["expert_ids"],
        routing["num_tokens_post_padded"],
        topk_weights,
        out,
        top_k,
        block_m,
        bn,
        wn,
        stages,
    )
    return out


def bench_cuda(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def approx_flops(num_tokens, top_k, K, N):
    # w1: [K x N], where N contains gate+up.
    # w2: [N/2 x K].
    pairs = num_tokens * top_k
    return 2.0 * pairs * K * N + 2.0 * pairs * (N // 2) * K


def run_one(num_tokens, K, E, N, top_k, block_m, bn, wn, stages, args):
    torch.manual_seed(args.seed)

    x = torch.empty((num_tokens, K), dtype=torch.bfloat16, device="cuda").normal_(0, 0.05)
    w1 = torch.randn((E, N, K), dtype=torch.bfloat16, device="cuda") * 0.05
    w2 = torch.randn((E, K, N // 2), dtype=torch.bfloat16, device="cuda") * 0.05

    # Same pattern as your test: top_k - 1 random experts plus one forced expert.
    topk_ids = generate_topk_ids(E - 1, num_tokens, top_k - 1)
    forced = torch.full((num_tokens, 1), E - 1, device="cuda", dtype=torch.int32)
    topk_ids = torch.hstack((topk_ids, forced))

    if args.weighted:
        topk_weights = torch.nn.functional.softmax(
            torch.randn((num_tokens, top_k), dtype=torch.float32, device="cuda"),
            dim=-1,
        )
    else:
        # Your current kernel ignores topk_weights, so unweighted is the valid comparison.
        topk_weights = torch.ones((num_tokens, top_k), dtype=torch.float32, device="cuda")

    routing = make_routing(topk_ids, block_m, E)
    out_kernel = torch.empty_like(x)

    native_fn = lambda: moe_native_grouped(
        x, w1, w2, topk_weights, top_k, routing, weighted=args.weighted
    )

    kernel_fn = lambda: moe_kernel(
        x, w1, w2, topk_weights, out_kernel, top_k, block_m, bn, wn, stages, routing
    )

    if args.check:
        ref = native_fn()
        got = kernel_fn()
        torch.cuda.synchronize()

        diff = (ref.float() - got.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
    else:
        max_abs = float("nan")
        mean_abs = float("nan")

    native = bench_cuda(native_fn, args.warmup, args.iters)
    kernel = bench_cuda(kernel_fn, args.warmup, args.iters)

    flops = approx_flops(num_tokens, top_k, K, N)
    native_tflops = flops / (native["median_ms"] * 1e-3) / 1e12
    kernel_tflops = flops / (kernel["median_ms"] * 1e-3) / 1e12

    return {
        "tokens": num_tokens,
        "K": K,
        "E": E,
        "N": N,
        "top_k": top_k,
        "block_m": block_m,
        "bn": bn,
        "wn": wn,
        "stages": stages,
        "native_ms": native["median_ms"],
        "kernel_ms": kernel["median_ms"],
        "speedup": native["median_ms"] / kernel["median_ms"],
        "native_tflops": native_tflops,
        "kernel_tflops": kernel_tflops,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
    }


def plot_results(rows, path):
    labels = [
        f"T={r['tokens']} E={r['E']} N={r['N']} K={r['K']}"
        for r in rows
    ]

    native = [r["native_ms"] for r in rows]
    kernel = [r["kernel_ms"] for r in rows]
    speedup = [r["speedup"] for r in rows]

    x = list(range(len(rows)))
    width = 0.38

    plt.figure(figsize=(max(12, len(rows) * 0.7), 6))
    plt.bar([i - width / 2 for i in x], native, width, label="torch grouped_mm native")
    plt.bar([i + width / 2 for i in x], kernel, width, label="prime_moe fused kernel")
    plt.ylabel("median latency [ms]")
    plt.xticks(x, labels, rotation=75, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path.replace(".png", "_latency.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(max(12, len(rows) * 0.7), 6))
    plt.bar(x, speedup)
    plt.ylabel("speedup native / kernel")
    plt.xticks(x, labels, rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(path.replace(".png", "_speedup.png"), dpi=160)
    plt.close()

def summarize(rows):
    speedups = [r["speedup"] for r in rows]
    kernel_wins = [s for s in speedups if s > 1.0]
    kernel_losses = [s for s in speedups if s <= 1.0]

    native_total = sum(r["native_ms"] for r in rows)
    kernel_total = sum(r["kernel_ms"] for r in rows)

    print("\nSUMMARY")
    print(f"configs: {len(rows)}")
    print(f"wins: {len(kernel_wins)} / {len(rows)}")
    print(f"losses: {len(kernel_losses)} / {len(rows)}")
    print(f"arithmetic mean speedup: {sum(speedups) / len(speedups):.3f}x")
    print(f"end-to-end total speedup: {native_total / kernel_total:.3f}x")
    print(f"best speedup: {max(speedups):.3f}x")
    print(f"worst speedup: {min(speedups):.3f}x")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--plot", type=str, default="moe_bench.png")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)

    batch_sizes = [8, 32, 128, 256]
    n_vals = [256, 512]
    k_vals = [7168, 4096]
    e_vals = [257, 8, 37]

    top_k = 9
    block_m = 128
    configs = [(32, 8, 1)]

    rows = []

    for E, N, K, T, (bn, wn, stages) in itertools.product(
        e_vals, n_vals, k_vals, batch_sizes, configs
    ):
        print(f"bench T={T} E={E} N={N} K={K} bn={bn} wn={wn} stages={stages}")

        try:
            row = run_one(T, K, E, N, top_k, block_m, bn, wn, stages, args)
            rows.append(row)

            print(
                f"  native={row['native_ms']:.3f} ms "
                f"kernel={row['kernel_ms']:.3f} ms "
                f"speedup={row['speedup']:.2f}x "
                f"kernel={row['kernel_tflops']:.1f} TFLOP/s "
                f"max_abs={row['max_abs']:.4g}"
            )
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\nCSV:")
    header = list(rows[0].keys()) if rows else []
    print(",".join(header))
    for r in rows:
        print(",".join(str(r[k]) for k in header))

    if rows:
        plot_results(rows, args.plot)
        print(f"\nwrote {args.plot.replace('.png', '_latency.png')}")
        print(f"wrote {args.plot.replace('.png', '_speedup.png')}")
        summarize(rows)


if __name__ == "__main__":
    main()
    