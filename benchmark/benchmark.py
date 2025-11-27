import time
import torch
import fused_experts as fe
import matplotlib.pyplot as plt

S=3
N, D = 2048>>S, 2048>>S
H = 4096>>S
M = 3072>>S
E = 16>>S

DEVICE = 'cuda'
DTYPE = torch.bfloat16

WARMUP_ITERS = 10
NUM_ITERS = 10_000

counts = torch.tensor([N // E] * E, device=DEVICE, dtype=torch.int32)
offs = counts.cumsum(0).to(torch.int32)
starts = torch.empty_like(offs)
starts[0] = 0
if E > 1:
    starts[1:] = offs[:-1]

x = torch.rand((N, D), device=DEVICE, dtype=DTYPE)
wg = torch.rand((E, H, D), device=DEVICE, dtype=DTYPE)
wu = torch.rand((E, H, D), device=DEVICE, dtype=DTYPE)
wd = torch.rand((E, M, H), device=DEVICE, dtype=DTYPE)

wgT = wg.transpose(-2, -1).contiguous()  # w Gate
wuT = wu.transpose(-2, -1).contiguous()  # w Up
wdT = wd.transpose(-2, -1).contiguous()  # w Down

def moe_naive(x, wgT, wuT, wdT, offs) -> torch.Tensor:
    gate_proj = torch._grouped_mm(
        x,
        wgT,
        offs,
        out_dtype=torch.bfloat16,
    )
    up_proj = torch._grouped_mm(
        x,
        wuT,
        offs,
        out_dtype=torch.bfloat16,
    )
    activated = torch.nn.functional.silu(gate_proj) * up_proj
    return torch._grouped_mm(
        activated,
        wdT,
        offs,
        out_dtype=torch.bfloat16,
    )


def moe_fused(x, wgT, wuT, wdT, offs) -> torch.Tensor:
    return fe.fused_experts_kernel(x, wgT, wuT, wdT, offs)


def check_correctness():
    with torch.inference_mode():
        y_naive = moe_naive(x, wgT, wuT, wdT, offs)
        y_fused = moe_fused(x, wgT, wuT, wdT, offs)

    torch.testing.assert_close(y_fused, y_naive, atol=1e-1, rtol=1e-1)
    print('Correctness check passed!')


def benchmark_kernel(fn, label, num_iters=NUM_ITERS):
    times = []
    for _ in range(WARMUP_ITERS):
        _ = fn()

    if DEVICE == 'cuda':
        torch.cuda.synchronize()

    for i in range(num_iters):
        start = time.perf_counter()
        _ = fn()
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

        if (i + 1) % (num_iters // 10 or 1) == 0:
            print(f'{label}: {i + 1}/{num_iters} iterations done')

    return times


def run_benchmarks():
    print('Running benchmarks...')

    with torch.inference_mode():
        naive_times = benchmark_kernel(
            lambda: moe_naive(x, wgT, wuT, wdT, offs), 'naive'
        )
        fused_times = benchmark_kernel(
            lambda: moe_fused(x, wgT, wuT, wdT, offs), 'fused'
        )

    naive_total = sum(naive_times)
    fused_total = sum(fused_times)

    print(f'\nNaive total time: {naive_total:.3f} s '
          f'(avg {naive_total / len(naive_times) * 1e3:.3f} ms/iter)')
    print(f'Fused total time: {fused_total:.3f} s '
          f'(avg {fused_total / len(fused_times) * 1e3:.3f} ms/iter)')

    return naive_times, fused_times


def plot_results(naive_times, fused_times):
    import numpy as np

    naive_times_ms = np.array(naive_times) * 1e3
    fused_times_ms = np.array(fused_times) * 1e3
    iters = np.arange(1, len(naive_times_ms) + 1)

    plt.figure(dpi=300)
    plt.plot(iters, naive_times_ms, label='Naive (grouped_mm 3x)')
    plt.plot(iters, fused_times_ms, label='Fused kernel')
    plt.xlabel('Iteration')
    plt.ylabel('Time per iteration [ms]')
    plt.title(f'MoE Forward Runtime per Iteration (N={N}, D={D}, H={H}, M={M}, E={E})')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(6, 4))
    totals = [naive_times_ms.sum(), fused_times_ms.sum()]
    labels = ['Naive', 'Fused']
    plt.bar(labels, totals)
    plt.ylabel('Total time [ms]')
    plt.title(f'Total Runtime over {len(naive_times)} iterations')

    plt.tight_layout()
    plt.savefig('benchmark.png')
    plt.show()


if __name__ == '__main__':
    naive_times, fused_times = run_benchmarks()
    plot_results(naive_times, fused_times)