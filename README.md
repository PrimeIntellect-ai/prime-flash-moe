# prime-flash-moe

Fused Mixture-of-Experts forward kernels for NVIDIA Blackwell, written against `tcgen05`.
The gate/up projection, the SwiGLU activation, the down projection and the top-k weighted
reduction are fused, so the `2H`-wide gate/up tensor is never materialized at all — it is
consumed straight out of the accumulator.

There are two pipelines, chosen with `split`:

- `split=False` — one kernel launch does the whole FFN and the intermediate activation
  never leaves the SM. The down projection is a split-K GEMM, so `out` is a global
  accumulator and **the caller must zero it first**.
- `split=True` (default) — the up projection and SwiGLU run as before, then the activation
  is written to a workspace once and a second kernel does the down projection with no
  split-K at all, chained by programmatic dependent launch. Trades one HBM round trip of
  the `H`-wide activation for a reduction cost that otherwise grows with token count.
  Zeroes `out` itself, on a side stream.

Two data types are provided:

| Path | Op | Weights | Activations | Accumulate |
| --- | --- | --- | --- | --- |
| bf16 | `prime_moe::fused_moe_bf16` | bf16 | bf16 | fp32 |
| mxfp8 | `prime_moe::fused_moe_mxfp8` | e4m3 + e8m0 block scales | e4m3 + e8m0 block scales | fp32 |

On the mxfp8 path the intermediate activation is requantized to mxfp8 on chip before the
down projection, so the down GEMM also runs at fp8 rates.

## Requirements

- Blackwell, compute capability `10.0a` (`sm_100a`). The `tcgen05` path has no fallback.
- CUDA Toolkit 12.8 or newer.
- PyTorch with matching CUDA, Python >= 3.12.

## Install

```sh
pip install -e .
```

`setup.py` defaults `TORCH_CUDA_ARCH_LIST` to `10.0a`; override it in the environment if you
need a different target.

## API

Three ops are registered in the `prime_moe` torch library. `fused_moe_bf16` and
`fused_moe_mxfp8` have schema `-> ()` and write into `out` in place (`Tensor(a!) out`); the
Python wrappers in `prime_moe/__init__.py` return `out` for convenience.

```python
prime_moe.fused_moe_bf16(x, w1, w2, sorted_token_ids, expert_ids, num_tokens_post_padded,
                         topk_weights, out, top_k, block_m, block_n, warp_n, stages,
                         bpc=1, cpc=1, split=True)

prime_moe.fused_moe_mxfp8(x, x_scales, w1, w1_scales, w2, w2_scales,
                          sorted_token_ids, expert_ids, num_tokens_post_padded,
                          topk_weights, out, top_k, block_m, block_n, warp_n, stages,
                          bpc=1, split=True)

sorted_token_ids, expert_ids, num_tokens_post_padded = torch.ops.prime_moe.moe_align(
    topk_ids, num_experts, block_m, bpc)
```

### Shapes

With `T` tokens, `K` model dim, `E` experts, and `N` fused gate+up columns (intermediate
size is `N // 2`):

| Tensor | Shape | dtype |
| --- | --- | --- |
| `x` | `(T, K)` | bf16 / e4m3 |
| `w1` | `(E, N, K)` | bf16 / e4m3 |
| `w2` | `(E, K, N // 2)` | bf16 / e4m3 |
| `out` | `(T, K)` | bf16 |
| `topk_weights` | `(T, top_k)` | fp32 |
| `sorted_token_ids`, `expert_ids`, `num_tokens_post_padded` | from `moe_align` | int32 |

`sorted_token_ids` holds flattened `token * top_k + route` pair ids grouped by expert and
padded to `block_m` with `-1`. Every tensor must be contiguous.

### Scale layout (mxfp8)

This is the one part a caller has to get right. Block size is 32 along `K`.

- **Activation scales** stay row major: `x_scales` is `(T, K // 32)` e8m0.
- **Weight scales** must be reordered into the blocked layout the kernel addresses as a
  sequence of 512-byte tiles, each covering a 128 MN x 128 K block. Use the helper:

```python
from prime_moe import pack_scales_blocked

w1_scales = pack_scales_blocked(w1_scales_row_major)   # (E, N, K // 32) -> blocked
w2_scales = pack_scales_blocked(w2_scales_row_major)   # (E, K, (N // 2) // 32) -> blocked
```

`pack_scales_blocked` is the only piece of quantization that lives in this package.
Producing the e4m3 values and e8m0 exponents themselves is the caller's job — that
normally happens upstream in the training framework. A reference implementation of the
OCP MX quantizer is in `test/test_mxfp8.py` if you need one to compare against.

### Constraints

Violations raise from `TORCH_CHECK` in `prime_moe/csrc/torch_interface.cpp`.

| | bf16 | mxfp8 |
| --- | --- | --- |
| `block_m` | 128 | 128 |
| `(block_n, warp_n)` | `(32, 8)` or `(64, 4)` | `(32, 8)` or `(64, 4)` |
| `stages` | 1..5 | 1..4 |
| `bpc` | 1 or 2 | 1 only |
| `K` | multiple of 128 | multiple of 256 |
| `N` | — | multiple of 256 |

`block_m` is fixed at 128 because the `tcgen05` MMA path is Layout-D only. `bpc=2` is
unavailable on the mxfp8 path because a 256-column accumulator plus the scale-factor
columns already fill the 512-column tensor memory.

### Tuning

`block_n`/`warp_n` and `stages` are exposed rather than auto-tuned. `(32, 8)` tends to win
for narrow `N`, `(64, 4)` for wider. More stages buys more prefetch overlap at the cost of
shared memory. Sweep them for your shapes — `benchmark/benchmark.py` does this.

## Usage

```python
import torch
import prime_moe
from prime_moe import pack_scales_blocked

T, K, E, N, top_k, block_m = 256, 4096, 8, 512, 4, 128

# routing comes from your gate; topk_ids is (T, top_k) int32
sorted_token_ids, expert_ids, num_tokens_post_padded = torch.ops.prime_moe.moe_align(
    topk_ids, E, block_m, 1)

out = torch.empty((T, K), dtype=torch.bfloat16, device='cuda')
prime_moe.fused_moe_mxfp8(
    x_q, x_scales,                                  # (T, K) e4m3, (T, K//32) e8m0 row major
    w1_q, pack_scales_blocked(w1_scales),           # (E, N, K) e4m3, blocked scales
    w2_q, pack_scales_blocked(w2_scales),           # (E, K, N//2) e4m3, blocked scales
    sorted_token_ids, expert_ids, num_tokens_post_padded,
    topk_weights, out,
    top_k, block_m, 32, 8, 1,                       # block_n=32, warp_n=8, stages=1
)
# out is written in place and also returned
```

Pack the weight scales once at load time, not per step.

## Tests and benchmarks

```sh
python test/test.py           # bf16 path vs a torch._grouped_mm reference
python test/test_mxfp8.py     # mxfp8 path vs the same reference on dequantized values
python benchmark/benchmark.py --check --iters 1000
```

Both test scripts sweep shapes and tuning configs, print per-config error statistics, and
exit non-zero if any config exceeds its accuracy gate. The gates (`REL_CLAMP_10_MAX`,
`REL_CLAMP_1_MAX` at the top of each file) are starting estimates and can be overridden
with `PRIME_MOE_REL_CLAMP_10` / `PRIME_MOE_REL_CLAMP_1`; calibrate them against a known
good run and tighten.

The mxfp8 reference dequantizes both sides back to bf16 and runs the naive path, so the
comparison isolates the kernel rather than the quantizer.

## Status

Pre-1.0. The bf16 path is the more exercised of the two; the mxfp8 path is newer. There is
no CI — the test scripts are the gate, and they need to run on a Blackwell machine.
