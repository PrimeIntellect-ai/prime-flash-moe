import prime_moe
import torch


def moe_naive_from_topk(x, w1, w2, topk_ids, topk_weights):
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
    route = pair_sorted % top_k
    x_sorted = x[tok]
    scale_sorted = topk_weights.reshape(-1)[pair_sorted].float()
    h = torch._grouped_mm(
        x_sorted,
        w1.transpose(1, 2).contiguous(),
        offs,
        out_dtype=torch.bfloat16,
    )
    gate, up = h.chunk(2, dim=-1)
    act = torch.nn.functional.silu(gate.float()) * up.float()
    act = act.to(torch.bfloat16)
    y_sorted = torch._grouped_mm(
        act,
        w2.transpose(1, 2).contiguous(),
        offs,
        out_dtype=torch.bfloat16,
    )
    y_sorted = (y_sorted.float() * scale_sorted[:, None]).to(torch.bfloat16)
    out = torch.zeros((T, K), device=x.device, dtype=torch.bfloat16)
    out.index_add_(0, tok.long(), y_sorted)
    return out


def moe_naive(x: torch.Tensor, wgT: torch.Tensor, wuT: torch.Tensor, wdT: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
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


N, D = 2048, 2048
H = 4096
M = 3072
E = 16

import torch


def moe_align_block_size_simple(topk_ids: torch.Tensor, block_size_m: int, num_experts: int):
    """
    topk_ids: int tensor [T, top_k], each entry in [0, num_experts)
    Returns:
      sorted_token_ids: int32 [T_padded]
      expert_ids: int32 [num_blocks]
      num_tokens_post_padded: int32 scalar tensor
    """
    assert topk_ids.dim() == 2
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
    write_ptr = 0
    block_ptr = 0
    read_ptr = 0
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
    topk_ids = topk_ids.view(num_tokens, top_k)
    return topk_ids


block_shape = [128, 128]
batch_sizes = [8, 32, 128, 256]
n_vals = [256, 512]
k_vals = [7168, 4096]
e_vals = [257, 8, 37]
top_k = 9
max_stages = 1
block_m = 128  # must be 128 on Blackwell

configs = []
for bn, wn in [(32, 8)]:
    for stages in range(1, max_stages + 1):
        configs.append((block_m, bn, wn, stages))


def pack_w1_for_kernel(w1):
    E, N, K = w1.shape
    assert N % 256 == 0
    H = N // 2
    assert H % 128 == 0
    gate = w1[:, :H, :]
    up = w1[:, H:, :]
    chunks = []
    for i in range(0, H, 128):
        chunks.append(gate[:, i : i + 128, :])
        chunks.append(up[:, i : i + 128, :])
    return torch.cat(chunks, dim=1).contiguous()


def test_cfg(num_tokens, K, E, N, top_k, block_m, bn, wn, stages):
    x = torch.empty((num_tokens, K), dtype=torch.bfloat16, device='cuda:0').normal_(mean=0, std=0.05)
    w1 = torch.randn((E, N, K), dtype=torch.bfloat16, device='cuda:0')
    w2 = torch.randn((E, K, N // 2), dtype=torch.bfloat16, device='cuda:0')
    topk_ids = generate_topk_ids(E - 1, num_tokens, top_k - 1)
    topk_ids = torch.hstack((topk_ids, torch.ones(num_tokens, device='cuda:0').view(num_tokens, 1).to(torch.int32) * (E - 1)))
    topk_weights = torch.nn.functional.softmax(torch.randn((num_tokens, top_k), dtype=torch.float32, device='cuda:0'), dim=-1)
    out = torch.zeros_like(x, dtype=torch.bfloat16, device='cuda:0')
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_simple(topk_ids, block_m, E)
    torch.ops.prime_moe.fused_moe_bf16(
        x, pack_w1_for_kernel(w1), w2, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out, top_k, block_m, bn, wn, stages
    )
    torch.cuda.synchronize()
    out_naive = moe_naive_from_topk(x, w1, w2, topk_ids, topk_weights)
    torch.cuda.synchronize()
    print('finite fused:', torch.isfinite(out.float()).all().item())
    print('finite naive:', torch.isfinite(out_naive.float()).all().item())
    diff = (out.float() - out_naive.float()).abs()
    ref = out_naive.float().abs()
    print('max_abs', diff.max().item())
    print('mean_abs', diff.mean().item())
    print('p99_abs', torch.quantile(diff.flatten(), 0.99).item())
    print('rel_clamp_1', (diff / ref.clamp_min(1.0)).max().item())
    print('rel_clamp_10', (diff / ref.clamp_min(10.0)).max().item())


for E in e_vals:
    for N in n_vals:
        for K in k_vals:
            for num_tokens in batch_sizes:
                for block_m, bn, wn, stages in configs:
                    test_cfg(num_tokens, K, E, N, top_k, block_m, bn, wn, stages)
