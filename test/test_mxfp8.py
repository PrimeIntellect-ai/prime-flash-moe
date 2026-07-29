import prime_moe
import torch

from prime_moe.mxfp8 import dequantize_mx, pack_scales_blocked, quantize_mx


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
    w1 = torch.randn((E, N, K), dtype=torch.bfloat16, device='cuda:0')
    w2 = torch.randn((E, K, N // 2), dtype=torch.bfloat16, device='cuda:0')

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
        1,
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
    print(f'T={num_tokens} K={K} E={E} N={N} bn={bn} wn={wn} stages={stages}')
    print('  finite fused/naive:', torch.isfinite(out.float()).all().item(), torch.isfinite(out_naive.float()).all().item())
    print('  max_abs', diff.max().item(), 'mean_abs', diff.mean().item())
    print('  rel_clamp_1', (diff / ref.clamp_min(1.0)).max().item())
    print('  rel_clamp_10', (diff / ref.clamp_min(10.0)).max().item())


if __name__ == '__main__':
    # mxfp8 path: N and K must be multiples of 256, block_m is 128, bpc is 1
    batch_sizes = [8, 32, 128, 256]
    n_vals = [256, 512]
    k_vals = [7168, 4096]
    e_vals = [257, 8, 37]
    top_k = 9
    block_m = 128
    configs = [(block_m, bn, wn, stages) for bn, wn in [(32, 8), (64, 4)] for stages in (1, 2)]
    for E in e_vals:
        for N in n_vals:
            for K in k_vals:
                for num_tokens in batch_sizes:
                    for block_m, bn, wn, stages in configs:
                        test_cfg(num_tokens, K, E, N, top_k, block_m, bn, wn, stages)
