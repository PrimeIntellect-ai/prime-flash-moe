import prime_moe
import torch

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

N, D = 2048>>4, 2048>>4
H = 4096>>4
M = 3072>>4
E = 16>>4

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

    # Flatten into token/expert pairs
    token_ids = torch.arange(T, device=device, dtype=torch.int32)
    token_ids = token_ids[:, None].expand(T, top_k).reshape(-1)      # [T*top_k]
    expert_flat = topk_ids.reshape(-1).to(torch.int32)               # [T*top_k]

    # Sort by expert so tokens are grouped per expert
    sort_idx = torch.argsort(expert_flat)                            # [T*top_k]
    expert_sorted = expert_flat[sort_idx]
    token_sorted  = token_ids[sort_idx]

    # Count how many pairs per expert
    counts = torch.bincount(expert_sorted, minlength=num_experts)    # [E]

    # Pad each expert's segment up to multiple of block_size_m
    padded_counts = ((counts + block_size_m - 1) // block_size_m) * block_size_m
    total_padded = int(padded_counts.sum().item())

    # Output buffers
    sorted_token_ids = torch.empty((total_padded,), device=device, dtype=torch.int32)

    # expert_ids: one per block
    num_blocks = int((padded_counts // block_size_m).sum().item())
    expert_ids = torch.empty((num_blocks,), device=device, dtype=torch.int32)

    # Fill per-expert segments
    write_ptr = 0
    block_ptr = 0

    # We need start offsets in the sorted list for each expert
    # Compute segment boundaries via prefix sum of counts
    # (counts is per expert, but expert_sorted is grouped already)
    # We'll walk experts in order 0..E-1
    # and consume counts[e] elements from token_sorted sequentially.
    read_ptr = 0
    for e in range(num_experts):
        c = int(counts[e].item())
        pc = int(padded_counts[e].item())

        if c > 0:
            sorted_token_ids[write_ptr : write_ptr + c] = token_sorted[read_ptr : read_ptr + c]
        # pad with -1 sentinel (common choice) for empty slots
        if pc > c:
            sorted_token_ids[write_ptr + c : write_ptr + pc] = -1

        # write expert id per block
        nb = pc // block_size_m
        if nb > 0:
            expert_ids[block_ptr : block_ptr + nb] = e
            block_ptr += nb

        write_ptr += pc
        read_ptr += c

    num_tokens_post_padded = torch.tensor([total_padded], device=device, dtype=torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_padded

def interleave_tensor(tensor, rep=8):
    M, N, K = tensor.shape
    first_half = tensor[:, :(N//2), :]
    second_half = tensor[:, (N//2):, :]
    first_chunks = first_half.view(M, (N//(2*rep)), rep, K)
    second_chunks = second_half.view(M, (N//(2*rep)), rep, K)
    interleaved = torch.stack([first_chunks, second_chunks], dim=2)
    result = interleaved.view(M, N, K)
    return result.contiguous()


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
max_stages = 2

configs = []
for block_m in range(8, 129, 8):
    for bn, wn in [(64, 4), (32, 8)]:
        for stages in range(1, max_stages+1):
            if stages == 5 and block_m > 100:
                continue
            configs.append((block_m, bn, wn, stages))

def pick_block_size_m(num_tokens: int):
    if num_tokens <= 32: return 32
    if num_tokens <= 64: return 64
    return 128

def test_cfg(num_tokens, K, E, N, top_k, block_m, bn, wn, stages):
    x = torch.empty((num_tokens, K), dtype=torch.bfloat16, device='cuda:0').normal_(mean=0, std=0.05)
    w1 = torch.randn((E, N, K), dtype=torch.bfloat16, device='cuda:0') * 50
    w2 = torch.randn((E, K, N // 2), dtype=torch.bfloat16, device='cuda:0') * 50
    topk_ids = generate_topk_ids(E-1, num_tokens, top_k-1)
    topk_ids = torch.hstack((topk_ids, torch.ones(num_tokens, device='cuda:0').view(num_tokens,1).to(torch.int32)*(E-1)))
    topk_weights = torch.nn.functional.softmax(torch.randn((num_tokens, top_k), dtype=torch.float32, device='cuda:0'), dim=-1)
    out = torch.zeros_like(x, dtype=torch.bfloat16, device='cuda:0')
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_simple(
        # IMPORTANT: The CUDA kernel interprets `block_m` as the token-blocking
        # used to create `expert_ids` (one expert id per block). For correctness
        # we must align using the same `block_m` we pass into the kernel.
        topk_ids, block_m, E
    )
    torch.ops.prime_moe.fused_moe_bf16(
        x,
        w1,
        w2,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        top_k,
        block_m,
        bn,
        wn,
        stages
    )
    torch.cuda.synchronize()
    print(out)

#test_cfg(1024, 7168, 257, 256, 9, 128, 64, 4, 1)
#exit()

for E in e_vals:
        for N in n_vals:
            for K in k_vals:
                for num_tokens in batch_sizes:
                    for block_m, bn, wn, stages in configs:
                        test_cfg(num_tokens, K, E, N, top_k, block_m, bn, wn, stages)
                        
