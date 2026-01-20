import prime_moe
import torch
import triton.language as tl

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    moe_align_block_size,
    try_get_optimal_moe_config,
)
from sglang.srt.server_args import set_global_server_args_for_scheduler

class FakeServerArgs:
    enable_deterministic_inference=False

set_global_server_args_for_scheduler(FakeServerArgs())

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
    uniform = torch.ones(num_experts) / num_experts
    skewed = torch.zeros(num_experts)
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

configs = []
for block_m in range(8, 129, 8):
    for bn, wn in [(64, 4), (32, 8)]:
        for stages in range(1, 6):
            if stages == 5 and block_m > 100:
                continue
            configs.append((block_m, bn, wn, stages))

for E in e_vals:
        for N in n_vals:
            for K in k_vals:
                for num_tokens in batch_sizes:
                    for block_m, bn, wn, stages in configs:
                        x = torch.empty((num_tokens, K), dtype=torch.bfloat16).normal_(mean=0, std=0.05).to('cuda:0')
                        w1 = (torch.randn((E, N, K)) * 50).to('cuda:0')
                        w2 = (torch.randn((E, K, N // 2)) * 50).to('cuda:0')
                        topk_ids = generate_topk_ids(E-1, num_tokens, top_k-1).to('cuda:0')
                        topk_ids = torch.hstack((topk_ids, torch.ones(num_tokens).view(num_tokens,1).to(torch.int32)*(E-1))).to('cuda:0')
                        topk_weights = torch.nn.functional.softmax(torch.randn((num_tokens, top_k), dtype=torch.float32), dim=-1).to('cuda:0')
                        out = torch.zeros_like(x).to('cuda:0')
                        triton_config = try_get_optimal_moe_config(w1.shape, w2.shape, top_k, 'bfloat16', block_shape=block_shape, M=num_tokens)
                        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                            topk_ids, triton_config["BLOCK_SIZE_M"], E
                        )
                        sorted_token_ids = sorted_token_ids.to('cuda:0')
                        expert_ids = expert_ids.to('cuda:0')
                        num_tokens_post_padded = num_tokens_post_padded.to('cuda:0')
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
