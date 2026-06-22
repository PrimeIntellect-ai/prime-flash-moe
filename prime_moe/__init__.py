import torch
from . import _C


@torch.library.register_fake('prime_moe::fused_moe_bf16')
def fused_moe_bf16_abstract(
    x: torch.Tensor,
    w: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int,
    block_n: int,
    warp_n: int,
    stages: int,
) -> torch.Tensor:
    return out


def fused_moe_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int,
    block_n: int,
    warp_n: int,
    stages: int,
) -> torch.Tensor:
    return torch.ops.prime_moe.fused_moe_bf16(
        x, w, w2, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out, top_k, block_m, block_n, warp_n, stages
    )


__all__ = ['fused_moe_bf16']
