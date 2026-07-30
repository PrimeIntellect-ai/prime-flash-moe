import torch
from . import _C
from .mxfp8 import pack_scales_blocked


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
    bpc: int,
) -> None:
    # the op schema is `-> ()` with `Tensor(a!) out`; a fake impl must return None to match it
    return None


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
    bpc: int = 1,
) -> torch.Tensor:
    """Run the fused bf16 MoE. Writes into ``out`` in place and returns it for convenience."""
    torch.ops.prime_moe.fused_moe_bf16(
        x, w, w2, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out, top_k, block_m, block_n, warp_n, stages, bpc
    )
    return out

@torch.library.register_fake('prime_moe::fused_moe_mxfp8')
def fused_moe_mxfp8_abstract(
    x: torch.Tensor,
    x_scales: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
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
    bpc: int,
) -> None:
    # the op schema is `-> ()` with `Tensor(a!) out`; a fake impl must return None to match it
    return None


def fused_moe_mxfp8(
    x: torch.Tensor,
    x_scales: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
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
    bpc: int = 1,
) -> torch.Tensor:
    """Run the fused mxfp8 MoE. Writes into ``out`` in place and returns it for convenience.

    ``w_scales`` and ``w2_scales`` must already be in the blocked layout produced by
    :func:`prime_moe.pack_scales_blocked`; ``x_scales`` stays row major ``(M, K // 32)``.
    """
    torch.ops.prime_moe.fused_moe_mxfp8(
        x, x_scales, w, w_scales, w2, w2_scales, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out, top_k, block_m, block_n, warp_n, stages, bpc
    )
    return out

__all__ = [
    'fused_moe_bf16',
    'fused_moe_mxfp8',
    'pack_scales_blocked',
]
