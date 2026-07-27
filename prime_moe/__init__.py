import torch
from . import _C
from .mxfp8 import quantize_activation_mxfp8, quantize_weight_mxfp8, pack_scales_blocked


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
    bpc: int = 1,
) -> torch.Tensor:
    return torch.ops.prime_moe.fused_moe_bf16(
        x, w, w2, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out, top_k, block_m, block_n, warp_n, stages, bpc
    )

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
) -> torch.Tensor:
    return out


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
) -> torch.Tensor:ch.ops.prime_moe.fused_moe_mxfp8(
        x, x_scales, w, w_scales, w2, w2_scales, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out, top_k, block_m, block_n, warp_n, stages, bpc
    )

__all__ = [
    'fused_moe_bf16',
    'fused_moe_mxfp8',
    'quantize_activation_mxfp8',
    'quantize_weight_mxfp8',
    'pack_scales_blocked',
]
