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
    bpc: int,
    cpc: int,
    split: bool,
) -> None:
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
    cpc: int = 1,
    split: bool = True,
) -> torch.Tensor:
    torch.ops.prime_moe.fused_moe_bf16(
        x, w, w2, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out,
        top_k, block_m, block_n, warp_n, stages, bpc, cpc, split
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
    split: bool,
) -> None:
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
    split: bool = True,
) -> torch.Tensor:
    torch.ops.prime_moe.fused_moe_mxfp8(
        x, x_scales, w, w_scales, w2, w2_scales, sorted_token_ids, expert_ids,
        num_tokens_post_padded, topk_weights, out,
        top_k, block_m, block_n, warp_n, stages, bpc, split
    )
    return out


BLOCK = 32  # MX scale block size


def pack_scales_blocked(scales: torch.Tensor) -> torch.Tensor:
    """Reorder row major weight scales into the blocked layout the kernel reads."""
    u = scales.view(torch.uint8)
    lead = u.shape[:-2]
    mn, kb = u.shape[-2:]
    assert (mn&127) == 0, f'MN ({mn}) must be a multiple of 128'
    assert (kb&3) == 0, f'K ({kb<<5}) must be a multiple of 128'
    t = u.reshape(-1, mn>>7, 4, 32, kb>>2, 4)
    t = t.permute(0, 1, 4, 3, 2, 5).contiguous()
    return t.reshape(*lead, -1).view(torch.float8_e8m0fnu)

__all__ = [
    'fused_moe_bf16',
    'fused_moe_mxfp8',
    'pack_scales_blocked',
    'BLOCK',
]
