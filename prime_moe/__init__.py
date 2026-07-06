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


def fused_rmsnorm_residual(
        branch: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float = 1e-6,
        *,
        vals: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        rrms: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    C = branch.shape[-1]
    N = branch.numel() // C

    if vals is None:
        vals = torch.empty_like(branch, dtype=torch.float8_e4m3fn)
    if scales is None:
        scales = branch.new_empty(
            (N // 128, C // 128, 32, 4, 4), dtype=torch.float8_e8m0fnu
        )
    if rrms is None:
        rrms = branch.new_empty((N,), dtype=torch.float32)

    torch.ops.your_ns.fused_rmsnorm_residual(
        branch, residual, weight, epsilon, vals, scales, rrms
    )
    return vals, scales, rrms


__all__ = ['fused_moe_bf16', 'fused_rmsnorm_residual']
