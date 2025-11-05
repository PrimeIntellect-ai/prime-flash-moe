import torch

from fused_experts.C.csrc.kernel import fused_experts_kernel

def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    return fused_experts_kernel(
        hidden_states,
        w1,
        w2,
    )
