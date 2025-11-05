import torch

def fused_experts_kernel(
    x: torch.Tensor,
    wgt: torch.Tensor,
    wut: torch.Tensor,
    wdt: torch.Tensor,
    offsets: torch.Tensor
) -> torch.Tensor: ...
