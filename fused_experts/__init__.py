import torch

from fused_experts.C.csrc.kernel import fused_experts_kernel

def fused_experts(
    x: torch.Tensor,        # hidden states             (N,D)
    wgt: torch.Tensor,      # weight gate transposed    (E,D,H)
    wut: torch.Tensor,      # weight up transposed      (E,D,H)
    wdt: torch.Tensor,      # weight down transposed    (E,H,M)
    offsets: torch.Tensor   # expert offsets            (E)
) -> torch.Tensor:
    assert x.is_cuda and wgt.is_cuda and wut.is_cuda and wdt.is_cuda and offsets.is_cuda, 'All inputs must be on CUDA device'
    assert x.is_contiguous(), 'Input x must be contiguous'
    assert wgt.is_contiguous(), 'Weight gate wgt must be contiguous'
    assert wut.is_contiguous(), 'Weight up wut must be contiguous'
    assert wdt.is_contiguous(), 'Weight down wdt must be contiguous'
    assert offsets.is_contiguous(), 'Offsets must be contiguous'
    assert x.dtype == torch.bfloat16, f'Input x must be of type bfloat16, but got {x.dtype}'
    assert wgt.dtype == torch.bfloat16, f'Weight gate wgt must be of type bfloat16, but got {wgt.dtype}'
    assert wut.dtype == torch.bfloat16, f'Weight up wut must be of type bfloat16, but got {wut.dtype}'
    assert wdt.dtype == torch.bfloat16, f'Weight down wdt must be of type bfloat16, but got {wdt.dtype}'
    assert offsets.dtype == torch.int32, f'Offsets must be of type int32, but got {offsets.dtype}'
    assert x.dim() == 2, f'Input x must be 2-dimensional, but got {x.dim()} dimensions'
    assert wgt.dim() == 3, f'Weight gate wgt must be 3-dimensional, but got {wgt.dim()} dimensions'
    assert wut.dim() == 3, f'Weight up wut must be 3-dimensional, but got {wut.dim()} dimensions'
    assert wdt.dim() == 3, f'Weight down wdt must be 3-dimensional, but got {wdt.dim()} dimensions'
    assert wgt.shape == wut.shape, f'Weight gate wgt and weight up wut must have the same shape, but got {wgt.shape} and {wut.shape}'
    assert wgt.shape[0] == wdt.shape[0], f'Number of experts in wgt and wdt must be the same, but got {wgt.shape[0]} and {wdt.shape[0]}'
    assert wgt.shape[2] == wdt.shape[1], f'Hidden size H in wgt and wdt must match, but got {wgt.shape[2]} and {wdt.shape[1]}'
    return fused_experts_kernel(
        x,
        wgt,
        wut,
        wdt,
        offsets
    )
