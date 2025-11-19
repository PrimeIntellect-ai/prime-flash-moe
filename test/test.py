import fused_experts as fe
import torch

S=4
N, D = 2048>>S, 2048>>S
H = 4096>>S
M = 3072>>S
E = 16>>S

device = 'cuda'

counts = torch.tensor([N // E] * E, device=device, dtype=torch.int32)
offs = counts.cumsum(0).to(torch.int32)
starts = torch.empty_like(offs)
starts[0] = 0
if E > 1:
    starts[1:] = offs[:-1]

x =  torch.rand((N, D), device=device, dtype=torch.bfloat16)
wg = torch.rand((E, H, D), device=device, dtype=torch.bfloat16)
wu = torch.rand((E, H, D), device=device, dtype=torch.bfloat16)
wd = torch.rand((E, M, H), device=device, dtype=torch.bfloat16)

wgT = wg.transpose(-2, -1).contiguous() # w Gate
wuT = wu.transpose(-2, -1).contiguous() # w Up
wdT = wd.transpose(-2, -1).contiguous() # w Down

def moe_naive() -> torch.Tensor:
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

y_naive = moe_naive()
y_fused = fe.fused_experts_kernel(x, wgT, wuT, wdT, offs)
torch.testing.assert_close(y_fused, y_naive, atol=1e-1, rtol=1e-1)

print('Test passed!')
