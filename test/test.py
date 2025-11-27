import fused_experts as fe
import torch


DEVICE = 'cuda'
ITERS: int = 5


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

for i in range(ITERS):
    print(f'Test #{i+1} N={N} D={D} H={H} M={M} E={E}')
    counts = torch.tensor([N // E] * E, device=DEVICE, dtype=torch.int32)
    offs = counts.cumsum(0).to(torch.int32)
    starts = torch.empty_like(offs)
    starts[0] = 0
    if E > 1:
        starts[1:] = offs[:-1]
    x = torch.rand((N, D), device=DEVICE, dtype=torch.bfloat16)
    wg = torch.rand((E, H, D), device=DEVICE, dtype=torch.bfloat16)
    wu = torch.rand((E, H, D), device=DEVICE, dtype=torch.bfloat16)
    wd = torch.rand((E, M, H), device=DEVICE, dtype=torch.bfloat16)
    wgT = wg.transpose(-2, -1).contiguous() # w Gate
    wuT = wu.transpose(-2, -1).contiguous() # w Up
    wdT = wd.transpose(-2, -1).contiguous() # w Down
    y_naive = moe_naive(x, wgT, wuT, wdT, offs)
    y_fused = fe.fused_experts_kernel(x, wgT, wuT, wdT, offs)
    torch.testing.assert_close(y_fused, y_naive, atol=1e-1, rtol=1e-1)
    N >>= 1
    D >>= 1
    M >>= 1
    E >>= 1

print('All tests passed!')
