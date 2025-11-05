import fused_experts as fe
import torch

hidden_states = torch.randn(2, 4, 8, device='cuda')
w1 = torch.randn(8, 16, device='cuda')
w2 = torch.randn(16, 8, device='cuda')
output = fe.fused_experts(hidden_states, w1, w2)
print(output)