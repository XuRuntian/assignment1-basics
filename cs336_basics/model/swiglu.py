import torch
from torch import nn
from .linear import Linear
class SwiGlu(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        
        super().__init__()

        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear1_output = self.w1(x)
        return self.w2(torch.sigmoid(linear1_output) * linear1_output * self.w3(x))
