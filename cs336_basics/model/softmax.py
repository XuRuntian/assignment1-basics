import torch
from torch import nn
class SoftMax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, x: torch.Tensor):
        max_value = torch.amax(x, dim=self.dim, keepdim=True)
        x = x-max_value

        exp_values = torch.exp(x)

        return exp_values / torch.sum(exp_values, dim=self.dim, keepdim=True)
        
