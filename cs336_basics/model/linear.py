import torch
from torch import nn
from einops import rearrange, einsum

class Linear(nn.Module):
    
    def __init__(self, in_features, out_features, device=None, dtype=None):
        
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
