from torch import nn
import torch
from .linear import Linear
from .rope import RoPE

class MHA(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.q_proj_weight = Linear()
        self.k_proj_weight = Linear()
    def forward():
        pass
