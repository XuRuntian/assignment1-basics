import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pass