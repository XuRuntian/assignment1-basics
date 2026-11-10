import torch
from torch import nn
from .softmax import SoftMax
class ScaledDotProductAttention(nn.Module):
    
    def __init__(self):
        super().__init__()

        
    
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
        return Q * K.T / torch.sqrt()