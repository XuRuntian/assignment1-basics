import torch
from torch import Tensor
from torch import nn
from .softmax import SoftMax
from einops import einsum
import math
from jaxtyping import Bool, Float
class ScaledDotProductAttention(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.softmax = SoftMax(dim=-1)
    
    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... keys d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
        )-> Float[Tensor, " ... queries d_v"]:
        
        QKt = einsum(Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys"
            )
        d_k = Q.shape[-1]
        scores = QKt / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return self.softmax(scores) @ V
