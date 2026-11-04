import torch
from torch import nn
from einops import einsum
import math
class Linear(nn.Module):
    
    def __init__(self, in_features, out_features, device=None, dtype=None):
        
        super().__init__()

        # 创建可以训练的参数
        # y = xW^T
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3*sigma, b=3*sigma)




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight,
                      "... in_features, out_features in_features -> ... out_features"
                      )
