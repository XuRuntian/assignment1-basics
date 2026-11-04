import torch
from torch import nn

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )

        nn.init.ones_(self.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, d_model)

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_norm = x / torch.sqrt(1 / self.d_model * torch.sum(torch.square(x), dim=-1, keepdim=True) + self.eps) * self.weight
        return rms_norm.to(in_dtype)