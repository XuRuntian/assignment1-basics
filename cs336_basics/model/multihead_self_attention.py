from torch import nn, Tensor
import torch
from .linear import Linear
from jaxtyping import Bool, Float
from einops import rearrange
import einops
from .scaled_dot_product_attention import ScaledDotProductAttention
class MHA(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_heads = num_heads
        d_k = d_model // num_heads 
        d_v = d_model // num_heads
        self.q_proj_weight = Linear(
            in_features=d_model , out_features=num_heads * d_k, device=self.device, dtype=self.dtype 
        )
        self.k_proj_weight = Linear(
            in_features=d_model , out_features=num_heads * d_k, device=self.device, dtype=self.dtype
        )
        self.v_proj_weight = Linear(
            in_features=d_model , out_features=num_heads * d_v, device=self.device, dtype=self.dtype
        )
        self.o_proj_weight = Linear(
            in_features=num_heads * d_v, out_features=d_model, device=self.device, dtype=self.dtype
        )
        self.scaled_dot_product_attention =  ScaledDotProductAttention()
    def forward(self, x: Float[Tensor, " ... sequence_length d_model"]):
        q = self.q_proj_weight(x)
        k = self.k_proj_weight(x)
        v = self.v_proj_weight(x)

        q_s = rearrange(
            q, "... s (h d) -> ... h s d", h = self.num_heads
        )
        
        k_s = rearrange(
            k, "... s (h d) -> ... h s d", h = self.num_heads

        )

        v_s = rearrange(
            v, "... s (h d) -> ... h s d", h = self.num_heads

        )
        # 缺少rope
        mask = torch.ones(q_s.shape[-2], k_s.shape[-2], device=q_s.device
                          ).tril().bool()
        attention_heads =self.scaled_dot_product_attention(q_s, k_s, v_s, mask)
        attention = rearrange(
            attention_heads, "... h s d -> ... s (h d)"
        )
        
        multi_head_attention = self.o_proj_weight(attention)
        return multi_head_attention
