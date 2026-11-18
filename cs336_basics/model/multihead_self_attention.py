from torch import nn, Tensor
import torch
from .linear import Linear
from jaxtyping import Float, Int
from einops import rearrange
from .scaled_dot_product_attention import ScaledDotProductAttention
from .rope import RoPE
class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int=None, max_seq_len: int=None, theta: float=None,device=None, dtype=None):
        super().__init__()
        if max_seq_len and theta:
            self.use_rope = True
        else:
            self.use_rope = False
        if  d_model % num_heads == 0 :
            self.num_heads = num_heads
            d_k = d_model // num_heads 
            d_v = d_model // num_heads
        else:
            raise ValueError
        self.q_proj = Linear(
            in_features=d_model , out_features=num_heads * d_k, device=device, dtype=dtype 
        )
        self.k_proj = Linear(
            in_features=d_model , out_features=num_heads * d_k, device=device, dtype=dtype
        )
        self.v_proj = Linear(
            in_features=d_model , out_features=num_heads * d_v, device=device, dtype=dtype
        )
        self.output_proj = Linear(
            in_features=num_heads * d_v, out_features=d_model, device=device, dtype=dtype
        )
        self.scaled_dot_product_attention =  ScaledDotProductAttention()
        if self.use_rope:
            self.rope = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)
    def forward(self, x: Float[Tensor, " ... sequence_length d_model"], token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_s = rearrange(
            q, "... s (h d) -> ... h s d", h = self.num_heads
        )
        
        k_s = rearrange(
            k, "... s (h d) -> ... h s d", h = self.num_heads

        )

        v_s = rearrange(
            v, "... s (h d) -> ... h s d", h = self.num_heads

        )
        if token_positions is None:
            token_positions  = torch.arange(q.shape[-2], device=q.device)
        if self.use_rope and token_positions is not None:
            token_positions = token_positions.unsqueeze(-2)
            q_s = self.rope(q_s, token_positions)
            k_s = self.rope(k_s, token_positions)
        mask = torch.ones(q_s.shape[-2], k_s.shape[-2], device=q_s.device
                          ).tril().bool()
        attention_heads =self.scaled_dot_product_attention(q_s, k_s, v_s, mask)
        attention = rearrange(
            attention_heads, "... h s d -> ... s (h d)"
        )
        
        multi_head_attention = self.output_proj(attention)
        return multi_head_attention
