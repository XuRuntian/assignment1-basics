import torch 
from torch import Tensor
from torch import nn
from jaxtyping import Float, Int

from .rmsnorm import RMSNorm
from .multihead_self_attention import MHA
from .swiglu import SwiGlu
class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 device= None,
                 dtype =None
                 ):
        
        super().__init__()
        
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.attn = MHA(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        self.ffn = SwiGlu(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        
    def forward(self, x: Float[Tensor, " batch sequence_length d_model"],):
        x1 = x + self.attn(self.ln1(x))
        return x1 + self.ffn(self.ln2(x1))
    