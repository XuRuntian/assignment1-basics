import torch 
from torch import Tensor
from torch import nn
from jaxtyping import Float, Int

from .rmsnorm import RMSNorm
from .multihead_self_attention import MHA
from .swiglu import SwiGlu
from .embedding import Embedding
from .linear import Linear
from .softmax import SoftMax

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

class TransformerLM(nn.Module):
    def __init__(self, 
                vocab_size:int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,
                device= None,
                dtype =None
                ):
        super().__init__()

        self.token_embeddings = Embedding(embedding_dim=d_model, num_embeddings=vocab_size, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList(
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta, device=device, dtype=dtype) 
            for _ in range(num_layers)
        )

        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
    