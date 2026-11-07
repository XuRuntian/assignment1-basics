import torch
from torch import nn
from einops import rearrange
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k 必须是偶数")
        positions = torch.arange(0, max_seq_len, device=device).unsqueeze(1) #(seqlen, 1)
        k = torch.arange(1, d_k // 2 + 1, device=device).unsqueeze(0) #[)，需要多+2 (1, dk/2)
        thetas = positions / torch.pow(theta, (2 * k -2) / d_k) #(seqlen, dk/2)
        self.register_buffer(
            "cos_buffer",
            torch.cos(thetas),
            persistent=False
        )
        self.register_buffer(
            "sin_buffer",
            torch.sin(thetas),
            persistent=False
        )


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # token_position: (batch, sequence_pos)
        # xxx_selected:(batch, sequence_pos_selected, d/2)
        cos_selected = self.cos_buffer[token_positions] 
        sin_selected = self.sin_buffer[token_positions]

        # 拆分pair
        pairs = rearrange(
            x,
            "... seq_len (num_pairs pair_size) -> ... seq_len num_pairs pair_size",
            pair_size = 2
        )
        # (batch, seq_len, d/2)
        first, second = pairs.unbind(dim=-1)

        rope_first = cos_selected * first - sin_selected * second
        rope_second = sin_selected * first + cos_selected * second

        rotated_pairs = torch.stack((rope_first, rope_second), dim=-1)

        return rearrange(
            rotated_pairs,
            "... num_pairs pair_size -> ... (num_pairs pair_size)",
        )

        