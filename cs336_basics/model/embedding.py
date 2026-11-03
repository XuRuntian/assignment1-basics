import torch
from torch import nn

class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        pass