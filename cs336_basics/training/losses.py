import torch
from torch import nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, inputs, targets):
        max_batch = torch.amax(inputs, dim=-1, keepdim=True)
        inputs = inputs - max_batch

        loss = torch.mean(torch.sum(-torch.gather(input=inputs, dim=-1, index=targets.unsqueeze(-1)) + torch.log(torch.sum(torch.exp(inputs), dim=-1, keepdim=True)), dim=-1))

        return loss

        