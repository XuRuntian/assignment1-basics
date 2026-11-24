import torch
from collections.abc import Callable
from typing import Optional
import math
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas":betas,
            "eps":eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is  None:
                    continue
               

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)

                grad = p.grad.data
                p.data = p.data - p.data * lr * weight_decay 
                m = betas[0] * m + (1-betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad**2

                lr_t = lr * (math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t))

                p.data = p.data - lr_t * m / (torch.sqrt(v) + eps)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss