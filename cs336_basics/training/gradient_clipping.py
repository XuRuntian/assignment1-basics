from typing import Iterable
import torch
import math
class GradClip():
    def __init__(self, max_l2_norm: float, eps: float = 1e-6):
        self.max_l2_norm = max_l2_norm
        self.eps = eps

    def gradient_clipping(self, parameters):
        l2_norm = 0
        square_sum = 0
        paraments_list = list(parameters)
        for p in paraments_list:
            if p.grad is None:
                continue
            square_sum += torch.square(p.grad).sum()
        if square_sum !=0:
            l2_norm = torch.sqrt(square_sum)
        if l2_norm > self.max_l2_norm:
            for p in paraments_list:
                if p.grad is not None:
                    p.grad.mul_(self.max_l2_norm / (l2_norm + self.eps))
 