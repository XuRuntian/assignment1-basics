import math

class LrCosSchedule():
    def __init__(self,    
                max_learning_rate: float,
                min_learning_rate: float,
                warmup_iters: int,
                cosine_cycle_iters: int,
                ):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
    def get_lr(self, it: int):
        if it < self.warmup_iters:
            return it / self.warmup_iters * self.max_learning_rate
        elif self.warmup_iters <= it < self.cosine_cycle_iters:
            return self.min_learning_rate + 1/2 * (1 + math.cos((it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters) * math.pi)) * (self.max_learning_rate - self.min_learning_rate)

        else:
            return self.min_learning_rate