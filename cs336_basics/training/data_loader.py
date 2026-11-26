import numpy as np
import torch
import numpy.typing as npt


class TokenBatchSampler():
    def __init__(self, batch_size: int, context_length: int, device: str, seed: int=None):
        
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.rng = np.random.default_rng(seed=seed)

    def get_batch(self, dataset: npt.NDArray):
        """
        dataset 是一个一维 NumPy 整数数组，里面按顺序保存已经分词并编码好的 token ID。
        """
        data_length = len(dataset)
        inputs = np.empty((self.batch_size, self.context_length), dtype=int) # token id是整数
        targets = np.empty((self.batch_size, self.context_length), dtype=int)

        inputs_start_nums = self.rng.integers(low=0, high=data_length - self.context_length, size=self.batch_size)


        for i, nums in enumerate(inputs_start_nums):
            inputs[i] = dataset[nums: nums + self.context_length]
            targets[i] = dataset[nums + 1: nums + self.context_length + 1]

        return (torch.tensor(inputs, dtype=torch.long, device=self.device), torch.tensor(targets, dtype=torch.long, device=self.device))