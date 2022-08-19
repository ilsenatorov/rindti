from typing import Iterable, List

import torch
from torch.utils.data.sampler import Sampler
from torch_geometric.data import Dataset


class DynamicBatchSampler(Sampler):
    r"""Sampler that fills up the batch until `max_num` nodes"""

    def __init__(self, dataset: Dataset, max_num: int):
        self.dataset = dataset
        self.max_num = max_num

    def __iter__(self) -> Iterable[List[int]]:
        idx = 0
        while idx < len(self.dataset):  # create whole epoch
            batch = []
            num_nodes = 0
            while idx < len(self.dataset):  # create single batch
                i = self.dataset[idx]
                num_nodes += i.num_nodes
                if num_nodes > self.max_num:  # if latest addition is too much, finish batch
                    break
                else:  # if latest addition is not too much, continue
                    batch.append(idx)
                    idx += 1
            yield batch

    def __len__(self) -> int:
        return len(self.dataset)
