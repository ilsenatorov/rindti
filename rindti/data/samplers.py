import random
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np
from torch.utils.data import Sampler

from .datasets import PreTrainDataset


class PfamSampler(Sampler):
    """Sampler for generating pfam-conforming batches.
    Ensure that in each batch there are positive and negative matches for each protein

    Args:
        dataset (PreTrainDataset): dataset, each data point has to contain data.fam
        batch_size (int, optional): Defaults to 64.
        prot_per_fam (int, optional): Number of proteins per family. Defaults to 8
        batch_per_epoch (int, optional): Number of batches per epoch. Defaults to 1000

    """

    def __init__(
        self,
        dataset: PreTrainDataset,
        batch_size: int = 64,
        prot_per_fam: int = 8,
        batch_per_epoch: int = 1000,
        **kwargs,
    ):
        assert batch_size % prot_per_fam == 0, "Batch size should be divisible by prot_per_fam!"
        self.dataset = dataset
        self.batch_size = batch_size
        self.prot_per_fam = prot_per_fam
        self.batch_per_epoch = batch_per_epoch
        self.fam_idx = defaultdict(set)
        for i, data in enumerate(self.dataset):
            self.fam_idx[data.fam].add(i)
        self.fam_idx = {k: list(v) for k, v in self.fam_idx.items()}

    def _construct_batch(self) -> List[int]:
        """Creates a single batch. takes self.prot_per_fam families and samples them

        Returns:
            List[int]: Indices of the proteins from the main dataset
        """
        batch = []
        anchor_fams = random.choices(list(self.fam_idx.keys()), k=self.batch_size // self.prot_per_fam)
        for fam in anchor_fams:
            batch += random.choices(self.fam_idx[fam], k=self.prot_per_fam)
        return batch

    def __iter__(self) -> iter:
        """Returns iterator over all the batches in the epoch

        Returns:
            iter: over list of lists, each list is one batch of indices
        """
        return iter([self._construct_batch() for _ in range(self.batch_per_epoch)])

    def __len__(self):
        return self.batch_per_epoch


class WeightedPfamSampler(PfamSampler):
    """Weighted for generating pfam-conforming batches.
    Ensure that in each batch there are positive and negative matches for each protein.
    Picks families with certain probabilities, based on losses

    Args:
        dataset (PreTrainDataset): dataset, each data point has to contain data.fam
        batch_size (int, optional): Defaults to 64.
        prot_per_fam (int, optional): Number of proteins per family. Defaults to 8
        batch_per_epoch (int, optional): Number of batches per epoch. Defaults to 1000

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fam_weights = {k: 1000 for k in self.fam_idx.keys()}

    def update_weights(self, losses: Dict[str, Iterable]):
        """Update sampling weight of families

        Args:
            losses (Dict[str, Iterable]): family ids and their respective losses
        """
        self.fam_weights.update({k: np.mean(v) for k, v in losses.items()})

    def _construct_batch(self) -> List[int]:
        batch = []
        anchor_fams = random.choices(
            list(self.fam_weights.keys()),
            weights=self.fam_weights.values(),
            k=self.batch_size // self.prot_per_fam,
        )
        for fam in anchor_fams:
            batch += random.choices(self.fam_idx[fam], k=self.prot_per_fam)
        return batch
