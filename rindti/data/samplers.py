import random
from collections import defaultdict
from os import replace
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

from ..utils import minmax_normalise, to_prob
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
        self.prot_idx = {}
        for i, data in enumerate(self.dataset):
            self.fam_idx[data.fam].add(i)
            self.prot_idx[data.id] = i
        self.fam_idx = {k: list(v) for k, v in self.fam_idx.items()}

    def _construct_batch(self) -> List[int]:
        """Creates a single batch. takes self.prot_per_fam families and samples them

        Returns:
            List[int]: Indices of the proteins from the main dataset
        """
        batch = []
        anchor_fams = np.random.choice(
            list(self.fam_idx.keys()),
            size=self.batch_size // self.prot_per_fam,
            replace=False,
        )
        for fam in anchor_fams:
            batch += list(np.random.choice(self.fam_idx[fam], size=self.prot_per_fam, replace=False))
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
        minprob (float, optional): lowest weighted probability. Defaults to 0.1
        batch_size (int, optional): Defaults to 64.
        prot_per_fam (int, optional): Number of proteins per family. Defaults to 8
        batch_per_epoch (int, optional): Number of batches per epoch. Defaults to 1000

    """

    def __init__(self, *args, minprob: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.minprob = minprob
        self.prot_weights = {k: 1000 for k in range(len(self.dataset))}
        self.fam_weights = pd.Series({k: 1000 * len(v) for k, v in self.fam_idx.items()})

    def update_fam_weights(self):
        """Get minmax normalised weights"""
        fam_weights = defaultdict(list)
        for fam_id, fam in self.fam_idx.items():
            for prot in fam:
                fam_weights[fam_id].append(self.prot_weights[prot])
        self.fam_weights.update({k: np.mean(v) for k, v in fam_weights.items()})
        self.fam_weights = minmax_normalise(self.fam_weights) + self.minprob
        self.fam_weights = self.fam_weights / self.fam_weights.sum()

    def update_weights(self, losses: Dict[str, Iterable]):
        """Update sampling weight of families

        Args:
            losses (Dict[str, Iterable]): family ids and their respective losses
        """
        self.prot_weights.update({self.prot_idx[k]: np.mean(v) for k, v in losses.items()})
        self.update_fam_weights()

    def _construct_batch(self) -> List[int]:
        """Creates a single batch. takes self.prot_per_fam families with
        self.fam_weights probabilities and samples them

        Returns:
            List[int]: Indices of the proteins from the main dataset
        """
        batch = []
        anchor_fams = list(
            np.random.choice(
                list(self.fam_weights.index),
                p=to_prob(self.fam_weights.values),
                size=self.batch_size // self.prot_per_fam,
                replace=False,
            )
        )
        for fam in anchor_fams:
            batch += list(
                np.random.choice(
                    self.fam_idx[fam],
                    p=to_prob(np.asarray([self.prot_weights[x] for x in self.fam_idx[fam]])),
                    size=self.prot_per_fam,
                    replace=False,
                )
            )
        return batch
