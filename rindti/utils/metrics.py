from typing import Any
from statistics import NormalDist

import torch
from torchmetrics import Metric


class DistOverlap(Metric):
    """A metric keeping track of the distributions of predicted values for positive and negative samples"""
    def __init__(self):
        super(DistOverlap, self).__init__()
        self.add_state("pos", default=[], dist_reduce_fx="cat")
        self.add_state("neg", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Store the predictions separated into those of positive samples and those of negative samples"""
        pos = preds[target == 1]
        neg = preds[target == 0]
        self.pos += pos
        self.neg += neg

    def compute(self) -> Any:
        """Calculate the metric based on the samples from the update rounds"""
        self.pos = torch.stack(self.pos)
        self.neg = torch.stack(self.neg)
        pos_mu, pos_sigma = torch.mean(self.pos), torch.std(self.pos)
        neg_mu, neg_sigma = torch.mean(self.neg), torch.std(self.neg)

        return torch.tensor(NormalDist(mu=pos_mu.item(), sigma=pos_sigma.item()).overlap(
            NormalDist(mu=neg_mu.item(), sigma=neg_sigma.item())
        ))


