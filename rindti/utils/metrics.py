from typing import Any
from statistics import NormalDist

import torch
from torchmetrics import Metric


class DistOverlap(Metric):
    def __init__(self):
        super(DistOverlap, self).__init__()
        self.add_state("pos", default=[], dist_reduce_fx="cat")
        self.add_state("neg", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pos = preds[target == 1]
        neg = preds[target == 0]
        self.pos += pos
        self.neg += neg

    def compute(self) -> Any:
        pos_mu, pos_sigma = torch.mean(self.pos), torch.std(self.pos)
        neg_mu, neg_sigma = torch.mean(self.pos), torch.std(self.pos)

        return torch.tensor(NormalDist(mu=pos_mu.item(), sigma=pos_sigma.item()).overlap(
            NormalDist(mu=neg_mu.item(), sigma=neg_sigma.item())
        ))


