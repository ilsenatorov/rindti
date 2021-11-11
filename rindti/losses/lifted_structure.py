from typing import List

import torch
from pytorch_lightning import LightningModule
from torch import Tensor


class GeneralisedLiftedStructureLoss(LightningModule):
    """https://arxiv.org/abs/1511.06452"""

    def __init__(self, pos_margin: int = 0, neg_margin: int = 1, **kwargs) -> None:
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, embeds: Tensor, fam_idx: List[int]) -> Tensor:
        """Calculate the loss"""
        dist = torch.cdist(embeds, embeds)
        fam_mask = (fam_idx == fam_idx.t()).float()
        pos = (dist - self.pos_margin) * fam_mask
        neg = (self.neg_margin - dist) * (1 - fam_mask)
        pos_loss = torch.logsumexp(pos, dim=0)
        neg_loss = torch.logsumexp(neg, dim=0)
        return torch.relu(pos_loss + neg_loss)