from typing import List

import torch
from pytorch_lightning import LightningModule
from torch import Tensor


class GeneralisedLiftedStructureLoss(LightningModule):
    """https://arxiv.org/abs/1511.06452"""

    def __init__(self, pos_margin: int, neg_margin: int) -> None:
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, embeds: Tensor, fam_idx: List[List[int]]) -> Tensor:
        """Calculate the loss"""
        batch_size = embeds.size(0)
        losses = []
        dist = torch.cdist(embeds, embeds)
        all_idx = set(range(batch_size))
        for idx in fam_idx:
            pos_idxt = torch.tensor(idx)
            neg_idxt = torch.tensor(list(all_idx.difference(idx)))
            pos = dist[pos_idxt[:, None], pos_idxt]
            neg = dist[neg_idxt[:, None], pos_idxt]
            pos_loss = torch.logsumexp(pos - self.pos_margin, dim=0)
            neg_loss = torch.logsumexp(self.neg_margin - neg, dim=0)
            losses.append(torch.relu(pos_loss + neg_loss) ** 2)
        return torch.cat(losses).view(-1)
