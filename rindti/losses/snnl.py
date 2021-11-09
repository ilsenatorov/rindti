from typing import List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor


class SoftNearestNeighborLoss(LightningModule):
    """https://arxiv.org/pdf/1902.01889.pdf"""

    def __init__(self, temperature: float = 1.0, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeds: Tensor, fam_idx: List[int]) -> Tensor:
        """Calculate the soft nearest neighbor loss"""
        losses = []
        embeds = F.normalize(embeds)
        sim = 1 - torch.matmul(embeds, embeds.t())
        expsim = torch.exp(-sim / self.temperature) - torch.eye(sim.size(0), device=self.device)
        f = expsim / (self.eps + expsim.sum(dim=1))
        fam_mask = (fam_idx == fam_idx.t()).float()
        f = f * fam_mask
        loss = -torch.log(self.eps + f.sum(dim=1))
        losses.append(loss)
        return torch.stack(losses).view(-1)
