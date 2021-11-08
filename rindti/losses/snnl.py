from typing import List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor


class SoftNearestNeighborLoss(LightningModule):
    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeds: Tensor, fam_idx: List[List[int]]) -> Tensor:
        norm_emb = F.normalize(embeds, dim=1)
        sim = torch.cdist(norm_emb, norm_emb)
        expsim = torch.exp(-sim / self.temperature) * (1 - torch.eye(len(sim), device=self.device)) + 1e-6
        losses = []
        for idx in fam_idx:
            pos_idxt = torch.tensor(idx)
            pos = expsim[pos_idxt[:, None], pos_idxt]
            batch = expsim[:, pos_idxt]
            loss = -torch.log(torch.sum(pos, dim=0) / torch.sum(batch, dim=0))
            losses.append(loss)
        return torch.stack(losses).view(-1)
