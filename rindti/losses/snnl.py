from typing import List, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor


class SoftNearestNeighborLoss(LightningModule):
    """https://arxiv.org/pdf/1902.01889.pdf"""

    def __init__(
        self,
        temperature: float = 1.0,
        eps: float = 1e-6,
        optim_temperature: bool = False,
        grad_step: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.optim_temperature = optim_temperature
        self.grad_step = grad_step

    def _forward(self, embeds: Tensor, fam_idx: List[int], temp_frac: Union[int, Tensor]) -> Tensor:
        """Calculate the soft nearest neighbor loss for a given temp denominator"""
        embeds = F.normalize(embeds)
        sim = 1 - torch.matmul(embeds, embeds.t())
        expsim = torch.exp(-sim / (self.temperature / temp_frac)) * (1 - torch.eye(sim.size(0), device=self.device))
        f = expsim / (self.eps + expsim.sum(dim=1))
        fam_mask = (fam_idx == fam_idx.t()).float()
        f = f * fam_mask
        loss = -torch.log(self.eps + f.sum(dim=1))
        return dict(graph_loss=loss.mean())

    def forward(self, embeds: Tensor, fam_idx: List[int]) -> Tensor:
        """Calculate the soft nearest neighbor loss, optimise temperature if necessary"""
        if self.optim_temperature:
            temp_frac = torch.tensor(1, device=self.device, dtype=torch.float32, requires_grad=True)
            loss = self._forward(embeds, fam_idx, temp_frac)
            loss.mean().backward(inputs=[temp_frac])
            with torch.no_grad():
                temp_frac -= self.grad_step * temp_frac.grad
            return self._forward(embeds, fam_idx, temp_frac)
        else:
            return self._forward(embeds, fam_idx, 1.0)
