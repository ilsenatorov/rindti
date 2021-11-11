from argparse import ArgumentParser
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.functional import Tensor
from torchmetrics.functional import confusion_matrix

from ...data import DataCorruptor, TwoGraphData
from ...losses import GeneralisedLiftedStructureLoss, NodeLoss, SoftNearestNeighborLoss
from ...utils import MyArgParser
from ..base_model import BaseModel, node_embedders, poolers
from ..encoder import Encoder


class PfamModel(BaseModel):
    """Model for Pfam class comparison problem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.node_pred = self._get_node_embed(kwargs, kwargs["feat_dim"])
        self.encoder = Encoder(return_nodes=True, **kwargs)
        self.loss = {"snnl": SoftNearestNeighborLoss, "lifted": GeneralisedLiftedStructureLoss}[kwargs["loss"]](
            **kwargs
        )
        self.node_loss = NodeLoss(**kwargs)
        self.masker = DataCorruptor(dict(x=self.hparams.frac), type="mask")

    @property
    def fam_idx(self) -> List[int]:
        """Using batch_size and prot_per_fam, get idx of each family

        Returns:
            List[List]: First list is families, second list is entries in the family
        """
        res = []
        fam = 0
        for _ in range(0, self.hparams.batch_size, self.hparams.prot_per_fam):
            res += [fam] * self.hparams.prot_per_fam
            fam += 1
        return res

    def forward(self, data: dict) -> Tensor:
        """Forward pass of the model"""
        data = self.masker(data)
        embeds, node_embeds = self.encoder(data)
        node_preds = self.node_pred(node_embeds, data.edge_index)
        return embeds, node_preds

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        embeds, node_preds = self.forward(data)
        fam_idx = torch.tensor(
            self.fam_idx,
            dtype=torch.long,
            device=self.device,
        ).view(-1, 1)
        node_metrics = self.node_loss(node_preds[data["x_idx"]], data["x_orig"] - 1)
        loss = self.loss(embeds, fam_idx).mean()
        node_metrics.update(
            dict(
                loss=loss + node_metrics["node_loss"] * self.hparams.alpha,
                graph_loss=loss,
            )
        )
        res = {}
        for k, v in node_metrics.items():
            if k != "loss":
                res[k] = v.detach()
            else:
                res[k] = v
        return res

    def log_node_confusionmatrix(self, confmatrix: Tensor):
        """Saves the confusion matrix of node prediction

        Args:
            confmatrix (Tensor): 20x20 matrix
        """
        fig = plt.figure()
        sns.heatmap(confmatrix.detach().cpu())
        self.logger.experiment.add_figure("confmatrix", fig, global_step=self.global_step)

    def log_distmap(self, data: TwoGraphData, embeds: Tensor):
        """Plot and save distance matrix of this batch"""
        fig = plt.figure()
        sns.heatmap(torch.cdist(embeds, embeds).detach().cpu())
        self.logger.experiment.add_figure("distmap", fig, global_step=self.global_step)
        self.logger.experiment.add_embedding(embeds, metadata=data.fam, global_step=self.global_step)
