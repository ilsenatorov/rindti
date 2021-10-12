from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.functional import Tensor

from ..data import TwoGraphData
from ..utils import MyArgParser, plot_loss_count_dist
from .base_model import BaseModel, node_embedders, poolers
from .encoder import Encoder


def generalised_lifted_structure_loss(pos_dist: Tensor, neg_dist: Tensor, margin: float = 1.0) -> Tensor:
    """
    https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html

    Args:
        pos_dist (Tensor): distances between positive pairs
        neg_dist (Tensor): distances between negative pairs
        margin (float, optional): alpha, margin. Defaults to 1.0.

    Returns:
        Tensor: resulting loss
    """
    pos_loss = torch.logsumexp(pos_dist, dim=0)
    neg_loss = torch.logsumexp(margin - neg_dist, dim=0)
    return torch.relu(pos_loss + neg_loss)


class PfamModel(BaseModel):
    """Model for Pfam class comparison problem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(return_nodes=False, **kwargs)
        self.losses = defaultdict(list)

    def forward(self, data: dict) -> Tensor:
        """Forward pass of the model"""
        return self.encoder(data)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        embeds = self.forward(data)
        dist = torch.cdist(embeds, embeds)
        if self.global_step % 100 == 1:
            fig = plt.figure()
            sns.heatmap(dist.detach().cpu())
            self.logger.experiment.add_figure("distmap", fig, global_step=self.global_step)
            self.logger.experiment.add_embedding(embeds, metadata=data.fam, global_step=self.global_step)
        fam_idx = defaultdict(list)
        all_idx = set(list(range(len(data.id))))
        for idx, fam in enumerate(data.fam):
            fam_idx[fam].append(idx)
        loss = []
        for fam, idx in fam_idx.items():
            pos_idxt = torch.tensor(list(idx))
            neg_idxt = torch.tensor(list(all_idx.difference(idx)))
            pos_dist = dist[pos_idxt[:, None], pos_idxt]
            neg_dist = dist[neg_idxt[:, None], pos_idxt]
            loss.append(generalised_lifted_structure_loss(pos_dist, neg_dist, margin=self.hparams.margin))
        loss = torch.cat(loss)
        for l, name in zip(loss, data.id):
            self.losses[name].append(l.item())
        return dict(loss=loss.mean())

    def training_epoch_end(self, outputs: dict):
        self.sampler.update_weights(self.losses)
        self.logger.experiment.add_figure("loss_dist", plot_loss_count_dist(self.losses), global_step=self.global_step)
        return super().training_epoch_end(outputs)

    @staticmethod
    def add_arguments(parser: MyArgParser) -> MyArgParser:
        """Generate arguments for this module"""
        # Hack to find which embedding are used and add their arguments
        tmp_parser = ArgumentParser(add_help=False)
        tmp_parser.add_argument("--node_embed", type=str, default="ginconv")
        tmp_parser.add_argument("--pool", type=str, default="gmt")
        args = tmp_parser.parse_known_args()[0]

        node_embed = node_embedders[args.node_embed]
        pool = poolers[args.pool]
        pooler_args = parser.add_argument_group("Pool", prefix="--")
        node_embed_args = parser.add_argument_group("Node embedding", prefix="--")
        node_embed.add_arguments(node_embed_args)
        pool.add_arguments(pooler_args)
        parser.add_argument("--margin", type=float, default=1)
        parser.add_argument("--prot_per_fam", type=int, default=8)
        parser.add_argument("--batch_per_epoch", type=int, default=1000)
        return parser
