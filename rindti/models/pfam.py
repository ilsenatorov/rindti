from argparse import ArgumentParser
from collections import defaultdict

import torch
from torch.functional import Tensor

from ..data import TwoGraphData
from ..utils import MyArgParser
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
            fam_loss = generalised_lifted_structure_loss(pos_dist, neg_dist, margin=self.hparams.margin)
            loss.append(fam_loss)
            self.losses[fam].append(fam_loss.mean().item())
        return dict(loss=torch.cat(loss).mean())

    def training_epoch_end(self, outputs: dict):
        self.sampler.update_weights(self.losses)
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
