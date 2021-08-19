from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.functional import Tensor

from ..layers import MutualInformation
from ..layers.graphconv import GINConvNet
from ..utils.data import TwoGraphData, corrupt_features
from .base_model import BaseModel, node_embedders, poolers


class InfoGraphModel(BaseModel):
    """Maximise mutual information between node and graph representations
    https://arxiv.org/pdf/1808.06670.pdf"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.feat_embed = self._get_feat_embed(kwargs)
        self.node_embed = self._get_node_embed(kwargs)
        self.pool = self._get_pooler(kwargs)
        self.mi = MutualInformation(kwargs["hidden_dim"], kwargs["hidden_dim"])
        self.node_pred = GINConvNet(kwargs["hidden_dim"], kwargs["feat_dim"], kwargs["hidden_dim"])

    def forward(self, data: TwoGraphData) -> Tuple[Tensor, Tensor]:
        """Forward pass of the module"""
        data["x"] = self.feat_embed(data["x"])
        data["x"] = self.node_embed(**data)
        graph_embed = self.pool(**data)
        node_index = torch.arange(data["x"].size(0), device=self.device)
        pair_index = torch.stack([data["batch"], node_index], dim=-1)
        mi = self.mi(graph_embed, data["x"], pair_index)
        node_pred = self.node_pred(**data)
        return mi, node_pred

    def shared_step(self, data: TwoGraphData) -> dict:
        """Shared step"""
        orig_x = data["x"].clone()
        cor_x, cor_idx = corrupt_features(data["x"], self.hparams.frac)
        data["x"] = cor_x
        mi, node_pred = self.forward(data.__dict__)
        node_loss = F.cross_entropy(node_pred[cor_idx], orig_x[cor_idx])
        loss = -mi + node_loss * self.hparams.alpha
        return {"loss": loss, "mi": mi.detach(), "node_loss": node_loss.detach()}

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module"""
        # Hack to find which embedding are used and add their arguments
        tmp_parser = ArgumentParser(add_help=False)
        tmp_parser.add_argument("--node_embed", type=str, default="ginconv")
        tmp_parser.add_argument("--pool", type=str, default="gmt")
        args = tmp_parser.parse_known_args()[0]

        node_embed = node_embedders[args.node_embed]
        pool = poolers[args.pool]
        parser.add_argument("--feat_embed_dim", default=32, type=int)
        parser.add_argument("--frac", default=0.15, type=float, help="Fraction of nodes to corrupt")
        parser.add_argument("--alpha", default=1.0, type=float)
        pooler_args = parser.add_argument_group("Pool", prefix="--")
        node_embed_args = parser.add_argument_group("Node embedding", prefix="--")
        node_embed.add_arguments(node_embed_args)
        pool.add_arguments(pooler_args)
        return parser
