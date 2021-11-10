from argparse import ArgumentParser
from typing import Tuple

import torch
from torch.functional import Tensor
from torch_geometric.data import Data

from ...data import DataCorruptor
from ...layers import MutualInformation
from ...losses import NodeLoss
from ..base_model import BaseModel, node_embedders, poolers
from ..encoder import Encoder


class InfoGraphModel(BaseModel):
    """Maximise mutual information between node and graph representations
    https://arxiv.org/pdf/1808.06670.pdf"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(return_nodes=True, **kwargs)
        self.mi = MutualInformation(kwargs["hidden_dim"], kwargs["hidden_dim"])
        self.node_pred = self._get_node_embed(kwargs, out_dim=kwargs["feat_dim"] + 1)
        self.masker = DataCorruptor(dict(x=self.hparams.frac), type="mask")
        self.node_loss = NodeLoss(**kwargs)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """Forward pass of the module"""
        data = self.masker(data)
        node_index = torch.arange(data["x"].size(0), device=self.device)
        pair_index = torch.stack([data["batch"], node_index], dim=-1)
        graph_embed, node_embed = self.encoder(data)
        mi = self.mi(graph_embed, node_embed, pair_index)
        node_pred = self.node_pred(node_embed, data.edge_index)
        return mi, node_pred

    def shared_step(self, data: Data) -> dict:
        """Shared step"""
        mi, node_preds = self.forward(data)
        node_metrics = self.node_loss(node_preds[data["x_idx"]], data["x_orig"] - 1)
        node_metrics.update(
            dict(
                loss=-mi + node_metrics["node_loss"] * self.hparams.alpha,
                mi=mi,
            )
        )
        res = {}
        for k, v in node_metrics.items():
            if k != "loss":
                res[k] = v.detach()
            else:
                res[k] = v
        return res

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
        parser.add_argument("--frac", default=0.15, type=float)
        parser.add_argument("--corruption", default="corrupt", type=str)
        parser.add_argument("--alpha", default=1.0, type=float)
        pooler_args = parser.add_argument_group("Pool", prefix="--")
        node_embed_args = parser.add_argument_group("Node embedding", prefix="--")
        node_embed.add_arguments(node_embed_args)
        pool.add_arguments(pooler_args)
        return parser
