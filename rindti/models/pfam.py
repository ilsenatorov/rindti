from argparse import ArgumentParser
from copy import copy

import torch.nn.functional as F
from torch.functional import Tensor

from rindti.layers.graphconv.ginconv import GINConvNet

from ..utils import MyArgParser, remove_arg_prefix
from ..utils.data import TwoGraphData
from ..utils.transforms import DataCorruptor
from .base_model import BaseModel, node_embedders, poolers
from .encoder import Encoder


class PfamModel(BaseModel):
    """Model for Pfam class comparison problem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._determine_feat_method(kwargs["feat_method"], kwargs["hidden_dim"], kwargs["hidden_dim"])
        self.encoder = Encoder(return_nodes=True, **kwargs)
        self.node_pred = GINConvNet(kwargs["feat_embed_dim"], kwargs["hidden_dim"], kwargs["hidden_dim"], 3)
        self.corruptor = DataCorruptor(dict(a_x=kwargs["frac"], b_x=kwargs["frac"]), type="mask")

    def forward(self, a: dict, b: dict) -> Tensor:
        """Forward pass of the model"""
        a_embed, a_pred = self.encoder(a)
        b_embed, b_pred = self.encoder(b)
        return a_embed, b_embed, a_pred, b_pred

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        cor_data = self.corruptor(data)
        a = remove_arg_prefix("a_", cor_data)
        b = remove_arg_prefix("b_", cor_data)
        a_embed, b_embed, a_pred, b_pred = self.forward(a, b)
        labels = data.label
        labels[labels == 0] = -1
        loss = F.cosine_embedding_loss(a_embed, b_embed, labels.float(), margin=0.3)
        metrics = {}
        # metrics = self._get_class_metrics(output, labels)
        a_idx = cor_data.a_idx
        b_idx = cor_data.b_idx
        a_loss = F.cross_entropy(a_pred[a_idx], data["a_x"][a_idx])
        b_loss = F.cross_entropy(b_pred[b_idx], data["b_x"][b_idx])
        metrics.update(
            dict(
                loss=loss + self.hparams.alpha * a_loss + self.hparams.alpha * b_loss,
                a_loss=a_loss.detach(),
                b_loss=b_loss.detach(),
                pred_loss=loss.detach(),
            )
        )
        return metrics

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
        parser.add_argument("--corruption", default="corrupt", type=str)
        parser.add_argument("--feat_embed_dim", default=32, type=int)
        parser.add_argument("--feat_method", default="element_l1", type=str)
        parser.add_argument("--frac", default=0.1, type=float, help="Corruption percentage")
        parser.add_argument("--alpha", default=0.1, type=float, help="Weight of noisy node loss")
        pooler_args = parser.add_argument_group("Pool", prefix="--")
        node_embed_args = parser.add_argument_group("Node embedding", prefix="--")
        node_embed.add_arguments(node_embed_args)
        pool.add_arguments(pooler_args)
        return parser
