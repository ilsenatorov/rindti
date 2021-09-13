from argparse import ArgumentParser
from torch.functional import Tensor

import torch.nn.functional as F

from ..layers import MLP
from ..utils import MyArgParser, remove_arg_prefix
from ..utils.data import TwoGraphData
from ..utils.transforms import DataCorruptor
from .base_model import BaseModel, node_embedders, poolers


class PfamModel(BaseModel):
    """Model for Pfam class comparison problem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._determine_feat_method(kwargs["feat_method"], kwargs["hidden_dim"], kwargs["hidden_dim"])
        self.feat_embed = self._get_feat_embed(kwargs)
        self.node_embed = self._get_node_embed(kwargs)
        self.pool = self._get_pooler(kwargs)
        self.mlp = self._get_mlp(remove_arg_prefix("--mlp", kwargs))
        self.node_pred = self._get_node_embed(kwargs, out_dim=kwargs["feat_dim"])
        self.corruptor = DataCorruptor(dict(a_x=kwargs['frac'], b_x=kwargs['frac']), type="mask")

    def forward(self, a: dict, b: dict) -> Tensor:
        """Forward pass of the model"""
        a["x"] = self.feat_embed(a["x"])
        b["x"] = self.feat_embed(b["x"])
        a["x"] = self.node_embed(**a)
        b["x"] = self.node_embed(**b)
        a_embed = self.pool(**a)
        b_embed = self.pool(**b)
        a_pred = self.node_pred(**a)
        b_pred = self.node_pred(**b)
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
        loss = F.cosine_embedding_loss(a_embed, b_embed, labels.float())
        metrics = {}
        # metrics = self._get_classification_metrics(output, labels)
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
        parser.add_argument("--corruption", default="corruption", type=str)
        parser.add_argument("--feat_embed_dim", default=32, type=int)
        parser.add_argument("--feat_method", default="element_l1", type=str)
        parser.add_argument("--frac", default=0.1, type=float, help="Corruption percentage")
        parser.add_argument("--alpha", default=0.1, type=float, help="Weight of noisy node loss")
        pooler_args = parser.add_argument_group("Pool", prefix="--")
        node_embed_args = parser.add_argument_group("Node embedding", prefix="--")
        mlp_args = parser.add_argument_group("MLP", prefix="--mlp_")
        node_embed.add_arguments(node_embed_args)
        pool.add_arguments(pooler_args)
        MLP.add_arguments(mlp_args)
        return parser
