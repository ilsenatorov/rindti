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
        # self._determine_feat_method(kwargs["feat_method"], kwargs["hidden_dim"], kwargs["hidden_dim"])
        self.encoder = Encoder(return_nodes=False, **kwargs)
        # self.node_pred = GINConvNet(kwargs["feat_embed_dim"], kwargs["hidden_dim"], kwargs["hidden_dim"], 3)
        # self.corruptor = DataCorruptor(dict(a_x=kwargs["frac"], b_x=kwargs["frac"]), type="mask")

    def forward(self, a: dict, b: dict) -> Tensor:
        """Forward pass of the model"""
        a_embed = self.encoder(a)
        b_embed = self.encoder(b)
        return a_embed, b_embed

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        cor_data = data
        a = remove_arg_prefix("a_", cor_data)
        b = remove_arg_prefix("b_", cor_data)
        a_embed, b_embed = self.forward(a, b)
        labels = data.label
        loss = F.cosine_embedding_loss(a_embed, b_embed, labels.float(), margin=0.5)
        return dict(loss=loss)

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
