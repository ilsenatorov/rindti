from argparse import ArgumentParser
from collections import defaultdict

import pandas as pd
import torch.nn.functional as F
from torch.functional import Tensor

from ..utils import MyArgParser, remove_arg_prefix
from ..utils.data import TwoGraphData
from .base_model import BaseModel, node_embedders, poolers
from .encoder import Encoder


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
        anchor = self.forward(data)
        loss = F.triplet_margin_loss(anchor, pos, neg)
        return dict(loss=loss)

    def update_transformer(self):
        self.transformer.update_weights(self.losses)

    def validation_epoch_end(self, outputs: dict):
        self.update_transformer()
        pd.Series(self.losses).to_pickle("losses{}.pkl".format(self.current_epoch))
        return super().validation_epoch_end(outputs)

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
        return parser
