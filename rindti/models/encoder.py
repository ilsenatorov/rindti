from argparse import ArgumentParser
from typing import Tuple, Union

from torch import nn
from torch.functional import Tensor
from torch_geometric.data import Data

from .base_model import BaseModel


class Encoder(BaseModel):
    """Encoder for graphs"""

    def __init__(self, feat_embed: str = "bag", return_nodes: bool = False, **kwargs):
        super().__init__()
        if feat_embed == "bag":
            self.feat_embed = self._get_feat_embed(kwargs)
        elif feat_embed == "linear":
            self.feat_embed = nn.LazyLinear(kwargs["hidden_dim"], bias=False)
        elif feat_embed == "none":
            self.feat_embed = None
        self.node_embed = self._get_node_embed(kwargs)
        self.pool = self._get_pooler(kwargs)
        self.return_nodes = return_nodes

    def forward(
        self,
        data: Data,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Encode a graph

        Args:
            data (Data): torch_geometric - 'x', 'edge_index' etc

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Either graph of graph+node embeddings
        """
        x, edge_index, batch = data["x"], data["edge_index"], data["batch"]
        if self.feat_embed is not None:
            feat_embed = self.feat_embed(x)
        node_embed = self.node_embed(feat_embed, edge_index)
        embed = self.pool(node_embed, edge_index, batch)
        if self.return_nodes:
            return embed, node_embed
        return embed

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module"""
        parser.add_argument("--feat_embedding", default="bag", type=str)
        return parser
