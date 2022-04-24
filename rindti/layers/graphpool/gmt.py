from argparse import ArgumentParser

import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch_geometric.nn import GraphMultisetTransformer
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class GMTNet(BaseLayer):
    """https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.GraphMultisetTransformer"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        ratio: float = 0.25,
        max_nodes: int = 600,
        num_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.pool = GraphMultisetTransformer(
            input_dim,
            hidden_dim,
            output_dim,
            num_nodes=max_nodes * 1.5,
            pooling_ratio=ratio,
            num_heads=num_heads,
            pool_sequences=["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"],
            layer_norm=True,
        )

    def forward(self, x: Tensor, edge_index: Adj, batch: LongTensor) -> Tensor:
        """Forward pass"""
        embeds = self.pool(x, batch, edge_index)
        return F.normalize(embeds, dim=1)
