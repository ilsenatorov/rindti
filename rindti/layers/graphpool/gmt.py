import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch_geometric.nn import GraphMultisetTransformer
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class GMTNet(BaseLayer):
    """Graph Multiset Transformer pooling.

    Refer to :class:`torch_geometric.nn.glob.GraphMultisetTransformer` for more details.

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden layer(s). Defaults to 128.
        ratio (float, optional): Ratio of the number of nodes to be pooled. Defaults to 0.25.
        max_nodes (int, optional): Maximal number of nodes in a graph. Defaults to 600.
        num_heads (int, optional): Number of heads. Defaults to 4.
    """

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
        """Forward the data through the GNN module"""
        embeds = self.pool(x, batch, edge_index=edge_index)
        return F.normalize(embeds, dim=1)
