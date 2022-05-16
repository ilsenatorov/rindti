from argparse import ArgumentParser

from torch.functional import Tensor
from torch.nn import Embedding
from torch_geometric.nn import PNAConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class PNAConvNet(BaseLayer):
    r"""Principal Neighborhood Aggregation.

    Refer to :class:`torch_geometric.nn.conv.PNAConv` for more details.

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden layer. Defaults to 32.
        edge_dim (int, optional): Size of the edge dim. Defaults to None.
        deg (Tensor, optional): Degree distribution. Defaults to None.
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 32, edge_dim: int = None, deg: Tensor = None, **kwargs
    ):
        super().__init__()
        self.edge_embedding = Embedding(edge_dim, edge_dim)
        self.conv1 = PNAConv(
            input_dim,
            hidden_dim,
            aggregators=["sum", "mean", "max"],
            scalers=["linear", "identity"],
            edge_dim=edge_dim,
            deg=deg,
        )
        self.conv2 = PNAConv(
            hidden_dim,
            output_dim,
            aggregators=["sum", "mean", "max"],
            scalers=["linear", "identity"],
            edge_dim=edge_dim,
            deg=deg,
        )

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """"""
        x = self.inp(x, edge_index)
        for module in self.mid_layers:
            x = module(x, edge_index)
        x = self.out(x, edge_index)
        return x
