from argparse import ArgumentParser
from math import ceil

import torch
import torch.nn.functional as F
import torch_geometric
from torch.functional import Tensor
from torch.nn import Embedding
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, dense_mincut_pool
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class DiffPoolNet(BaseLayer):
    """Differential Pooling module

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        max_nodes (int, optional): Maximal number of nodes in a graph. Defaults to 600.
        dropout (float, optional): Dropout ratio. Defaults to 0.2.
        ratio (float, optional): Pooling ratio. Defaults to 0.25.
        pooling_method (str, optional): Type of pooling. Defaults to "mincut".
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        max_nodes: int = 600,
        dropout: float = 0.2,
        ratio: float = 0.25,
        pooling_method: str = "mincut",
        **kwargs,
    ):
        super().__init__()

        self.max_nodes = max_nodes
        self.dropout = dropout

        self.pool = {
            "diffpool": dense_diff_pool,
            "mincut": dense_mincut_pool,
        }[pooling_method]

        num_nodes = ceil(self.max_nodes * ratio)
        self.poolblock1 = DiffPoolBlock(input_dim, num_nodes)
        self.embedblock1 = DiffPoolBlock(input_dim, hidden_dim)
        num_nodes = ceil(num_nodes * ratio)
        self.poolblock2 = DiffPoolBlock(hidden_dim, num_nodes)
        self.embedblock2 = DiffPoolBlock(hidden_dim, hidden_dim)

        self.embedblock3 = DiffPoolBlock(hidden_dim, hidden_dim)
        self.lin1 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, edge_index: Adj, batch: Tensor, **kwargs) -> Tensor:
        """Forward pass of the module

        Args:
            x (Tensor): Node features
            edge_index (Adj): Edge information
            batch (Tensor): Batch information

        Returns:
            Tensor: Graph representation vector
        """
        if self.node_embed:
            x = self.node_embedding(x)

        # Need adjacency for diffpool
        x, _ = torch_geometric.utils.to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch, max_num_nodes=self.max_nodes)

        # x = (256, 140, 30)
        # adj = (256, 140, 140)
        # xs = []

        s = self.poolblock1(x, adj)  # (256, 140, 75)
        x = self.embedblock1(x, adj)  # (256, 140, 96)
        # xs.append(x)
        x, adj, lp_loss1, e_loss1 = self.pool(x, adj, s)
        # x = (256, 70, 96)
        # adj = (256, 70, 70)

        s = self.poolblock2(x, adj)  # (256, 70, 35)
        x = self.embedblock2(x, adj)  # (256, 70, 96)
        # xs.append(x)
        x, adj, lp_loss2, e_loss2 = self.pool(x, adj, s)
        # x = (256, 35, 96)
        # adj = (256, 35, 35)

        x = self.embedblock3(x, adj)  # (256, 35, 96)
        # xs.append(x)
        # x = self.att(x, batch)

        # x = self.jump([x.mean(dim=1) for x in xs])  # (256, 96)
        x = F.relu(x)
        x = x.mean(dim=1)
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module

        Args:
            parser (ArgumentParser): Parent parser

        Returns:
            ArgumentParser: Updated parser
        """
        parser.add_argument("pool", default="gmt", type=str)
        parser.add_argument("hidden_dim", default=32, type=int, help="Size of output vector")
        parser.add_argument("ratio", default=0.25, type=float, help="Pooling ratio")
        parser.add_argument("dropout", default=0.25, type=float, help="Dropout ratio")
        parser.add_argument("pooling_method", default="mincut", type=str, help="Type of pooling")
        return parser


class DiffPoolBlock(torch.nn.Module):
    """Block of DiffPool

    Args:
        in_channels (int): Input size
        out_channels (int): Output size
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, out_channels)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)

    def bn(self, i: int, x: Tensor) -> Tensor:
        """Apply batch normalisation

        Args:
            i (int): layer idx
            x (Tensor): Node features

        Returns:
            Tensor: Updated node features
        """

        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, "bn{}".format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x: Tensor, adj: Adj) -> Tensor:
        """Single pass on a batch

        Args:
            x (Tensor): Node features
            adj (Adj): Adjacency matrix

        Returns:
            Tensor: Updated node features
        """
        x = self.bn(1, F.relu(self.conv1(x, adj)))
        return x
