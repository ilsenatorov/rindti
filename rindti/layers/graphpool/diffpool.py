from math import ceil

import torch
import torch.nn.functional as F
import torch_geometric
from torch.functional import Tensor
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, dense_mincut_pool
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class DiffPoolNet(BaseLayer):
    """Differential Pooling module.

    Refer to :class:`torch_geometric.nn.dense.dense_diff_pool` and :class:`torch_geometric.nn.dense.dense_mincut_pool` for more details.

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

        self.max_nodes = ceil(max_nodes * 1.2)
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
        """"""

        x, _ = torch_geometric.utils.to_dense_batch(
            x, batch, max_num_nodes=self.max_nodes
        )
        adj = torch_geometric.utils.to_dense_adj(
            edge_index, batch, max_num_nodes=self.max_nodes
        )

        s = self.poolblock1(x, adj)  # (256, 140, 75)
        x = self.embedblock1(x, adj)  # (256, 140, 96)
        x, adj, lp_loss1, e_loss1 = self.pool(x, adj, s)

        s = self.poolblock2(x, adj)  # (256, 70, 35)
        x = self.embedblock2(x, adj)  # (256, 70, 96)
        x, adj, lp_loss2, e_loss2 = self.pool(x, adj, s)

        x = self.embedblock3(x, adj)  # (256, 35, 96)
        x = F.relu(x)
        x = x.mean(dim=1)
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.normalize(x, dim=1)
        return x


class DiffPoolBlock(torch.nn.Module):
    """Block of DiffPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, out_channels)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)

    def bn(self, i: int, x: Tensor) -> Tensor:
        """Apply batch normalisation.

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
        """"""
        x = self.bn(1, F.relu(self.conv1(x, adj)))
        return x
