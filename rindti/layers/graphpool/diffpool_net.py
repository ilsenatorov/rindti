from math import ceil

import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn import Embedding
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, dense_mincut_pool

from ..base_layer import BaseLayer


class DiffPoolNet(BaseLayer):
    @staticmethod
    def add_arguments(group):
        group.add_argument("pool", default="gmt", type=str)
        group.add_argument("hidden_dim", default=32, type=int, help="Size of output vector")
        group.add_argument("ratio", default=0.25, type=float, help="Pooling ratio")
        group.add_argument("dropout", default=0.25, type=float, help="Dropout ratio")
        group.add_argument("pooling_method", default="mincut", type=str, help="Type of pooling")
        return group

    def __init__(
        self,
        input_dim,
        output_dim,
        max_nodes=600,
        hidden_dim=128,
        dropout=0.2,
        ratio=0.25,
        pooling_method="mincut",
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, **kwargs):
        """
        Calculate the embedding
        :param x: node_features of dimension (num_nodes_in_batch, feat_dim)
        :param edge_index: sparse connectivity of dimension (num_edges_in_batch, 2)
        :param batch: which batch each node belongs to of dimension (num_nodes_in_batch, 1)
        :returns: embedding of the input graph
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


class DiffPoolBlock(torch.nn.Module):
    """
    A single block that for the DiffPool that either embeds or pools
    :param in_channels: number of features for input
    :param out_channels: number of features for output
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, out_channels)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, "bn{}".format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj):
        """
        Do a single pass on a batch
        :param x: dense node features of dimension (batch_size, max_nodes, in_channels)
        :param adj: dense adjacency matrix of dimension (batch_size, max_nodes, max_nodes)
        :returns: new dense node features of dimension (batch_size, max_nodes, out_channels)
        """
        x = self.bn(1, F.relu(self.conv1(x, adj)))
        return x
