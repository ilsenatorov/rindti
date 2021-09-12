import math
from argparse import ArgumentParser
from math import ceil

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_batch

from ..base_layer import BaseLayer


class GMTNet(BaseLayer):
    """Graph Multiset Pooling
    https://github.com/JinheonBaek/GMT/"""

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module"""
        parser.add_argument("pool", default="gmt", type=str)
        parser.add_argument("hidden_dim", default=32, type=int, help="Size of output vector")
        parser.add_argument("ratio", default=0.25, type=float, help="Pooling ratio")
        parser.add_argument("num_heads", default=4, type=float, help="Number of attention heads")
        return parser

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=128,
        ratio=0.25,
        max_nodes=600,
        num_heads=4,
        **kwargs,
    ):
        super().__init__()
        num_nodes = max_nodes
        num_nodes = ceil(ratio * num_nodes)
        self.gmpoolg1 = PMA(input_dim, num_heads, num_nodes, mab_conv="GIN")
        self.sab1 = SAB(input_dim, hidden_dim, num_heads)
        num_nodes = ceil(ratio * num_nodes)
        self.gmpoolg2 = PMA(hidden_dim, num_heads, num_nodes, mab_conv=None)
        self.sab2 = SAB(hidden_dim, output_dim, num_heads)
        self.gmpooli = PMA(output_dim, num_heads, 1, mab_conv=None)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, **kwargs):
        """Forward pass of the module"""
        batch_x, mask = to_dense_batch(x, batch)
        extended_attention_mask = mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
        batch_x = self.gmpoolg1(
            batch_x,
            attention_mask=extended_attention_mask,
            graph=(x, edge_index, batch),
        )
        # batch_x = self.ln1(batch_x)
        batch_x = self.sab1(batch_x)
        # batch_x = self.ln2(batch_x)
        batch_x = self.gmpoolg2(batch_x)
        # batch_x = self.ln3(batch_x)
        batch_x = self.sab2(batch_x)
        # batch_x = self.ln4(batch_x)
        batch_x = self.gmpooli(batch_x)
        x = batch_x.squeeze(1)
        x = self.bn(x)
        x = F.normalize(x, dim=1)
        return x


class MAB(LightningModule):
    """Multiset Attention"""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, conv=None):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k, self.fc_v = self.get_fc_kv(dim_K, dim_V, conv)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.softmax_dim = 2

    def forward(self, Q, K, attention_mask=None, graph=None, return_attn=False):
        """Forward pass"""
        Q = self.fc_q(Q)

        # Adj: Exist (graph is not None), or Identity (else)
        if graph is not None:
            (x, edge_index, batch) = graph
            K, V = self.fc_k(x, edge_index), self.fc_v(x, edge_index)
            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)
        else:
            K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
            A = torch.softmax(attention_mask + attention_score, self.softmax_dim)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), self.softmax_dim)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O + F.relu(self.fc_o(O))
        if return_attn:
            return O, A
        else:
            return O

    def get_fc_kv(self, dim_K, dim_V, conv):
        """Get attention values"""
        if conv == "GIN":
            fc_k = GINConv(
                nn.Sequential(
                    nn.Linear(dim_K, dim_K),
                    nn.ReLU(),
                    nn.Linear(dim_K, dim_V),
                    nn.ReLU(),
                    nn.BatchNorm1d(dim_V),
                ),
                train_eps=False,
            )
            fc_v = GINConv(
                nn.Sequential(
                    nn.Linear(dim_K, dim_K),
                    nn.ReLU(),
                    nn.Linear(dim_K, dim_V),
                    nn.ReLU(),
                    nn.BatchNorm1d(dim_V),
                ),
                train_eps=False,
            )
        else:
            fc_k = nn.Linear(dim_K, dim_V)
            fc_v = nn.Linear(dim_K, dim_V)
        return fc_k, fc_v


class SAB(LightningModule):
    """Self-attention module"""

    def __init__(self, dim_in, dim_out, num_heads, mab_conv=None):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, conv=mab_conv)

    def forward(self, X, attention_mask=None, graph=None):
        """Forward pass"""
        return self.mab(X, X, attention_mask, graph)


class ISAB(LightningModule):
    """I think it's not used ATM"""

    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        num_inds,
        mab_conv=None,
    ):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)

        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, conv=mab_conv)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, conv=mab_conv)

    def forward(self, X, attention_mask=None, graph=None):
        """Forward pass"""
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, attention_mask, graph)
        return self.mab1(X, H)


class PMA(LightningModule):
    """Need to figure this out"""

    def __init__(self, dim, num_heads, num_seeds, mab_conv=None):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)

        self.mab = MAB(dim, dim, dim, num_heads, conv=mab_conv)

    def forward(self, X, attention_mask=None, graph=None, return_attn=False):
        """Forward pass"""
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, attention_mask, graph, return_attn)
