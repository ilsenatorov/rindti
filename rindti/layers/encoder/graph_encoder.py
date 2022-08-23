import argparse
import json
from typing import Union

import torch
from jsonargparse import lazy_instance
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch

from ..aggr import MeanPool
from ..processor import GraphGPSNet


class GraphEncoder(LightningModule):
    """Encode a graph."""

    def __init__(
        self,
        feat_type: str,
        feat_dim: int,
        edge_type: str,
        edge_dim: int,
        max_nodes: int,
        pos_dim: int = 0,
        hidden_dim: int = 128,
        processor: LightningModule = lazy_instance(GraphGPSNet),
        aggregator: LightningModule = lazy_instance(MeanPool),
    ):
        super().__init__()
        self.max_nodes = max_nodes
        if pos_dim > 0:
            self.pos_embed = nn.Linear(pos_dim, hidden_dim // 2)
        if feat_type == "label":
            self.feat_embed = nn.Embedding(feat_dim + 1, hidden_dim // 2 if pos_dim > 0 else hidden_dim)
        else:
            self.feat_embed = nn.Linear(feat_dim, hidden_dim // 2 if pos_dim > 0 else hidden_dim)
        if edge_type == "label":
            self.edge_embed = nn.Embedding(edge_dim, hidden_dim)
        elif edge_type == "onehot":
            self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_embed = None
        self.processor = processor
        self.aggregator = aggregator

    def forward(self, data: Batch) -> Batch:
        """"""
        data.x = self.feat_embed(data.x)
        data.edge_attr = self.edge_embed(data.edge_attr) if self.edge_embed else data.edge_attr
        if hasattr(self, "pos_embed"):
            data.pos = self.pos_embed(data.pos)
            data.x = torch.cat([data.x, data.pos], dim=-1)
        data = self.processor(data)
        data = self.aggregator(data)
        return data
