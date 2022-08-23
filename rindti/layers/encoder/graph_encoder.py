import argparse
import json
from typing import Union

import torch
from jsonargparse import lazy_instance
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch

from ..aggr import GMTNet
from ..processor import GraphGPSNet


class GraphEncoder(LightningModule):
    """Encode a graph."""

    def __init__(
        self,
        inputs: str,
        hidden_dim: int = 128,
        processor: LightningModule = lazy_instance(GraphGPSNet),
        aggregator: LightningModule = lazy_instance(GMTNet),
    ):
        super().__init__()
        inputs = json.loads(inputs)
        if inputs["feat_type"] == "label":
            self.feat_embed = nn.Embedding(inputs["feat_dim"], hidden_dim // 2)
        else:
            self.feat_embed = nn.Linear(inputs["feat_dim"], hidden_dim // 2)
        if inputs["edge_type"] == "label":
            self.edge_embed = nn.Embedding(inputs["edge_dim"], hidden_dim)
        elif inputs["edge_type"] == "onehot":
            self.edge_embed = nn.Linear(inputs["edge_dim"], hidden_dim)
        else:
            self.edge_embed = None
        self.pos_embed = nn.Linear(3, hidden_dim // 2)
        self.processor = processor
        self.aggregator = aggregator

    def forward(self, data: Batch) -> Batch:
        """"""
        data.x = self.feat_embed(data.x)
        data.edge_attr = self.edge_embed(data.edge_attr) if self.edge_embed else data.edge_attr
        data.pos = self.pos_embed(data.pos)
        data.x = torch.cat([data.x, data.pos], dim=-1)
        data = self.processor(data)
        data = self.aggregator(data)
        return data
