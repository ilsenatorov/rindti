import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch

from ..aggr import DiffPoolNet, GMTNet
from ..processor import GINConvNet, GraphGPSNet, TransformerNet

processors = {
    "ginconv": GINConvNet,
    "transformer": TransformerNet,
    "graphgps": GraphGPSNet,
}
aggregators = {
    "gmt": GMTNet,
    "diffpool": DiffPoolNet,
    "none": None,
}


class GraphEncoder(LightningModule):
    """Encode a graph."""

    def __init__(
        self,
        hidden_dim: int,
        feat_type: str,
        feat_dim: int,
        edge_type: str,
        edge_dim: int,
        pos_dim: int,
        max_nodes: int,
        processor: str,
        processor_config: dict,
        aggregator: str,
        aggregator_config: dict,
    ):
        super().__init__()
        if feat_type == "label":
            self.feat_embed = nn.Embedding(feat_dim, hidden_dim // 2)
        else:
            self.feat_embed = nn.Linear(feat_dim, hidden_dim // 2)
        if edge_type == "label":
            self.edge_embed = nn.Embedding(edge_dim, hidden_dim)
        elif edge_type == "onehot":
            self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_embed = None
        self.pos_embed = nn.Linear(pos_dim, hidden_dim // 2)
        self.proc = processors[processor](hidden_dim=hidden_dim, **processor_config)
        self.aggr = aggregators[aggregator](hidden_dim=hidden_dim, max_nodes=max_nodes, **aggregator_config)

    def forward(self, data: Batch) -> Batch:
        """"""
        data.x = self.feat_embed(data.x)
        data.edge_attr = self.edge_embed(data.edge_attr) if self.edge_embed else data.edge_attr
        data.pos = self.pos_embed(data.pos)
        data.x = torch.cat([data.x, data.pos], dim=-1)
        data = self.proc(data)
        data = self.aggr(data)
        return data
