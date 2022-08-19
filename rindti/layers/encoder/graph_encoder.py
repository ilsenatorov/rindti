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
        inputs: dict,
        processor: str,
        processor_config: dict,
        aggregator: str,
        aggregator_config: dict,
    ):
        super().__init__()
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
        self.proc = processors[processor](hidden_dim=hidden_dim, **processor_config)
        self.aggr = aggregators[aggregator](hidden_dim=hidden_dim, max_nodes=inputs["max_nodes"], **aggregator_config)

    def forward(self, data: Batch) -> Batch:
        """"""
        data.x = self.feat_embed(data.x)
        data.edge_attr = self.edge_embed(data.edge_attr) if self.edge_embed else data.edge_attr
        data.pos = self.pos_embed(data.pos)
        data.x = torch.cat([data.x, data.pos], dim=-1)
        data = self.proc(data)
        data = self.aggr(data)
        return data
