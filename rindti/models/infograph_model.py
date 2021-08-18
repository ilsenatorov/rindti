from typing import Tuple

import torch
import torch.nn.functional as F
from torch.functional import Tensor

from ..layers import MutualInformation
from ..layers.graphconv import GINConvNet
from ..utils.data import TwoGraphData
from .noisy_nodes_model import NoisyNodesModel


class InfoGraph(NoisyNodesModel):
    """Maximise mutual information between node and graph representations
    https://arxiv.org/pdf/1808.06670.pdf"""

    def __init__(self, **kwargs):
        super().__init__()
        self.feat_embed = self._get_feat_embed(kwargs)
        self.node_embed = self._get_node_embed(kwargs)
        self.pool = self._get_pooler(kwargs)
        self.mi = MutualInformation(kwargs["hidden_dim"], kwargs["hidden_dim"])
        self.node_pred = GINConvNet(kwargs["hidden_dim"], kwargs["feat_dim"], kwargs["hidden_dim"])

    def forward(self, data: TwoGraphData) -> Tuple[Tensor, Tensor]:
        """Forward pass of the module"""
        data["x"] = self.feat_embed(data["x"])
        data["x"] = self.node_embed(**data)
        graph_embed = self.pool(**data)
        mi = self.mi(graph_embed, data["x"], data["batch"])
        node_pred = self.node_pred(**data)
        return mi, node_pred

    def shared_step(self, data: TwoGraphData) -> dict:
        """Shared step"""
        orig_x = data["x"].clone()
        cor_x, cor_idx = self.corrupt_features(data["x"], self.hparams.frac)
        data["x"] = cor_x
        mi, node_pred = self.forward(data)
        node_loss = F.cross_entropy(node_pred[cor_idx], orig_x[cor_idx])
        loss = -mi + node_pred * self.hparams.alpha
        return {"loss": loss, "mi": mi, "node_loss": node_loss}
