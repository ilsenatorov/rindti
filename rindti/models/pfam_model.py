from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.typing import Adj
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef

from ..layers import MLP
from ..utils import remove_arg_prefix
from ..utils.data import TwoGraphData
from .base_model import BaseModel, node_embedders, poolers


class PfamModel(BaseModel):
    """Model for Pfam class comparison problem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._determine_feat_method(kwargs["feat_method"], kwargs["hidden_dim"], kwargs["hidden_dim"])
        self.feat_embed = self._get_feat_embed(kwargs)
        self.node_embed = self._get_node_embed(kwargs)
        self.pool = self._get_pooler(kwargs)
        self.mlp = MLP(kwargs["hidden_dim"], 1, **kwargs)

    def forward(self, a: dict, b: dict) -> Tensor:
        """Forward pass of the model"""
        a["x"] = self.feat_embed(a["x"])
        b["x"] = self.feat_embed(b["x"])
        a["x"] = self.node_embed(**a)
        b["x"] = self.node_embed(**b)
        a_embed = self.pool(**a)
        b_embed = self.pool(**b)
        joint_embedding = self.merge_features(a_embed, b_embed)
        logit = self.mlp(joint_embedding)
        return torch.sigmoid(logit)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        a = remove_arg_prefix("a_", data)
        b = remove_arg_prefix("b_", data)
        output = self.forward(a, b)
        labels = data.label.unsqueeze(1)
        loss = F.binary_cross_entropy(output, labels.float())
        t = (output > 0.5).float()
        acc = accuracy(t, labels)
        try:
            _auroc = auroc(t, labels)
        except Exception:
            _auroc = torch.tensor(np.nan, device=self.device)
        _mc = matthews_corrcoef(output, labels.squeeze(1), num_classes=2)
        return {
            "loss": loss,
            "acc": acc,
            "auroc": _auroc,
            "matthews": _mc,
        }
