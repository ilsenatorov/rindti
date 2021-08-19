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

    def forward(
        self,
        a_x: Tensor,
        b_x: Tensor,
        a_edge_index: Adj,
        b_edge_index: Adj,
        a_batch: Tensor,
        b_batch: Tensor,
        *args,
    ) -> Tensor:

        a_x = self.feat_embed(a_x)
        b_x = self.feat_embed(b_x)
        a_x = self.node_embed(a_x, a_edge_index, a_batch)
        b_x = self.node_embed(b_x, b_edge_index, b_batch)
        a_embed = self.pool(a_x, a_edge_index, a_batch)
        b_embed = self.pool(b_x, b_edge_index, b_batch)
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
        output = self.forward(
            data.a_x,
            data.b_x,
            data.a_edge_index,
            data.b_edge_index,
            data.a_x_batch,
            data.b_x_batch,
        )
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

    def training_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during training step. Also logs the values for various callbacks.

        Args:
            data (TwoGraphData): Input data
            data_idx (int): Number of the batch

        Returns:
            dict: dictionary with all losses and accuracies
        """
        ss = self.shared_step(data)
        # val_loss has to be logged for early stopping and reduce_lr
        for key, value in ss.items():
            self.log("train_" + key, value)
        return ss

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimiser and/or lr schedulers

        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
            tuple of optimiser list and lr_scheduler list
        """
        optimiser = AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimiser,
                factor=self.hparams.reduce_lr_factor,
                patience=self.hparams.reduce_lr_patience,
                verbose=True,
            ),
            "monitor": "train_loss",
        }
        return [optimiser], [lr_scheduler]
