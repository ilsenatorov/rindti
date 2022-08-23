import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import (
    AUROC,
    Accuracy,
    ExplainedVariance,
    MatthewsCorrCoef,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
)

from ..data import TwoGraphData
from ..utils.optim import LinearWarmupCosineAnnealingLR


class BaseModel(LightningModule):
    """Base model, defines a lot of helper functions."""

    def __init__(self, **kwargs):
        super().__init__()

    def _set_class_metrics(self, num_classes: int = 2):
        metrics = MetricCollection(
            [
                Accuracy(num_classes=None if num_classes == 2 else num_classes),
                AUROC(num_classes=None if num_classes == 2 else num_classes),
                MatthewsCorrCoef(num_classes=num_classes),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _set_reg_metrics(self):
        metrics = MetricCollection([MeanAbsoluteError(), MeanSquaredError(), ExplainedVariance()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _determine_feat_method(
        self,
        feat_method: str,
        drug_hidden_dim: int = None,
        prot_hidden_dim: int = None,
        **kwargs,
    ):
        """Which method to use for concatenating drug and protein representations."""
        if feat_method == "concat":
            self.merge_features = self._concat
            self.embed_dim = drug_hidden_dim + prot_hidden_dim
        elif feat_method == "element_l2":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._element_l2
            self.embed_dim = drug_hidden_dim
        else:
            raise ValueError("unsupported feature method")

    def _concat(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Concatenation."""
        return torch.cat((drug_embed, prot_embed), dim=1)

    def _element_l2(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L2 distance."""
        return torch.sqrt(((drug_embed - prot_embed) ** 2) + 1e-6).float()

    def training_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during training step."""
        ss = self.shared_step(data)
        for k, v in ss.items():
            self.log(f"train_{k}", v)
        return ss

    def validation_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during validation step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        for k, v in ss.items():
            self.log(f"val_{k}", v)
        return ss

    def test_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during test step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        for k, v in ss.items():
            self.log(f"test_{k}", v)
        return ss
