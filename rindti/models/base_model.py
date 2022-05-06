from typing import Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import (
    AUROC,
    Accuracy,
    ExplainedVariance,
    MatthewsCorrCoef,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    R2Score,
)

from ..data import TwoGraphData
from ..layers import MLP


class BaseModel(LightningModule):
    """Base model, defines a lot of helper functions."""

    def __init__(self):
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
        metrics = MetricCollection([MeanAbsoluteError(), MeanSquaredError(), R2Score(), ExplainedVariance()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _get_mlp(self, **params) -> MLP:
        return MLP(input_dim=self.embed_dim, out_dim=1, **params)

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
        elif feat_method == "element_l1":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._element_l1
            self.embed_dim = drug_hidden_dim
        elif feat_method == "mult":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._mult
            self.embed_dim = drug_hidden_dim
        else:
            raise ValueError("unsupported feature method")

    def _concat(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Concatenation."""
        return torch.cat((drug_embed, prot_embed), dim=1)

    def _element_l2(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L2 distance."""
        return torch.sqrt(((drug_embed - prot_embed) ** 2) + 1e-6).float()

    def _element_l1(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L1 distance."""
        return (drug_embed - prot_embed).abs()

    def _mult(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Multiplication."""
        return drug_embed * prot_embed

    def training_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during training step."""
        ss = self.shared_step(data)
        self.train_metrics.update(ss["preds"], ss["labels"])
        self.log("train_loss", ss["loss"])
        return ss

    def validation_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during validation step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        self.val_metrics.update(ss["preds"], ss["labels"])
        self.log("val_loss", ss["loss"])
        return ss

    def test_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during test step."""
        return self.shared_step(data)

    def log_histograms(self):
        """Logs the histograms of all the available parameters."""
        if self.logger:
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def training_epoch_end(self, outputs: dict):
        """What to do at the end of a training epoch. Logs everything."""
        self.log_histograms()
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)

    def validation_epoch_end(self, outputs: dict):
        """What to do at the end of a validation epoch. Logs everything."""
        self.log_histograms()
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)

    # def test_epoch_end(self, outputs: dict):
    #     """What to do at the end of a test epoch. Logs everything, saves hyperparameters"""
    #     self.shared_epoch_end(outputs, "test_epoch_", log_hparams=True)

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimizer and/or lr schedulers"""
        opt_params = self.hparams.optimizer
        optimizer = {"adamw": AdamW, "adam": Adam, "sgd": SGD, "rmsprop": RMSprop}[opt_params["module"]]
        params = [{"params": self.parameters()}]
        if hasattr(self, "prot_encoder"):
            params.append({"params": self.prot_encoder.parameters(), "lr": opt_params["prot_lr"]})
        if hasattr(self, "drug_encoder"):
            {"params": self.drug_encoder.parameters(), "lr": opt_params["drug_lr"]}
        optimizer = optimizer(params=self.parameters(), lr=opt_params["lr"])
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                verbose=True,
                **opt_params["reduce_lr"],
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]
