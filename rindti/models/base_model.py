from typing import Tuple

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
)

from ..data import TwoGraphData


class BaseModel(LightningModule):
    """Base model, defines a lot of helper functions."""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = kwargs["datamodule"]["batch_size"]
        return kwargs["model"]

    def _set_class_metrics(self, num_classes: int = 2):
        metrics = MetricCollection(
            [
                Accuracy(
                    num_classes=None if num_classes == 2 else num_classes,
                    task="binary",
                ),
                AUROC(
                    num_classes=None if num_classes == 2 else num_classes, task="binary"
                ),
                MatthewsCorrCoef(num_classes=num_classes, task="binary"),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _set_reg_metrics(self):
        metrics = MetricCollection(
            [MeanAbsoluteError(), MeanSquaredError(), ExplainedVariance()]
        )
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
        self.log("train_loss", ss["loss"], batch_size=self.batch_size)
        return ss

    def validation_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during validation step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        self.val_metrics.update(ss["preds"], ss["labels"])
        self.log("val_loss", ss["loss"], batch_size=self.batch_size)
        return ss

    def test_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during test step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        self.test_metrics.update(ss["preds"], ss["labels"])
        self.log("test_loss", ss["loss"], batch_size=self.batch_size)
        return ss

    def log_histograms(self):
        """Logs the histograms of all the available parameters."""
        if self.logger:
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def log_all(self, metrics: dict, hparams: bool = False):
        """Log all metrics."""
        if self.logger:
            for k, v in metrics.items():
                self.logger.experiment.add_scalar(k, v, self.current_epoch)
            if hparams:
                self.logger.log_hyperparams(
                    self.hparams, {k.split("_")[-1]: v for k, v in metrics.items()}
                )

    ## FIXME the logic of epoch end got changed, need to update the code

    # def on_train_epoch_end(self, outputs: dict):
    #     """What to do at the end of a training epoch. Logs everything."""
    #     self.log_histograms()
    #     metrics = self.train_metrics.compute()
    #     self.train_metrics.reset()
    #     self.log_all(metrics)

    # def on_validation_epoch_end(self, outputs: dict):
    #     """What to do at the end of a validation epoch. Logs everything."""
    #     metrics = self.val_metrics.compute()
    #     self.val_metrics.reset()
    #     self.log_all(metrics, hparams=True)

    # def on_test_epoch_end(self, outputs: dict):
    #     """What to do at the end of a test epoch. Logs everything."""
    #     metrics = self.test_metrics.compute()
    #     self.test_metrics.reset()
    #     self.log_all(metrics)

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimizer and/or lr schedulers"""
        opt_params = self.hparams.model["optimizer"]
        optimizer = {"adamw": AdamW, "adam": Adam, "sgd": SGD, "rmsprop": RMSprop}[
            opt_params["module"]
        ]
        params = [{"params": self.parameters()}]
        if hasattr(self, "prot_encoder"):
            params.append(
                {"params": self.prot_encoder.parameters(), "lr": opt_params["prot_lr"]}
            )
        if hasattr(self, "drug_encoder"):
            {"params": self.drug_encoder.parameters(), "lr": opt_params["drug_lr"]}
        optimizer = optimizer(params=self.parameters(), lr=opt_params["lr"])
        lr_scheduler = {
            "monitor": self.hparams["model"]["monitor"],
            "scheduler": ReduceLROnPlateau(
                optimizer,
                verbose=True,
                factor=opt_params["reduce_lr"]["factor"],
                patience=opt_params["reduce_lr"]["patience"],
            ),
        }
        return [optimizer], [lr_scheduler]
