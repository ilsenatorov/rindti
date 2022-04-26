from argparse import ArgumentParser
from typing import Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import LongTensor, Tensor, nn
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import (
    AUROC,
    Accuracy,
    ExplainedVariance,
    MatthewsCorrcoef,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    R2Score,
)

from ..data import TwoGraphData
from ..layers import (
    MLP,
    ChebConvNet,
    DiffPoolNet,
    FilmConvNet,
    GatConvNet,
    GINConvNet,
    GMTNet,
    MeanPool,
    TransformerNet,
)

node_embedders = {
    "ginconv": GINConvNet,
    "chebconv": ChebConvNet,
    "gatconv": GatConvNet,
    "filmconv": FilmConvNet,
    "transformer": TransformerNet,
}
poolers = {"gmt": GMTNet, "diffpool": DiffPoolNet, "mean": MeanPool}


class BaseModel(LightningModule):
    """
    Base model, defines a lot of helper functions.
    To subclass shared_step and __init__ need to be defined.
    """

    def __init__(self):
        super().__init__()

    def _set_class_metrics(self):
        metrics = MetricCollection([Accuracy(), AUROC(), MatthewsCorrcoef(num_classes=2)])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _set_reg_metrics(self):
        metrics = MetricCollection([MeanAbsoluteError(), MeanSquaredError(), R2Score(), ExplainedVariance()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _get_label_embed(self, params: dict) -> nn.Embedding:
        return nn.Embedding(params["feat_dim"] + 1, params["hidden_dim"])

    def _get_onehot_embed(self, params: dict) -> nn.LazyLinear:
        return nn.Linear(params["feat_dim"], params["hidden_dim"], bias=False)

    def _get_feat_embed(self, params: dict) -> Union[nn.Embedding, nn.LazyLinear]:
        if params["feat_type"] == "onehot":
            return self._get_onehot_embed(params)
        elif params["feat_type"] == "label":
            return self._get_label_embed(params)
        else:
            raise ValueError("Unknown feature type!")

    def _get_node_embed(self, params: dict, out_dim=None) -> LightningModule:
        if params["edge_type"] == "none":
            params["edge_dim"] = None
        if out_dim:
            return node_embedders[params["node_embed"]](params["hidden_dim"], out_dim, **params)
        return node_embedders[params["node_embed"]](params["hidden_dim"], params["hidden_dim"], **params)

    def _get_pooler(self, params: dict) -> LightningModule:
        return poolers[params["pool"]](params["hidden_dim"], params["hidden_dim"], **params)

    def _get_mlp(self, params: dict) -> MLP:
        return MLP(**params, input_dim=self.embed_dim, out_dim=1)

    def _determine_feat_method(
        self,
        feat_method: str = None,
        drug_hidden_dim: int = None,
        prot_hidden_dim: int = None,
        **kwargs,
    ):
        """Which method to use for concatenating drug and protein representations"""
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
        """Concatenation"""
        return torch.cat((drug_embed, prot_embed), dim=1)

    def _element_l2(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L2 distance"""
        return torch.sqrt(((drug_embed - prot_embed) ** 2) + 1e-6).float()

    def _element_l1(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L1 distance"""
        return (drug_embed - prot_embed).abs()

    def _mult(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Multiplication"""
        return drug_embed * prot_embed

    def training_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during training step"""
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
        """What to do during test step"""
        return self.shared_step(data)

    def log_histograms(self):
        """Logs the histograms of all the available parameters"""
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def training_epoch_end(self, outputs: dict):
        """What to do at the end of a training epoch. Logs everything"""
        self.log_histograms()
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)

    def validation_epoch_end(self, outputs: dict):
        """What to do at the end of a validation epoch. Logs everything"""
        self.log_histograms()
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)

    # def test_epoch_end(self, outputs: dict):
    #     """What to do at the end of a test epoch. Logs everything, saves hyperparameters"""
    #     self.shared_epoch_end(outputs, "test_epoch_", log_hparams=True)

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimiser and/or lr schedulers"""
        optimiser = {"adamw": AdamW, "adam": Adam, "sgd": SGD, "rmsprop": RMSprop}[self.hparams.optimiser]
        params = [{"params": self.parameters()}]
        if hasattr(self, "prot_encoder"):
            params.append({"params": self.prot_encoder.parameters(), "lr": self.hparams.prot_lr})
        if hasattr(self, "drug_encoder"):
            {"params": self.drug_encoder.parameters(), "lr": self.hparams.drug_lr}
        optimiser = optimiser(params=self.parameters(), lr=self.hparams.lr)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimiser,
                factor=self.hparams.reduce_lr_factor,
                patience=self.hparams.reduce_lr_patience,
                verbose=True,
            ),
            "monitor": self.hparams.monitor,
        }
        return [optimiser], [lr_scheduler]
