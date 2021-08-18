from argparse import ArgumentParser
from logging import warning
from pprint import pprint
from typing import Optional, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch.functional import Tensor
from torch.nn.modules.sparse import Embedding
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..layers import (
    MLP,
    ChebConvNet,
    DiffPoolNet,
    FilmConvNet,
    GatConvNet,
    GINConvNet,
    GMTNet,
    MeanPool,
    NoneNet,
    PNAConvNet,
)
from ..layers.base_layer import BaseLayer
from ..utils.data import TwoGraphData

node_embedders = {
    "ginconv": GINConvNet,
    "chebconv": ChebConvNet,
    "gatconv": GatConvNet,
    "filmconv": FilmConvNet,
    "pnaconv": PNAConvNet,
    "none": NoneNet,
}
poolers = {"gmt": GMTNet, "diffpool": DiffPoolNet, "mean": MeanPool}


class BaseModel(LightningModule):
    """
    Base model, defines a lot of helper functions
    """

    def __init__(self):
        super().__init__()

    def _get_feat_embed(self, params: dict) -> Embedding:
        return Embedding(params["feat_dim"] + 1, params["feat_embed_dim"])

    def _get_node_embed(self, params: dict, out_dim=None) -> BaseLayer:
        if out_dim:
            return node_embedders[params["node_embed"]](params["feat_embed_dim"], out_dim, **params)
        return node_embedders[params["node_embed"]](params["feat_embed_dim"], params["hidden_dim"], **params)

    def _get_pooler(self, params: dict) -> BaseLayer:
        return poolers[params["pool"]](params["hidden_dim"], params["hidden_dim"], **params)

    def _get_mlp(self, params: dict) -> MLP:
        return MLP(**params, input_dim=self.embed_dim, out_dim=1)

    def _determine_feat_method(self, feat_method: str, drug_hidden_dim: int, prot_hidden_dim: int, **kwargs):
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
        return self.shared_step(data)

    def validation_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during validation step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        # val_loss has to be logged for early stopping and reduce_lr
        for key, value in ss.items():
            self.log("val_" + key, value)
        return ss

    def test_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during test step"""
        return self.shared_step(data)

    def log_histograms(self):
        """Logs the histograms of all the available parameters"""
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def training_epoch_end(self, outputs):
        """What to do at the end of a training epoch. Logs everything"""
        self.log_histograms()
        entries = outputs[0].keys()
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            self.logger.experiment.add_scalar("train_epoch_" + i, val, self.current_epoch)

    def validation_epoch_end(self, outputs: dict):
        """What to do at the end of a validation epoch. Logs everything, saves hyperparameters"""
        entries = outputs[0].keys()
        metrics = {}
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            self.logger.experiment.add_scalar("val_epoch_" + i, val, self.current_epoch)
            metrics["test_" + i] = val
        self.logger.log_hyperparams(self.hparams, metrics)

    def test_epoch_end(self, outputs: dict):
        """What to do at the end of a test epoch. Logs everything, saves hyperparameters"""
        entries = outputs[0].keys()
        metrics = {}
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            metrics["test_" + i] = val
        self.logger.log_hyperparams(self.hparams, metrics)

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimiser and/or lr schedulers"""
        optimiser = {"adamw": AdamW, "adam": Adam, "sgd": SGD, "rmsprop": RMSprop}[self.hparams.optimiser]
        optimiser = optimiser(
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
            "monitor": "val_loss",
        }
        return [optimiser], [lr_scheduler]

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module"""
        tmp_parser = ArgumentParser(add_help=False)
        tmp_parser.add_argument("--drug_node_embed", type=str, default="chebconv")
        tmp_parser.add_argument("--prot_node_embed", type=str, default="chebconv")
        tmp_parser.add_argument("--prot_pool", type=str, default="gmt")
        tmp_parser.add_argument("--drug_pool", type=str, default="gmt")
        return parser
