from argparse import ArgumentParser
from logging import warning

from torch.functional import Tensor
from rindti.utils.data import TwoGraphData
from typing import Optional, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseModel(LightningModule):
    """
    Base model, only requires the dataset to function
    """

    def __init__(self):
        super().__init__()

    def _determine_feat_method(self, feat_method: str, drug_dim: int, prot_dim: int):
        """[summary]

        Args:
            feat_method (str): how to concatenate drugs
            drug_dim (int): [description]
            prot_dim (int): [description]

        Raises:
            ValueError: [description]
        """
        if feat_method == "concat":
            self.merge_features = self._concat
            self.embed_dim = drug_dim + prot_dim
        elif feat_method == "element_l2":
            assert drug_dim == prot_dim
            self.merge_features = self._element_l2
            self.embed_dim = drug_dim
        elif feat_method == "element_l1":
            assert drug_dim == prot_dim
            self.merge_features = self._element_l1
            self.embed_dim = drug_dim
        elif feat_method == "mult":
            assert drug_dim == prot_dim
            self.merge_features = self._mult
            self.embed_dim = drug_dim
        else:
            raise ValueError("unsupported feature method")

    def _concat(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Concatenation

        Args:
            drug_embed (Tensor): drug embedding
            prot_embed (Tensor): prot embedding

        Returns:
            Tensor: Concatenated tensor (dim = drug_embedding dim + prot_embedding dim)
        """
        return torch.cat((drug_embed, prot_embed), dim=1)

    def _element_l2(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L2 distance

        Args:
            drug_embed (Tensor): drug embedding
            prot_embed (Tensor): prot embedding

        Returns:
            Tensor: combined tensor
        """
        return torch.sqrt(((drug_embed - prot_embed) ** 2) + 1e-6).float()

    def _element_l1(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L1 distance

        Args:
            drug_embed (Tensor): drug embedding
            prot_embed (Tensor): prot embedding

        Returns:
            Tensor: combined tensor
        """
        return (drug_embed - prot_embed).abs()

    def _mult(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Multiplication

        Args:
            drug_embed (Tensor): drug embedding
            prot_embed (Tensor): prot embedding

        Returns:
            Tensor: combined tensor
        """
        return drug_embed * prot_embed

    def training_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during training step

        Args:
            data (TwoGraphData): Input data
            data_idx (int): Number of the batch

        Returns:
            dict: dictionary with all losses and accuracies
        """
        return self.shared_step(data)

    def validation_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during validation step. Also logs the values for various callbacks.

        Args:
            data (TwoGraphData): Input data
            data_idx (int): Number of the batch

        Returns:
            dict: dictionary with all losses and accuracies
        """
        ss = self.shared_step(data)
        # val_loss has to be logged for early stopping and reduce_lr
        for key, value in ss.items():
            self.log("val_" + key, value)
        return ss

    def test_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during test step

        Args:
            data (TwoGraphData): Input data
            data_idx (int): Number of the batch

        Returns:
            dict: dictionary with all losses and accuracies
        """
        return self.shared_step(data)

    def log_histograms(self):
        """Logs the histograms of all the available parameters"""
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def training_epoch_end(self, outputs):
        """What to do at the end of a training epoch. Logs everything

        Args:
            outputs (dict): dict from shared_step - contains losses and accuracies
        """
        self.log_histograms()
        entries = outputs[0].keys()
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            self.logger.experiment.add_scalar("train_epoch_" + i, val, self.current_epoch)

    def validation_epoch_end(self, outputs):
        """What to do at the end of a validation epoch. Logs everything

        Args:
            outputs (dict): dict from shared_step - contains losses and accuracies
        """
        entries = outputs[0].keys()
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            self.logger.experiment.add_scalar("val_epoch_" + i, val, self.current_epoch)

    def test_epoch_end(self, outputs: dict):
        """What to do at the end of a test epoch. Logs everything, saves hyperparameters

        Args:
            outputs (dict): dict from shared_step - contains losses and accuracies
        """
        entries = outputs[0].keys()
        metrics = {}
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            metrics["test_" + i] = val
        self.logger.log_hyperparams(self.hparams, metrics)

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimiser and/or lr schedulers

        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
            tuple of optimiser list and lr_scheduler list
        """
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
        """Generate arguments for this module

        Args:
            parser (ArgumentParser): Parent parser

        Returns:
            ArgumentParser: Updated parser
        """
        tmp_parser = ArgumentParser(add_help=False)
        tmp_parser.add_argument("--drug_node_embed", type=str, default="chebconv")
        tmp_parser.add_argument("--prot_node_embed", type=str, default="chebconv")
        tmp_parser.add_argument("--prot_pool", type=str, default="gmt")
        tmp_parser.add_argument("--drug_pool", type=str, default="gmt")
        return parser
