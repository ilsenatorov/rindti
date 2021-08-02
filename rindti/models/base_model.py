from argparse import ArgumentParser
from logging import warning
from typing import Optional, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseModel(LightningModule):
    @staticmethod
    def add_arguments(parser):
        """
        Add the arguments for the training
        """
        tmp_parser = ArgumentParser(add_help=False)
        tmp_parser.add_argument("--drug_node_embed", type=str, default="chebconv")
        tmp_parser.add_argument("--prot_node_embed", type=str, default="chebconv")
        tmp_parser.add_argument("--prot_pool", type=str, default="gmt")
        tmp_parser.add_argument("--drug_pool", type=str, default="gmt")
        return parser

    def __init__(self):
        """Base model, only requires the dataset to function

        Args:
            dataset (Optional[Dataset], optional): Member of the Dataset class. Defaults to Dataset().
        """
        super().__init__()

    def _determine_feat_method(self, kwargs):
        """
        Choose appropriate way to combine prot and drug embedding vectors
        :param kwargs: Kwargs from __init__
        """
        feat_method = kwargs.pop("feat_method")
        drug_dim = kwargs["drug_hidden_dim"]
        prot_dim = kwargs["prot_hidden_dim"]
        if feat_method == "concatenate":
            self.merge_features = self._concatenate
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
            raise Exception("unsupported feature method")

    def _concatenate(self, drug_embed, prot_embed):
        return torch.cat((drug_embed, prot_embed), dim=1)

    def _element_l2(self, drug_embedding, prot_embedding):
        return torch.sqrt(((drug_embedding - prot_embedding) ** 2) + 1e-6).float()

    def _element_l1(self, drug_embedding, prot_embedding):
        return (drug_embedding - prot_embedding).abs()

    def _mult(self, drug_embedding, prot_embedding):
        return drug_embedding * prot_embedding

    def training_step(self, data, batch_idx):
        """
        What to do during training step
        :param data: ProteinData batch object
        :param batch_idx: index of the batch

        """
        return self.shared_step(data)

    def validation_step(self, data, data_idx):
        """
        What to do during validation step
        :param data: ProteinData batch object
        :param batch_idx: index of the batch
        """
        ss = self.shared_step(data)
        # val_loss has to be logged for early stopping and reduce_lr
        for key, value in ss.items():
            self.log("val_" + key, value)
        return ss

    def test_step(self, data, data_idx):
        """
        What to do during test step
        :param data: ProteinData batch object
        :param batch_idx: index of the batch
        """
        return self.shared_step(data)

    def log_histograms(self):
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def training_epoch_end(self, outputs):
        """
        What to log and save on train epoch end
        :param outputs: return of training_step in a tensor (dicts)
        """
        self.log_histograms()
        entries = outputs[0].keys()
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            self.logger.experiment.add_scalar("train_epoch_" + i, val, self.current_epoch)

    def validation_epoch_end(self, outputs):
        """
        What to log and save on validation epoch end
        :param outputs: return of validation_step in a tensor (dicts)
        """
        entries = outputs[0].keys()
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            self.logger.experiment.add_scalar("val_epoch_" + i, val, self.current_epoch)

    def test_epoch_end(self, outputs):
        """
        What to log and save on train epoch end
        :param outputs: return of training_step in a tensor (dicts)
        """
        entries = outputs[0].keys()
        metrics = {}
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            metrics["test_" + i] = val
        self.logger.log_hyperparams(self.hparams, metrics)

    def configure_optimizers(self):
        """
        Configure the optimizer/s.
        Relies on initially saved hparams to contain learning rates, weight decays etc
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
