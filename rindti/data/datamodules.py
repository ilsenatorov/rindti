from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data.sampler import Sampler
from torch_geometric.loader import DataLoader

from ..utils import split_random
from .datasets import DTIDataset, PreTrainDataset


class BaseDataModule(LightningDataModule):
    """Base data module, contains all the datasets for train, val and test."""

    def __init__(self, filename: str, batch_size: int = 128, num_workers: int = 16, shuffle: bool = True):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def update_config(self, config: dict) -> None:
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train, **self._dl_kwargs(True))

    def val_dataloader(self):
        return DataLoader(self.val, **self._dl_kwargs(False))

    def test_dataloader(self):
        return DataLoader(self.test, **self._dl_kwargs(False))

    def predict_dataloader(self):
        return DataLoader(self.test, **self._dl_kwargs(False))

    def __repr__(self):
        return "DataModule\n" + "\n".join(
            [repr(getattr(self, x)) for x in ["train", "val", "test"] if hasattr(self, x)]
        )

    def get_labels(self) -> set:
        labels = set()
        for ds in [self.train, self.val, self.test]:
            for i in ds:
                labels.add(i.y)
        return labels


class DTIDataModule(BaseDataModule):
    """Data module for the DTI dataset."""

    def setup(self, stage: str = None):
        """Load the individual datasets"""
        self.train = DTIDataset(self.filename, split="train").shuffle()
        self.val = DTIDataset(self.filename, split="val").shuffle()
        self.test = DTIDataset(self.filename, split="test").shuffle()
        self.config = self.train.config

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
            follow_batch=["prot_x", "drug_x"],
        )

    def __repr__(self):
        return "DTI " + super().__repr__()

    def update_config(self, config: dict) -> None:
        """Update the main config with the config of the dataset."""
        for i in ["prot", "drug"]:
            config["model"][f"{i}"]["data"] = self.config["data"][f"{i}"]
            config["model"][f"{i}"]["data"]["feat_dim"] = self.config["snakemake"][f"{i}_feat_dim"]
            config["model"][f"{i}"]["data"]["edge_dim"] = self.config["snakemake"][f"{i}_edge_dim"]


class PreTrainDataModule(BaseDataModule):
    """DataModule for pretraining on proteins."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        ds = PreTrainDataset(self.filename)
        self.train, self.val, self.test = split_random(ds, [0.7, 0.2, 0.1])
        self.config = ds.config

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
        )
