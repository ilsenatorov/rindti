from pytorch_lightning import LightningDataModule
from torch.utils.data.sampler import Sampler
from torch_geometric.loader import DataLoader

from ..utils import split_random
from .datasets import DTIDataset, PreTrainDataset


class BaseDataModule(LightningDataModule):
    """Base data module, contains all the datasets for train, val and test."""

    def __init__(
        self, filename: str, exp_name: str, batch_size: int = 128, num_workers: int = 1, shuffle: bool = True
    ):
        super().__init__()
        self.filename = filename
        self.exp_name = exp_name
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


class DTIDataModule(BaseDataModule):
    """Data module for the DTI dataset."""

    def setup(self, stage: str = None, split=None):
        """Load the individual datasets"""
        self.config = None
        if split == "train" or split is None:
            self.train = DTIDataset(self.filename, self.exp_name, split="train").shuffle()
            self.config = self.train.config
        if split == "val" or split is None:
            self.val = DTIDataset(self.filename, self.exp_name, split="val").shuffle()
            if self.config is None:
                self.config = self.val.config
        if split == "test" or split is None:
            self.test = DTIDataset(self.filename, self.exp_name, split="test").shuffle()
            if self.config is None:
                self.config = self.test.config

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
            follow_batch=["prot_x", "drug_x"],
        )

    def update_config(self, config: dict) -> None:
        """Update the main config with the config of the dataset."""
        print(self.config)
        for i in ["prot", "drug"]:
            config["model"][i]["data"] = self.config["snakemake"]["data"][i]


class PreTrainDataModule(BaseDataModule):
    """DataModule for pretraining on prots."""

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

    def update_config(self, config: dict) -> None:
        """Update the main config with the config of the dataset."""
        config["model"]["encoder"]["data"] = self.config["data"]
        config["model"]["num_classes"] = self.config["data"]["num_classes"]
