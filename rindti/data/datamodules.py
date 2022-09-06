from typing import Callable

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader, DynamicBatchSampler

from ..utils import split_random
from .datasets import DTIDataset, PreTrainDataset


class BaseDataModule(LightningDataModule):
    """Base data module, contains all the datasets for train, val and test."""

    def __init__(
        self,
        filename: str,
        exp_name: str,
        batch_size: int = 128,
        num_workers: int = 1,
        shuffle: bool = True,
        dyn_sampler: bool = False,
    ):
        super().__init__()
        self.filename = filename
        self.exp_name = exp_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dyn_sampler = dyn_sampler
        self.config = None
        self.train, self.val, self.test = None, None, None

    def update_config(self, config: dict) -> None:
        raise NotImplementedError

    def train_dataloader(self):
        """Get the dataloader for the training set"""
        config = self._dl_kwargs(True)
        if self.dyn_sampler:  # if set, use DynamicBatchSampler, batch_size becomes max number of nodes/edges
            config["batch_sampler"] = DynamicBatchSampler(self.train, max_num=self.batch_size, mode="node")
        return DataLoader(self.train, **config)

    def val_dataloader(self):
        """Get the dataloader for the validation set"""
        config = self._dl_kwargs(False)
        if self.dyn_sampler:  # if set, use DynamicBatchSampler, batch_size becomes max number of nodes/edges
            config["batch_sampler"] = DynamicBatchSampler(self.val, max_num=self.batch_size, mode="node")
        return DataLoader(self.val, **config)

    def test_dataloader(self):
        """Get the dataloader for the test set"""
        config = self._dl_kwargs(False)
        if self.dyn_sampler:  # if set, use DynamicBatchSampler, batch_size becomes max number of nodes/edges
            config["batch_sampler"] = DynamicBatchSampler(self.test, max_num=self.batch_size, mode="node")
        return DataLoader(self.test, **config)

    def predict_dataloader(self):
        """Get the dataloader for the prediction set"""
        config = self._dl_kwargs(False)
        if self.dyn_sampler:  # if set, use DynamicBatchSampler, batch_size becomes max number of nodes/edges
            config["batch_sampler"] = DynamicBatchSampler(self.test, max_num=self.batch_size, mode="node")
        return DataLoader(self.test, **config)

    def _dl_kwargs(self, shuffle: bool = False):
        """Get configs for deep learning, used in the dataloader-methods"""
        output = dict(
            num_workers=self.num_workers,
            follow_batch=["prot_x", "drug_x"],
        )
        if not self.dyn_sampler:  # if DynamicBatchSampler should not be used, put batch_size and shuffling in config
            output["batch_size"] = self.batch_size
            output["shuffle"] = self.shuffle and shuffle
        return output


class DTIDataModule(BaseDataModule):
    """Data module for the DTI dataset."""

    def setup(self, stage: str = None, transform: Callable = None, split=None):
        """Load the individual datasets, enables for only loading single splits in case not all splits exist"""
        self.config = None
        if split == "train" or split is None:
            self.train = DTIDataset(self.filename, self.exp_name, split="train", transform=transform).shuffle()
            self.config = self.train.config
        if split == "val" or split is None:
            self.val = DTIDataset(self.filename, self.exp_name, split="val", transform=transform).shuffle()
            if self.config is None:
                self.config = self.val.config
        if split == "test" or split is None:
            self.test = DTIDataset(self.filename, self.exp_name, split="test", transform=transform).shuffle()
            if self.config is None:
                self.config = self.test.config

    def update_config(self, config: dict) -> None:
        """Update the main config with the config of the dataset."""
        print(self.config)
        for i in ["prot", "drug"]:
            if i in self.config["snakemake"]["data"] and i in config["model"]:
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
