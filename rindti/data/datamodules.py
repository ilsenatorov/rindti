import torch_geometric.transforms as T
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from .dti_dataset import DTIDataset
from .protein_dataset import ProtPreTrainDataset
from .samplers import DynamicBatchSampler
from .transforms import MaskType, MaskTypeWeighted, PosNoise


class BaseDataModule(LightningDataModule):
    """Base data module, contains all the datasets for train, val and test."""

    def __init__(
        self,
        filename: str,
        batch_size: int = 128,
        num_workers: int = 1,
        shuffle: bool = True,
        batch_sampling: bool = False,
        max_num_nodes: int = 0,
    ):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.batch_sampling = batch_sampling
        self.max_num_nodes = max_num_nodes

    def _get_dataloader(self, ds: Dataset) -> DataLoader:
        if self.batch_sampling:
            assert self.max_num_nodes > 0
            sampler = DynamicBatchSampler(ds, self.max_num_nodes)
            return DataLoader(ds, batch_sampler=sampler, num_workers=self.num_workers)
        else:
            return DataLoader(ds, **self._dl_kwargs(False))

    def update_config(self, config: dict) -> None:
        raise NotImplementedError

    def train_dataloader(self):
        return self._get_dataloader(self.train)

    def val_dataloader(self):
        return self._get_dataloader(self.val) if self.val else None

    def test_dataloader(self):
        return self._get_dataloader(self.test) if self.test else None


class DTIDataModule(BaseDataModule):
    """Data module for the DTI dataset."""

    def setup(self, stage: str = None):
        """Load the individual datasets"""
        self.train = DTIDataset(self.filename, split="train")
        self.val = DTIDataset(self.filename, split="val")
        self.test = DTIDataset(self.filename, split="test")
        self.config = self.train.config

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
            follow_batch=["prot_x", "drug_x"],
        )


class ProteinDataModule(BaseDataModule):
    """DataModule for pretraining on prots."""

    def __init__(self, *args, sigma: float = 0.75, prob: float = 0.15, weighted_mask: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.prob = prob
        self.weighted_mask = weighted_mask

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        pre_transform = T.Compose(
            [
                T.Center(),
                T.NormalizeRotation(),
                T.RadiusGraph(r=7),
                T.ToUndirected(),
            ]
        )
        transform = T.Compose(
            [
                PosNoise(sigma=self.sigma),
                MaskTypeWeighted(prob=self.prob) if self.weighted_mask else MaskType(prob=self.prob),
            ]
        )
        self.train = ProtPreTrainDataset(self.filename, transform=transform, pre_transform=pre_transform)

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
        )
