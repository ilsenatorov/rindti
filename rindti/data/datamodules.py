from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader

from .datasets import DTIDataset


class BaseDataModule(LightningDataModule):
    """Base (template) data module, contains all the datasets for training, validation and testing. Default batch size is 128. Number of sub-processes used for loading data is 1 by default. More workers can speed up data loading on a multi-core CPU. The training data is randomly shuffled before epoch (using a Boolean flag). The base class also contains train_dataloader(), val_dataloader(), test_dataloader(), and predict_dataloader() methods expected by PyTorch Lightning."""

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

    def setup(self, stage: str = None):
        """Load the individual datasets for training, validation and testing, based on their labels. The individual datasets are shuffled independently, to prevent learning artifacts due to data ordering."""
        self.train = DTIDataset(self.filename, self.exp_name, split="train").shuffle()
        self.val = DTIDataset(self.filename, self.exp_name, split="val").shuffle()
        self.test = DTIDataset(self.filename, self.exp_name, split="test").shuffle()
        self.config = self.train.config

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
