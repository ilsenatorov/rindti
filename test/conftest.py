import os
import pickle
import random
import shutil
import subprocess

import pandas as pd
import pytest
from pytorch_lightning.utilities.seed import seed_everything

from rindti.data import DTIDataModule, PreTrainDataModule


def update_pretrain_data(data: dict) -> dict:
    """Update pretrain Series."""
    data["y"] = random.choice([0, 1])
    return data


def run_snakemake(*args):
    """Run snakemake with the given source dir."""
    subprocess.run(
        ["snakemake", "-j", "4", "--forceall", "--use-conda", "--config", *args],
        check=True,
    )


@pytest.fixture(scope="session")
def snakemake_dir(tmpdir_factory):
    """Copy test data to a temporary directory and run snakemake on it."""
    tmpdir = tmpdir_factory.mktemp("test_data")
    newdir = shutil.copytree("test/test_data/resources", tmpdir.join("resources"))
    run_snakemake(f"source={newdir}")
    return tmpdir


@pytest.fixture(scope="session")
def dti_pickle(snakemake_dir: str) -> str:
    """Return the path to the full pickle file."""
    result = os.listdir(snakemake_dir.join("results/prepare_all"))[0]
    return snakemake_dir.join("results/prepare_all", result)


@pytest.fixture(scope="session")
def pretrain_pickle(snakemake_dir: str) -> str:
    """Return the path to the pretrain pickle file."""
    result = os.listdir(snakemake_dir.join("results/prot_data"))[0]
    return snakemake_dir.join("results/prot_data", result)


@pytest.fixture()
def dti_datamodule(dti_pickle: str):
    """DTI datamodule from snakemake test data."""
    return DTIDataModule(dti_pickle, "test", batch_size=4)


@pytest.fixture()
def pretrain_datamodule(pretrain_pickle: str):
    """Pretrain datamodule from test data proteins."""
    data = pd.read_pickle(pretrain_pickle)
    data["data"] = data["data"].apply(update_pretrain_data)
    data.to_pickle(pretrain_pickle)
    return PreTrainDataModule(pretrain_pickle, "test", batch_size=4)


@pytest.fixture(autouse=True)
def seed():
    """Set random seed."""
    seed_everything(42)
