import os
import shutil
import subprocess

import pytest
from pytorch_lightning.utilities.seed import seed_everything

from rindti.data import DTIDataModule, PreTrainDataModule


def run_snakemake(*args):
    """Run snakemake with the given source dir."""
    subprocess.run(
        ["snakemake", "-j", "4", "--forceall", "--use-conda", "--config", *args],
        check=True,
    )


@pytest.fixture(scope="session")
def snakemake_dir(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("test_data")
    newdir = shutil.copytree("test/test_data/resources", str(tmpdir.join("resources")))
    run_snakemake(f"source={newdir}")
    return tmpdir


@pytest.fixture(scope="session")
def dti_pickle(snakemake_dir) -> str:
    result = os.listdir(snakemake_dir.join("results/prepare_all"))[0]
    return snakemake_dir.join("results/prepare_all", result)


@pytest.fixture(scope="session")
def pretrain_pickle(snakemake_dir) -> str:
    result = os.listdir(snakemake_dir.join("results/prot_data"))[0]
    return snakemake_dir.join("results/prot_data", result)


@pytest.fixture()
def dti_datamodule(dti_pickle: str):
    """Run snakemake with the given config."""
    return DTIDataModule(dti_pickle, "test", batch_size=4)


@pytest.fixture()
def pretrain_datamodule(pretrain_pickle: str):
    """Run snakemake with the given config."""
    return PreTrainDataModule(pretrain_pickle, "test", batch_size=4)


@pytest.fixture(autouse=True)
def seed():
    """Set random seed."""
    seed_everything(42)
