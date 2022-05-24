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
def full_snakemake_run(tmpdir_factory):
    """Copy test data to a temporary directory and run snakemake on it."""
    tmpdir = tmpdir_factory.mktemp("test_data")
    newdir = shutil.copytree("test/test_data/resources", tmpdir.join("resources"))
    run_snakemake(f"source={newdir}")
    return tmpdir


@pytest.fixture(scope="session")
def pretrain_snakemake_run(tmpdir_factory):
    """Copy test data to a temporary directory and run snakemake on it."""
    tmpdir = tmpdir_factory.mktemp("test_data")
    newdir = shutil.copytree("test/test_data/resources", tmpdir.join("resources"))
    run_snakemake(f"source={newdir}", "only_prots=true")
    return tmpdir


@pytest.fixture(scope="session")
def dti_pickle(full_snakemake_run: str) -> str:
    """Return the path to the full pickle file."""
    folder = "results/prepare_all"
    result = os.listdir(full_snakemake_run.join(folder))[0]
    return full_snakemake_run.join(folder, result)


@pytest.fixture(scope="session")
def pretrain_pickle(pretrain_snakemake_run: str) -> str:
    """Return the path to the pretrain pickle file."""
    folder = "results/pretrain_prot_data"
    result = os.listdir(pretrain_snakemake_run.join(folder))[0]
    return pretrain_snakemake_run.join(folder, result)


@pytest.fixture()
def dti_datamodule(dti_pickle: str):
    """DTI datamodule from snakemake test data."""
    return DTIDataModule(dti_pickle, "test", batch_size=4)


@pytest.fixture()
def pretrain_datamodule(pretrain_pickle: str):
    """Pretrain datamodule from test data proteins."""
    return PreTrainDataModule(pretrain_pickle, "test", batch_size=4)


@pytest.fixture(autouse=True)
def seed():
    """Set random seed."""
    seed_everything(42)
