import os
import shutil

import pytest
from pytorch_lightning.utilities.seed import seed_everything
from snakemake import snakemake

from rindti.data import DTIDataModule, ProteinDataModule
from rindti.utils import read_config, write_config

SNAKEMAKE_CONFIG_DIR = "config/snakemake"
DEFAULT_CONFIG = os.path.join(SNAKEMAKE_CONFIG_DIR, "default.yaml")
TEST_CONFIG = os.path.join(SNAKEMAKE_CONFIG_DIR, "test.yaml")


@pytest.fixture(scope="session")
def snakemake_config():
    default_config = read_config(DEFAULT_CONFIG)
    test_config = read_config(TEST_CONFIG)
    default_config.update(test_config)
    return default_config


def run_snakemake(config: dict, tmpdir_factory: str):
    """Run snakemake with the given config."""
    tmpdir = tmpdir_factory.mktemp("test")
    config_path = str(tmpdir.join("tmp_config.yaml"))
    source_path = str(tmpdir.join("resources"))
    shutil.copytree("test/test_data/resources", source_path)
    config["source"] = source_path
    write_config(config_path, config)
    assert snakemake(
        "workflow/Snakefile",
        configfiles=[config_path],
        use_conda=True,
        cores=4,
        forceall=True,
    )
    return tmpdir


@pytest.fixture(scope="session")
def snakemake_run(snakemake_config: dict, tmpdir_factory):
    """Copy test data to a temporary directory and run snakemake on it."""
    return run_snakemake(snakemake_config, tmpdir_factory)


@pytest.fixture(scope="session")
def split_data(snakemake_run):
    """Return the split data."""
    folder = "results/split_data"
    result = os.listdir(snakemake_run.join(folder))[0]
    return snakemake_run.join(folder, result)


@pytest.fixture(scope="session")
def dti_pickle(snakemake_run: str) -> str:
    """Return the path to the full pickle file."""
    folder = "results/prepare_all"
    result = os.listdir(snakemake_run.join(folder))[0]
    return snakemake_run.join(folder, result)


@pytest.fixture(scope="session")
def pretrain_pickle(snakemake_run: str) -> str:
    """Return the path to the pretrain pickle file."""
    folder = "results/pretrain_prot_data"
    result = os.listdir(snakemake_run.join(folder))[0]
    return snakemake_run.join(folder, result)


@pytest.fixture()
def dti_datamodule(dti_pickle: str):
    """DTI datamodule from snakemake test data."""
    return DTIDataModule(dti_pickle, "test", batch_size=4)


@pytest.fixture()
def pretrain_datamodule(pretrain_pickle: str):
    """Pretrain datamodule from test data proteins."""
    return ProteinDataModule(pretrain_pickle, "test", batch_size=4)


@pytest.fixture(autouse=True)
def seed():
    """Set random seed."""
    seed_everything(42)
