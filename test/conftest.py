import os
import shutil

import pytest
from pytorch_lightning.utilities.seed import seed_everything
from snakemake import snakemake

from rindti.data import DTIDataModule, PreTrainDataModule
from rindti.utils import read_config, write_config

SNAKEMAKE_CONFIG_DIR = "config/snakemake"
DEFAULT_CONFIG = os.path.join(SNAKEMAKE_CONFIG_DIR, "default.yaml")
TEST_CONFIG = os.path.join(SNAKEMAKE_CONFIG_DIR, "test_dti.yaml")
GLYLEC_CONFIG = os.path.join(SNAKEMAKE_CONFIG_DIR, "test_glylec.yaml")


@pytest.fixture(scope="session")
def dti_snakemake_config():
    default_test_config = read_config(DEFAULT_CONFIG)
    test_config = read_config(TEST_CONFIG)
    default_test_config.update(test_config)
    return default_test_config


@pytest.fixture(scope="session")
def glylec_snakemake_config():
    default_glylec_config = read_config(DEFAULT_CONFIG)
    glylec_config = read_config(GLYLEC_CONFIG)
    default_glylec_config.update(glylec_config)
    return default_glylec_config


def run_snakemake(config: dict, tmpdir_factory: str, data_dir: str = "test/test_data/resources"):
    """Run snakemake with the given config."""
    tmpdir = tmpdir_factory.mktemp("test")
    if "glylec" in data_dir:
        config_path = str(tmpdir.join("glylec/tmp_config.yaml"))
        source_path = str(tmpdir.join("glylec/resources"))
    else:
        config_path = str(tmpdir.join("tmp_config.yaml"))
        source_path = str(tmpdir.join("resources"))
    print(config_path)
    print(source_path)
    shutil.copytree(data_dir, source_path)
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
def split_data(snakemake_run):
    """Return the split data."""
    if "glylec" in snakemake_run:
        folder = "glylec/results/split_data"
    else:
        folder = "results/split_data"
    result = os.listdir(snakemake_run.join(folder))[0]
    return snakemake_run.join(folder, result)


@pytest.fixture(scope="session")
def dti_snakemake_run(dti_snakemake_config: dict, tmpdir_factory):
    """Copy test data to a temporary directory and run snakemake on it."""
    return run_snakemake(dti_snakemake_config, tmpdir_factory)


@pytest.fixture(scope="session")
def glylec_snakemake_run(glylec_snakemake_config: dict, tmpdir_factory):
    """Copy test data to a temporary directory and run snakemake on it."""
    return run_snakemake(glylec_snakemake_config, tmpdir_factory, "test/test_data/glylec/resources")


@pytest.fixture(scope="session")
def dti_pickle(dti_snakemake_run: str) -> str:
    """Return the path to the full pickle file."""
    folder = "results/prepare_all"
    result = os.listdir(dti_snakemake_run.join(folder))[0]
    return dti_snakemake_run.join(folder, result)


@pytest.fixture(scope="session")
def glylec_pickle(glylec_snakemake_run: str) -> str:
    """Return the path to the full pickle file."""
    folder = "glylec/results/prepare_all"
    result = os.listdir(glylec_snakemake_run.join(folder))[0]
    return glylec_snakemake_run.join(folder, result)


@pytest.fixture(scope="session")
def pretrain_pickle(dti_snakemake_run: str) -> str:
    """Return the path to the pretrain pickle file."""
    folder = "results/pretrain_prot_data"
    result = os.listdir(dti_snakemake_run.join(folder))[0]
    return dti_snakemake_run.join(folder, result)


@pytest.fixture()
def dti_datamodule(dti_pickle: str):
    """DTI datamodule from snakemake test data."""
    return DTIDataModule(dti_pickle, "test", batch_size=4)


@pytest.fixture()
def glylec_datamodule(glylec_pickle: str):
    """DTI datamodule from snakemake test data."""
    return DTIDataModule(glylec_pickle, "test", batch_size=4)


@pytest.fixture()
def pretrain_datamodule(pretrain_pickle: str):
    """Pretrain datamodule from test data proteins."""
    return PreTrainDataModule(pretrain_pickle, "test", batch_size=4)


@pytest.fixture(autouse=True)
def seed():
    """Set random seed."""
    seed_everything(42)
