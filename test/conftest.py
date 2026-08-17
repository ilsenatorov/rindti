import os
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from lightning.pytorch import seed_everything

from rindti.data import DTIDataModule
from rindti.utils import read_config, write_config

SNAKEMAKE_CONFIG_DIR = "config/snakemake"
DEFAULT_CONFIG = os.path.join(SNAKEMAKE_CONFIG_DIR, "default.yaml")
TEST_CONFIG = os.path.join(SNAKEMAKE_CONFIG_DIR, "test.yaml")


@pytest.fixture(scope="session")
def snakemake_config_base() -> dict:
    from snakemake.utils import update_config

    config = read_config(DEFAULT_CONFIG)
    update_config(config, read_config(TEST_CONFIG))
    return config


@pytest.fixture()
def snakemake_config(snakemake_config_base: dict) -> dict:
    """A fresh copy per test - the workflow tests mutate the config in place."""
    return deepcopy(snakemake_config_base)


def run_snakemake(config: dict, tmpdir_factory) -> Path:
    """Run the workflow on a throwaway copy of the test data.

    Uses the Snakemake >=8 ``SnakemakeApi``; the module-level ``snakemake()``
    function this used to call was removed in Snakemake 8.
    """
    from snakemake.api import SnakemakeApi
    from snakemake.settings.types import (
        ConfigSettings,
        DAGSettings,
        DeploymentMethod,
        DeploymentSettings,
        OutputSettings,
        ResourceSettings,
    )

    tmpdir = Path(str(tmpdir_factory.mktemp("test")))
    config_path = tmpdir / "tmp_config.yaml"
    source_path = tmpdir / "resources"
    shutil.copytree("test/test_data/resources", source_path)
    config["source"] = str(source_path)
    write_config(str(config_path), config)

    with SnakemakeApi(OutputSettings(printshellcmds=True)) as api:
        workflow = api.workflow(
            snakefile=Path("workflow/Snakefile"),
            config_settings=ConfigSettings(configfiles=[config_path]),
            resource_settings=ResourceSettings(cores=4),
            deployment_settings=DeploymentSettings(deployment_method={DeploymentMethod.CONDA}),
        )
        workflow.dag(dag_settings=DAGSettings(forceall=True)).execute_workflow()
    return tmpdir


@pytest.fixture(scope="session")
def snakemake_run(snakemake_config_base: dict, tmpdir_factory) -> Path:
    """Copy test data to a temporary directory and run snakemake on it."""
    return run_snakemake(deepcopy(snakemake_config_base), tmpdir_factory)


def _only_result(root: Path, folder: str) -> str:
    """Path to the single file the workflow produced under ``results/<folder>``."""
    directory = root / "results" / folder
    results = sorted(os.listdir(directory))
    assert len(results) == 1, f"expected one result in {directory}, got {results}"
    return str(directory / results[0])


@pytest.fixture(scope="session")
def split_data(snakemake_run: Path) -> str:
    """Return the split data."""
    return _only_result(snakemake_run, "split_data")


@pytest.fixture(scope="session")
def dti_pickle(snakemake_run: Path) -> str:
    """Return the path to the full pickle file."""
    return _only_result(snakemake_run, "prepare_all")


@pytest.fixture()
def dti_datamodule(dti_pickle: str):
    """DTI datamodule from snakemake test data."""
    return DTIDataModule(dti_pickle, "test", batch_size=4)


@pytest.fixture(autouse=True)
def seed():
    """Set random seed."""
    seed_everything(42)
