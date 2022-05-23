import os
import shutil
import subprocess

import pytest
from pytorch_lightning.utilities.seed import seed_everything

from rindti.data import DTIDataModule


def run_snakemake(source: str):
    """Run snakemake with the given source dir."""
    subprocess.run(
        [
            "snakemake",
            "-j",
            "4",
            "--forceall",
            "--use-conda",
            "--config",
            f"source={source}",
        ],
        check=True,
    )


@pytest.fixture(scope="session")
def dti_pickle(tmpdir_factory) -> str:
    tmpdir = tmpdir_factory.mktemp("test_data")
    newdir = shutil.copytree("test/test_data/resources", str(tmpdir.join("resources")))
    run_snakemake(newdir)
    result = os.listdir(tmpdir.join("results/prepare_all"))[0]
    return tmpdir.join("results/prepare_all", result)


@pytest.fixture()
def dti_datamodule(dti_pickle: str):
    """Run snakemake with the given config."""
    return DTIDataModule(dti_pickle, "test", batch_size=4)


@pytest.fixture(autouse=True)
def seed():
    """Set random seed."""
    seed_everything(42)
