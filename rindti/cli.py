"""Command-line entry point for training RINDTI models."""

import os
import random
import re

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from .data import DTIDataModule
from .models import ClassificationModel, RegressionModel
from .utils import get_git_hash, read_config

models = {
    "class": ClassificationModel,
    "reg": RegressionModel,
}

VERSION_RE = re.compile(r"^version_(\d+)$")


def next_version(folder: str) -> int:
    """Smallest unused ``version_<n>`` index in ``folder``.

    Compares the trailing integers numerically, so ``version_10`` sorts after
    ``version_9`` rather than before it.
    """
    versions = [
        int(match.group(1))
        for entry in os.listdir(folder)
        if (match := VERSION_RE.match(entry)) and os.path.isdir(os.path.join(folder, entry))
    ]
    return max(versions) + 1 if versions else 0


def train(**kwargs) -> None:
    """Train the model for ``runs`` seeds, logging each run under a shared version dir."""
    seed_everything(kwargs["seed"])
    seeds = random.sample(range(1, 100), kwargs["runs"])

    dataset = os.path.splitext(os.path.basename(kwargs["datamodule"]["filename"]))[0]
    folder = os.path.join("tb_logs", f"dti_{kwargs['datamodule']['exp_name']}", dataset)
    os.makedirs(folder, exist_ok=True)
    version = next_version(folder)

    for i, seed in enumerate(seeds):
        print(f"Run {i + 1} of {kwargs['runs']} with seed {seed}")
        kwargs["seed"] = seed
        single_run(folder, version, **kwargs)


def single_run(folder: str, version: int, **kwargs) -> None:
    """Does a single run."""
    seed_everything(kwargs["seed"])
    datamodule = DTIDataModule(**kwargs["datamodule"])
    datamodule.setup()
    datamodule.update_config(kwargs)

    logger = TensorBoardLogger(
        save_dir=folder,
        name=f"version_{version}",
        version=kwargs["seed"],
        default_hp_metric=False,
    )
    callbacks = [
        ModelCheckpoint(monitor=kwargs["model"]["monitor"], save_top_k=3, mode="min"),
        EarlyStopping(monitor=kwargs["model"]["monitor"], mode="min", **kwargs["early_stop"]),
        RichModelSummary(),
        RichProgressBar(),
    ]
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=25,
        enable_model_summary=False,
        **kwargs["trainer"],
    )
    model = models[kwargs["model"]["module"]](**kwargs)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


def train_cli() -> None:
    """``rindti-train <config.yaml>``."""
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="rindti-train")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = read_config(args.config)
    config["git_hash"] = get_git_hash()  # to know the version of the code
    train(**config)


if __name__ == "__main__":
    train_cli()
