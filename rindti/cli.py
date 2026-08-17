"""Command-line entry point for training RINDTI models."""

import os
import random
import re

import yaml
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


def apply_override(config: dict, assignment: str) -> None:
    """Apply one ``dotted.key=value`` override to a nested config, in place.

    The value is parsed as YAML, so ``0.001``, ``true``, ``null`` and ``[a, b]``
    all arrive with the right type. Only existing keys can be set, so a typo is an
    error rather than a silently ignored setting.
    """
    path, _, raw = assignment.partition("=")
    if not _:
        raise ValueError(f"Override {assignment!r} is not of the form key.path=value")

    keys = path.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"No such config section {'.'.join(keys[:-1])!r} (at {key!r})")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"No such config key {path!r}")
    node[keys[-1]] = yaml.safe_load(raw)


def train_cli() -> None:
    """``rindti-train <config.yaml> [--set key.path=value ...]``."""
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="rindti-train",
        epilog=(
            "Example: rindti-train config/dti/base.yaml "
            "--set datamodule.filename=datasets/davis/results/prepare_all/x.pkl "
            "--set datamodule.exp_name=davis_random"
        ),
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="Override a config value; repeatable.",
    )
    args = parser.parse_args()

    config = read_config(args.config)
    for assignment in args.overrides:
        apply_override(config, assignment)
    config["git_hash"] = get_git_hash()  # to know the version of the code
    train(**config)


if __name__ == "__main__":
    train_cli()
