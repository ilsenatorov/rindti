"""Command-line entry point for training RINDTI models."""

import collections
import os
import random
import re
from copy import deepcopy

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
from .utils import IterDict, get_git_hash, read_config

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


def _short_names(keys: list) -> dict:
    """Shortest unambiguous name per config key.

    Uses the leaf ("feat_method"), extending leftwards while it collides -
    node.module and pool.module are both "module", so a bare leaf would produce
    two identically-named sweep points.
    """
    names, depth = {}, 1
    remaining = set(keys)
    while remaining and depth <= 4:
        candidates = {k: ".".join(k.split(",")[-depth:]) for k in remaining}
        counts = collections.Counter(candidates.values())
        settled = {k: n for k, n in candidates.items() if counts[n] == 1}
        names.update(settled)
        remaining -= settled.keys()
        depth += 1
    names.update({k: k.replace(",", ".") for k in remaining})
    return names


def describe_variant(variant: dict, baseline: dict, names: dict = None) -> str:
    """Short tag naming only the settings that differ from the first variant.

    An ablation over several axes produces runs that are otherwise identical, so
    the tag has to carry what actually changed - "split=target-node.module=gatconv"
    rather than an opaque index.
    """
    flat_v, flat_b = _flatten_config(variant), _flatten_config(baseline)
    differing = [(k, v) for k, v in flat_v.items() if flat_b.get(k) != v]
    if not differing:
        return "base"
    if names is None:
        names = _short_names([k for k, _ in differing])
    return "-".join(f"{names.get(k, k.split(',')[-1])}={v}" for k, v in differing)


def _flatten_config(config: dict, prefix: str = "") -> dict:
    """Flatten nested config to comma-joined keys, for comparing variants."""
    flat = {}
    for key, value in config.items():
        path = f"{prefix},{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_config(value, path))
        else:
            flat[path] = value
    return flat


def train(**kwargs) -> None:
    """Train the model, expanding any list-valued config entry into a sweep.

    A config holding lists (``node: {module: [ginconv, gatconv]}``) is expanded by
    IterDict into one run per combination, so an ablation is a single invocation.
    Each combination is then repeated over ``runs`` seeds.
    """
    variants = IterDict()(kwargs)
    if len(variants) > 1:
        print(f"Sweep: {len(variants)} configurations x {kwargs['runs']} seeds")

    # Name keys against every axis in the sweep, not just the ones a given
    # variant changes, so tags stay consistent across the whole ablation.
    swept = sorted(
        {k for v in variants for k, val in _flatten_config(v).items() if _flatten_config(variants[0]).get(k) != val}
    )
    names = _short_names(swept)

    for n, variant in enumerate(variants):
        tag = describe_variant(variant, variants[0], names) if len(variants) > 1 else None
        if tag:
            print(f"\n=== Configuration {n + 1}/{len(variants)}: {tag} ===")
        train_one(tag=tag, **variant)


def train_one(tag: str = None, **kwargs) -> None:
    """Train one configuration for ``runs`` seeds, sharing a version directory."""
    seed_everything(kwargs["seed"])
    seeds = random.sample(range(1, 100), kwargs["runs"])

    dataset = os.path.splitext(os.path.basename(kwargs["datamodule"]["filename"]))[0]
    name = kwargs["datamodule"]["exp_name"]
    if tag:
        # Each sweep point gets its own directory, so hparams.yaml files and
        # checkpoints from different configurations cannot overwrite each other.
        name = f"{name}/{tag}"
    folder = os.path.join("tb_logs", f"dti_{name}", dataset)
    os.makedirs(folder, exist_ok=True)
    version = next_version(folder)

    for i, seed in enumerate(seeds):
        print(f"Run {i + 1} of {kwargs['runs']} with seed {seed}")
        kwargs["seed"] = seed
        single_run(folder, version, **kwargs)


def single_run(folder: str, version: int, **kwargs) -> None:
    """Does a single run.

    The config is copied first: GraphEncoder.update_params and
    DTIDataModule.update_config both mutate it in place, so a shared dict would
    carry one run's injected dims into the next run's saved hyperparameters.
    """
    kwargs = deepcopy(kwargs)
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
