from pprint import pprint

import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from rindti.data import DTIDataModule
from rindti.models import ClassificationModel, ESMClassModel, RegressionModel
from rindti.utils import read_config

models = {
    "class": ClassificationModel,
    "reg": RegressionModel,
    "esm_class": ESMClassModel,
}


def train(**kwargs):
    """Train the whole model"""
    seed_everything(kwargs["seed"])
    tmp = np.arange(100)
    np.random.shuffle(tmp)
    seeds = tmp[: kwargs["runs"]]

    for i, seed in enumerate(seeds):
        print(f"Run {i+1} of {kwargs['runs']} with seed {seed}")
        kwargs["seed"] = seed
        single_run(**kwargs)


def single_run(**kwargs):
    """Does a single run."""
    seed_everything(kwargs["seed"])
    datamodule = DTIDataModule(**kwargs["datamodule"])
    datamodule.setup()
    datamodule.update_config(kwargs)
    pprint(datamodule.config)

    logger = TensorBoardLogger(
        "tb_logs",
        name="dti",
        default_hp_metric=False,
    )

    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min"),
        EarlyStopping(monitor="val_loss", mode="min", **kwargs["early_stop"]),
        RichModelSummary(),
        RichProgressBar(),
    ]
    trainer = Trainer(callbacks=callbacks, logger=logger, log_every_n_steps=25, **kwargs["trainer"])
    model = models[kwargs["model"].pop("module")](**kwargs["model"])
    pprint(model)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    train(**orig_config)
