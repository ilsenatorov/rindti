import os
import random

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger

from rindti.data import DTIDataModule
from rindti.models import ClassificationModel, RegressionModel
from rindti.utils import get_git_hash, read_config

models = {
    "class": ClassificationModel,
    "reg": RegressionModel,
}


def train(folder, version, **kwargs):
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
        StochasticWeightAveraging(swa_lrs=kwargs["swa_lrs"]),
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
    from pprint import pprint

    pprint(kwargs)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code
    train(**orig_config)
