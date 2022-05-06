import random

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from rindti.data import DTIDataModule
from rindti.models import ClassificationModel, ESMClassModel, RegressionModel
from rindti.utils import get_git_hash, read_config

models = {
    "class": ClassificationModel,
    "reg": RegressionModel,
    "esm_class": ESMClassModel,
}


def train(**kwargs):
    """Train the whole model"""
    seed_everything(kwargs["seed"])
    seeds = random.sample(range(1, 1000), kwargs["runs"])

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
    model = models[kwargs["model"]["module"]](**kwargs)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code
    train(**orig_config)
