import os
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

    folder = os.path.join("tb_logs", f"dti{kwargs['model']}:{kwargs['exp_name']}", kwargs["data"].split("/")[-1].split(".")[0])
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if len(os.listdir(folder)) == 0:
        next_version = 0
    else:
        next_version = str(int([d for d in os.listdir(folder) if "version" in d and os.path.isdir(os.path.join(folder, d))][-1].split("_")[1]) + 1)

    for i, seed in enumerate(seeds):
        print(f"Run {i+1} of {kwargs['runs']} with seed {seed}")
        kwargs["seed"] = seed
        single_run(folder, next_version, **kwargs)


def single_run(**kwargs):
    """Does a single run."""
    seed_everything(kwargs["seed"])
    datamodule = DTIDataModule(**kwargs["datamodule"])
    datamodule.setup()
    pprint(datamodule.config)

    # kwargs.update(datamodule.config)
    for key, value in datamodule.config.items():
        if key not in kwargs:
            kwargs[key] = value

    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name=f"dti{kwargs['datamodule']['exp_name']}_{kwargs['datamodule']['filename'].split('/')[-1].split('.')[0]}",
        save_dir=folder,
        name=f"version_{version}",
        version=kwargs["seed"],
        default_hp_metric=False,
    )

    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min"),
        EarlyStopping(monitor="val_loss", mode="min", **kwargs["early_stop"]),
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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code
    train(**orig_config)
