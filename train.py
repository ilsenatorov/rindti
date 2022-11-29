import os
import random
from pprint import pprint

import numpy as np
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from rindti.data import DTIDataModule
from rindti.data.transforms import DataCorruptor, ESMasker, NeighborhoodMasker, NullTransformer
from rindti.models import ClassificationModel, MultitaskClassification, RegressionModel
from rindti.utils import get_git_hash, read_config
from rindti.utils.ddd_es import DeepDoubleDescentEarlyStopping as DDDES

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["WANDB_CACHE_DIR"] = "/scratch/SCRATCH_SAS/roman/.cache/wandb"


models = {
    "class": ClassificationModel,
    "reg": RegressionModel,
    "multiclass": MultitaskClassification,
}


transformers = {
    "none": NullTransformer,
    "neighbor": NeighborhoodMasker,
    "esm": ESMasker,
    "corrupter": DataCorruptor,
}


def train(**kwargs):
    """Train the whole model"""
    seed_everything(kwargs["seed"])
    seeds = random.sample(range(1, 100), kwargs["runs"])

    folder = os.path.join(
        "tb_logs",
        f"dti_{kwargs['datamodule']['exp_name']}",
        f"{kwargs['datamodule']['filename'].split('/')[-1].split('.')[0]}",
    )
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if len(os.listdir(folder)) == 0:
        next_version = 0
    else:
        next_version = str(
            int(
                [d for d in os.listdir(folder) if "version" in d and os.path.isdir(os.path.join(folder, d))][-1].split(
                    "_"
                )[1]
            )
            + 1
        )

    for i, seed in enumerate(seeds):
        print(f"Run {i+1} of {kwargs['runs']} with seed {seed}")
        kwargs["seed"] = seed
        single_run(folder, next_version, **kwargs)


def single_run(folder, version, **kwargs):
    """Does a single run."""
    seed_everything(kwargs["seed"])
    datamodule = DTIDataModule(**kwargs["datamodule"])
    datamodule.setup(transform=transformers[kwargs["transform"]["mode"]](**kwargs["transform"]))
    datamodule.update_config(kwargs)

    # logger = TensorBoardLogger(
    #     save_dir=folder,
    #     name=f"version_{version}",
    #     version=kwargs["seed"],
    #     default_hp_metric=False,
    # )
    logger = WandbLogger(
        log_model='all',
        project="glylec_dti",
        name=kwargs["name"],
    )
    # wandb.config.update(kwargs)
    logger.experiment.config.update(kwargs)

    callbacks = [
        ModelCheckpoint(save_last=True, **kwargs["checkpoints"]),
        # EarlyStopping(monitor=kwargs["early_stop"]["monitor"], mode="min", **kwargs["early_stop"]),
        # DDDES(**kwargs["early_stop"]),
        RichModelSummary(),
        RichProgressBar(),
    ]
    # kwargs["trainer"]["max_epochs"] = 10
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        # limit_train_batches=10,
        # limit_val_batches=10,
        # limit_test_batches=10,
        log_every_n_steps=25,
        enable_model_summary=False,
        **kwargs["trainer"],
    )

    """effective_num = np.array([1, 1]) - np.array([np.power(0.36786104643297, 0.5), np.power(0.36786104643297, 16)])
    weights = (1 - 0.9999) / np.array(effective_num)
    weights = weights / sum(weights) * 2"""
    weights = [1, 1]

    model = models[kwargs["model"]["module"]](pos_weight=weights[0], neg_weight=weights[1], **kwargs)

    # pprint(kwargs)
    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code
    train(**orig_config)
