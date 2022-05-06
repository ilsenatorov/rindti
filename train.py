from pprint import pprint

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from rindti.data import DTIDataModule
from rindti.models import ClassificationModel, ESMClassModel, RegressionModel
from rindti.utils import hparams_config, read_config

models = {
    "class": ClassificationModel,
    "reg": RegressionModel,
    "esm_class": ESMClassModel,
}


def train(**kwargs):
    """Train the whole model"""
    seed_everything(kwargs["seed"])
    datamodule = DTIDataModule(kwargs["data"], kwargs["batch_size"], kwargs["num_workers"])
    datamodule.setup()
    pprint(datamodule.config)
    kwargs.update(datamodule.config)
    logger = TensorBoardLogger(
        "tb_logs",
        name="dti",
        default_hp_metric=False,
    )

    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min"),
        EarlyStopping(monitor="val_loss", patience=kwargs["early_stop_patience"], mode="min"),
        RichModelSummary(),
        RichProgressBar(),
    ]
    trainer = Trainer(
        gpus=kwargs["gpus"],
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=kwargs["gradient_clip_val"],
        profiler=kwargs["profiler"],
        log_every_n_steps=25,
    )
    model = models[kwargs["model"]](**kwargs)
    pprint(model)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    import argparse
    import os

    from rindti.utils import MyArgParser

    tmp_parser = argparse.ArgumentParser(add_help=False)
    tmp_parser.add_argument("--model", type=str, default="class")
    args = tmp_parser.parse_known_args()[0]
    model_type = args.model

    parser = MyArgParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    configs = hparams_config(orig_config)
    for config in configs:
        train(**config)
