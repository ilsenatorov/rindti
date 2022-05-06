from pprint import pprint

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


def train(**cfg):
    """Train the whole model"""
    seed_everything(cfg["seed"])
    datamodule = DTIDataModule(**cfg["datamodule"])
    datamodule.setup()
    datamodule.update_config(cfg)
    pprint(cfg)
    logger = TensorBoardLogger(
        "tb_logs",
        name="dti",  # FIXME fix naming of the model
        default_hp_metric=False,
    )
    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min"),
        EarlyStopping(monitor="val_loss", mode="min", **cfg["early_stop"]),
        RichModelSummary(),
        RichProgressBar(),
    ]
    trainer = Trainer(callbacks=callbacks, logger=logger, log_every_n_steps=25, **cfg["trainer"])
    model = models[cfg["model"].pop("module")](**cfg["model"])
    pprint(model)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    train(**orig_config)
