from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from rindti.data import PreTrainDataModule
from rindti.models import (
    BGRLModel,
    DistanceModel,
    GraphLogModel,
    InfoGraphModel,
    ProtClassModel,
)
from rindti.utils import read_config

models = {
    "graphlog": GraphLogModel,
    "infograph": InfoGraphModel,
    "class": ProtClassModel,
    "bgrl": BGRLModel,
    "distance": DistanceModel,
}


def pretrain(**kwargs):
    """Run pretraining pipeline"""
    seed_everything(kwargs["seed"])
    dm = PreTrainDataModule(**kwargs["datamodule"])
    dm.setup()
    dm.update_config(kwargs)
    logger = TensorBoardLogger("tb_logs", name="prot_test", default_hp_metric=False)
    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min"),
        EarlyStopping(monitor="val_loss", mode="min", **kwargs["early_stop"]),
        RichModelSummary(),
        RichProgressBar(),
    ]
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        deterministic=False,
        **kwargs["trainer"],
    )
    model = models[kwargs["model"]["module"]](**kwargs)
    trainer.fit(model, dm)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pprint import pprint

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = read_config(args.config)
    pprint(config)
    pretrain(**config)
