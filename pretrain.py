from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from rindti.data import PreTrainDataModule
from rindti.models import BGRLModel, DistanceModel, GraphLogModel, InfoGraphModel, ProtClassESMModel, ProtClassModel
from rindti.utils import read_config

models = {
    "graphlog": GraphLogModel,
    "infograph": InfoGraphModel,
    "class": ProtClassModel,
    "bgrl": BGRLModel,
    "distance": DistanceModel,
    "esmclass": ProtClassESMModel,
}


def pretrain(**kwargs):
    """Run pretraining pipeline"""
    seed_everything(kwargs["seed"])
    dm = PreTrainDataModule(**kwargs["datamodule"])
    dm.setup()
    dm.update_config(kwargs)
    ## TODO need a more elegant solution for this
    labels = dm.get_labels()
    kwargs["model"]["label_list"] = list(labels)
    kwargs["model"]["encoder"]["data"]["feat_dim"] = 20
    kwargs["model"]["encoder"]["data"]["edge_dim"] = 5
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
    model = models[kwargs["model"].pop("module")](**kwargs["model"])
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
