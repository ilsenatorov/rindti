from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from rindti.data import DTIDataset
from rindti.models import ClassificationModel, RegressionModel
from rindti.utils import read_config

models = {
    "class": ClassificationModel,
    "reg": RegressionModel,
}


def train(**kwargs):
    """Train the whole model"""
    seed_everything(kwargs["seed"])
    train = DTIDataset(kwargs["data"], split="train").shuffle()
    val = DTIDataset(kwargs["data"], split="val").shuffle()
    test = DTIDataset(kwargs["data"], split="test").shuffle()

    kwargs.update(train.config)
    logger = TensorBoardLogger(
        "tb_logs",
        name=f'{kwargs["model"]}:{kwargs["data"].split("/")[-1].split(".")[0]}',
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
    dataloader_kwargs = {k: v for (k, v) in kwargs.items() if k in ["batch_size", "num_workers"]}
    dataloader_kwargs["follow_batch"] = ["prot_x", "drug_x"]
    train_dataloader = DataLoader(train, **dataloader_kwargs, shuffle=False)
    val_dataloader = DataLoader(val, **dataloader_kwargs, shuffle=False)
    test_dataloader = DataLoader(test, **dataloader_kwargs, shuffle=False)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    from rindti.utils import MyArgParser

    tmp_parser = argparse.ArgumentParser(add_help=False)
    tmp_parser.add_argument("--model", type=str, default="class")
    args = tmp_parser.parse_known_args()[0]
    model_type = args.model

    parser = MyArgParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = read_config(args.config)
    pprint(config)
    train(**config)
