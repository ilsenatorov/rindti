import sys
from pprint import pprint

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.data import DataLoader

from rindti.models import GraphLogModel, InfoGraphModel, PfamModel
from rindti.utils.data import PreTrainDataset, split_random
from rindti.utils.transforms import PfamTransformer
from rindti.utils.utils import MyArgParser

models = {"graphlog": GraphLogModel, "infograph": InfoGraphModel, "pfam": PfamModel}


def pretrain(**kwargs):
    """Run pretraining pipeline"""
    pprint(kwargs)
    if kwargs["model"] == "pfam":
        transformer = PfamTransformer.from_pickle(kwargs["data"])
    else:
        transformer = None
    dataset = PreTrainDataset(kwargs["data"], transform=transformer)
    kwargs.update(dataset.config)
    pprint(kwargs)
    kwargs["feat_dim"] = 20
    train, val = split_random(dataset)
    logger = TensorBoardLogger("tb_logs", name=kwargs["model"], default_hp_metric=False)
    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min"),
        EarlyStopping(monitor="val_loss", patience=kwargs["early_stop_patience"], mode="min"),
    ]
    trainer = Trainer(
        gpus=kwargs["gpus"],
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=30,
        max_epochs=kwargs["max_epochs"],
        stochastic_weight_avg=True,
        num_sanity_val_steps=0,
    )
    model = models[kwargs["model"]](**kwargs)
    follow_batch = ["a_x", "b_x"] if kwargs["model"] == "pfam" else []
    train_dl = DataLoader(
        train,
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
        follow_batch=follow_batch,
        shuffle=True,
    )
    val_dl = DataLoader(
        val, batch_size=kwargs["batch_size"], num_workers=kwargs["num_workers"], follow_batch=follow_batch
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    import argparse

    tmp_parser = argparse.ArgumentParser(add_help=False)
    tmp_parser.add_argument("--model", type=str, default="graphlog")
    args = tmp_parser.parse_known_args()[0]
    model_type = args.model
    parser = MyArgParser(
        prog="Model Trainer",
        usage="""
Run with python train.py <data pickle file>
To get help for different models run with python pretrain.py --help --model <model name>
To get help for different modules run with python pretrain.py --help --prot_node_embed <module name> """,
    )

    parser.add_argument("data", type=str)
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loading")
    parser.add_argument("--early_stop_patience", type=int, default=60, help="epochs with no improvement before stop")
    parser.add_argument("--max_epochs", type=int, default=None, help="Max number of epochs to train for")

    trainer = parser.add_argument_group("Trainer")
    model = parser.add_argument_group("Model")
    optim = parser.add_argument_group("Optimiser")

    trainer.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    trainer.add_argument("--max_epochs", type=int, default=1000, help="Max number of epochs")
    trainer.add_argument("--weighted", type=bool, default=1, help="Whether to weight the data points")
    trainer.add_argument("--gradient_clip_val", type=float, default=10, help="Gradient clipping")
    trainer.add_argument("--model", type=str, default="graphlog", help="Type of model")

    optim.add_argument("--optimiser", type=str, default="adamw", help="Optimisation algorithm")
    optim.add_argument("--momentum", type=float, default=0.3)
    optim.add_argument("--lr", type=float, default=0.0005, help="mlp learning rate")
    optim.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    optim.add_argument("--reduce_lr_patience", type=int, default=20)
    optim.add_argument("--reduce_lr_factor", type=float, default=0.1)

    parser = models[model_type].add_arguments(parser)

    args = parser.parse_args()
    argvars = vars(args)
    pretrain(**argvars)
