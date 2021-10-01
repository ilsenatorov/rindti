from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from rindti.models import BGRLModel, GraphLogModel, InfoGraphModel, PfamModel
from rindti.utils.data import PreTrainDataset, split_random
from rindti.utils.transforms import PfamTransformer
from rindti.utils.utils import MyArgParser

models = {"graphlog": GraphLogModel, "infograph": InfoGraphModel, "pfam": PfamModel, "bgrl": BGRLModel}


def pretrain(**kwargs):
    """Run pretraining pipeline"""
    seed_everything(kwargs["seed"])
    dataset = PreTrainDataset(kwargs["data"])
    if kwargs["model"] == "pfam":
        transformer = PfamTransformer(dataset.get_pfams())
    else:
        transformer = None
    dataset = PreTrainDataset(kwargs["data"], transform=transformer)
    kwargs.update(dataset.config)
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
        max_epochs=kwargs["max_epochs"],
        num_sanity_val_steps=0,
        deterministic=True,
        profiler=kwargs["profiler"],
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
        val,
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
        follow_batch=follow_batch,
    )
    trainer.fit(model, train_dl, val_dl)


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
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers for data loading")
    parser.add_argument("--early_stop_patience", type=int, default=60, help="epochs with no improvement before stop")
    parser.add_argument("--max_epochs", type=int, default=None, help="Max number of epochs to train for")

    trainer = parser.add_argument_group("Trainer")
    model = parser.add_argument_group("Model")
    optim = parser.add_argument_group("Optimiser")

    trainer.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    trainer.add_argument("--max_epochs", type=int, default=1000, help="Max number of epochs")
    trainer.add_argument("--gradient_clip_val", type=float, default=10, help="Gradient clipping")
    trainer.add_argument("--model", type=str, default="graphlog", help="Type of model")
    trainer.add_argument("--profiler", type=str, default=None)

    optim.add_argument("--optimiser", type=str, default="adam", help="Optimisation algorithm")
    optim.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    optim.add_argument("--reduce_lr_patience", type=int, default=20)
    optim.add_argument("--reduce_lr_factor", type=float, default=0.1)

    parser = models[model_type].add_arguments(parser)

    args = parser.parse_args()
    argvars = vars(args)
    pretrain(**argvars)
