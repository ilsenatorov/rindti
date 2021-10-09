from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from rindti.models import ClassificationModel, NoisyNodesClassModel, NoisyNodesRegModel, RegressionModel
from rindti.utils.data import Dataset
from rindti.utils.transforms import GnomadTransformer

models = {
    "class": ClassificationModel,
    "reg": RegressionModel,
    "noisyclass": NoisyNodesClassModel,
    "noisyreg": NoisyNodesRegModel,
}


def train(**kwargs):
    """Train the whole model"""
    seed_everything(kwargs["seed"])
    if kwargs["transformer"] != "none":
        transform = {"gnomad": GnomadTransformer}[kwargs["transformer"]].from_pickle(
            kwargs["transformer_pickle"], max_num_mut=kwargs["max_num_mut"]
        )
    else:
        transform = None
    train = Dataset(kwargs["data"], split="train", transform=transform)
    val = Dataset(kwargs["data"], split="val")
    test = Dataset(kwargs["data"], split="test")

    kwargs.update(train.config)
    logger = TensorBoardLogger(
        "tb_logs", name=kwargs["model"] + ":" + kwargs["data"].split("/")[-1].split(".")[0], default_hp_metric=False
    )
    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min"),
        EarlyStopping(monitor="val_loss", patience=kwargs["early_stop_patience"], mode="min"),
    ]
    trainer = Trainer(
        gpus=kwargs["gpus"],
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=kwargs["gradient_clip_val"],
        deterministic=True,
        profiler=kwargs["profiler"],
    )
    model = models[kwargs["model"]](**kwargs)
    dataloader_kwargs = {k: v for (k, v) in kwargs.items() if k in ["batch_size", "num_workers"]}
    dataloader_kwargs.update({"follow_batch": ["prot_x", "drug_x"]})
    train_dataloader = DataLoader(train, **dataloader_kwargs, shuffle=False)
    val_dataloader = DataLoader(val, **dataloader_kwargs, shuffle=False)
    test_dataloader = DataLoader(test, **dataloader_kwargs, shuffle=False)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    import argparse

    from rindti.utils import MyArgParser

    tmp_parser = argparse.ArgumentParser(add_help=False)
    tmp_parser.add_argument("--model", type=str, default="class")
    args = tmp_parser.parse_known_args()[0]
    model_type = args.model

    parser = MyArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("data", type=str)
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loading")
    parser.add_argument("--early_stop_patience", type=int, default=60, help="epochs with no improvement before stop")
    parser.add_argument("--feat_method", type=str, default="element_l1", help="How to combine embeddings")

    trainer = parser.add_argument_group("Trainer")
    model = parser.add_argument_group("Model")
    optim = parser.add_argument_group("Optimiser")
    transformer = parser.add_argument_group("Transformer")

    trainer.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    trainer.add_argument("--max_epochs", type=int, default=1000, help="Max number of epochs")
    trainer.add_argument("--model", type=str, default="class", help="Type of model")
    trainer.add_argument("--weighted", type=bool, default=1, help="Whether to weight the data points")
    trainer.add_argument("--gradient_clip_val", type=float, default=30, help="Gradient clipping")
    trainer.add_argument("--profiler", type=str, default=None)

    model.add_argument("--mlp_hidden_dim", default=64, type=int, help="MLP hidden dims")
    model.add_argument("--mlp_dropout", default=0.2, type=float, help="MLP dropout")

    optim.add_argument("--optimiser", type=str, default="adamw", help="Optimisation algorithm")
    optim.add_argument("--momentum", type=float, default=0.3)
    optim.add_argument("--lr", type=float, default=0.0005, help="mlp learning rate")
    optim.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    optim.add_argument("--reduce_lr_patience", type=int, default=20)
    optim.add_argument("--reduce_lr_factor", type=float, default=0.1)
    optim.add_argument("--monitor", type=str, default="val_loss", help="Value to monitor for lr reduction etc")

    transformer.add_argument("--transformer", type=str, default="none", help="Type of transformer to apply")
    transformer.add_argument(
        "--transformer_pickle",
        type=str,
        default="../rins/results/prepare_transformer/onehot_simple_transformer.pkl",
    )
    transformer.add_argument("--max_num_mut", type=int, default=100)

    parser = models[model_type].add_arguments(parser)

    args = parser.parse_args()
    argvars = vars(args)
    train(**argvars)
