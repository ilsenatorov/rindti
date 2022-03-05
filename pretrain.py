from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from rindti.data import PreTrainDataset
from rindti.models import BGRLModel, DistanceModel, GraphLogModel, InfoGraphModel, ProtClassModel
from rindti.utils import MyArgParser, read_config

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
    dataset = PreTrainDataset(kwargs["data"])
    ## TODO need a more elegant solution for this
    labels = {i.y for i in dataset}
    kwargs["label_list"] = list(labels)
    kwargs.update(dataset.config)
    kwargs["feat_dim"] = 20
    kwargs["edge_dim"] = 5
    logger = TensorBoardLogger("tb_logs", name=kwargs["model"], default_hp_metric=False)
    callbacks = [
        ModelCheckpoint(monitor="train_loss", save_top_k=3, mode="min"),
        EarlyStopping(monitor="train_loss", patience=kwargs["early_stop_patience"], mode="min"),
    ]
    trainer = Trainer(
        gpus=kwargs["gpus"],
        callbacks=callbacks,
        logger=logger,
        max_epochs=kwargs["max_epochs"],
        num_sanity_val_steps=0,
        deterministic=False,
        profiler=kwargs["profiler"],
    )
    model = models[kwargs["model"]](**kwargs)
    dl = DataLoader(dataset, batch_size=kwargs["batch_size"], num_workers=kwargs["num_workers"], shuffle=True)
    trainer.fit(model, dl)


if __name__ == "__main__":
    from pprint import pprint

    parser = MyArgParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = read_config(args.config)
    pprint(config)
    pretrain(**config)
