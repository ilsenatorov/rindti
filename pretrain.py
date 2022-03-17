from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from rindti.data import PreTrainDataModule
from rindti.models import BGRLModel, DistanceModel, GraphLogModel, InfoGraphModel, ProtClassESMModel, ProtClassModel
from rindti.utils import MyArgParser, read_config

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
    dm = PreTrainDataModule(kwargs["data"])
    dm.setup()
    ## TODO need a more elegant solution for this
    labels = dm.get_labels()
    kwargs["label_list"] = list(labels)
    kwargs.update(dm.config)
    kwargs["feat_dim"] = 20
    kwargs["edge_dim"] = 5
    logger = TensorBoardLogger("tb_logs", name=kwargs["model"], default_hp_metric=False)
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
        max_epochs=kwargs["max_epochs"],
        num_sanity_val_steps=0,
        deterministic=False,
        profiler=kwargs["profiler"],
    )
    model = models[kwargs["model"]](**kwargs)
    # if kwargs["model"] == "distance":
    #     sampler = PfamSampler(
    #         dataset,
    #         batch_size=kwargs["batch_size"],
    #         prot_per_fam=kwargs["prot_per_fam"],
    #     )
    #     dl = DataLoader(
    #         dataset,
    #         batch_sampler=sampler,
    #         num_workers=kwargs["num_workers"],
    #     )
    # else:
    #     dl = DataLoader(dataset, batch_size=kwargs["batch_size"], num_workers=kwargs["num_workers"], shuffle=True)
    trainer.fit(model, dm)


if __name__ == "__main__":
    from pprint import pprint

    parser = MyArgParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = read_config(args.config)
    pprint(config)
    pretrain(**config)
