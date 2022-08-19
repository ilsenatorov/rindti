from pprint import pprint

import torch_geometric
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger

from rindti.data import DTIDataset, DynamicBatchSampler
from rindti.models.dti import ClassificationModel
from rindti.utils.cli import read_config

if __name__ == "__main__":
    kwargs = read_config("config/test.yaml")
    seed_everything(kwargs["seed"])

    ds = DTIDataset(kwargs["datamodule"]["filename"])
    pprint(ds.config)
    kwargs["model"]["prot_encoder"].update(ds.config["snakemake"]["data"]["prot"])
    kwargs["model"]["drug_encoder"].update(ds.config["snakemake"]["data"]["drug"])
    sampler = DynamicBatchSampler(ds, max_num=4000)
    model = ClassificationModel(
        "graph", kwargs["model"]["prot_encoder"], "graph", kwargs["model"]["drug_encoder"], {}, "concat"
    )
    dl = torch_geometric.loader.DataLoader(ds, batch_sampler=sampler, num_workers=32)
    logger = WandbLogger(name="pretrain_alphafold", save_dir="wandb_logs", log_model=True)
    trainer = Trainer(
        gpus=-1,
        gradient_clip_val=1,
        callbacks=[
            StochasticWeightAveraging(swa_lrs=1e-2),
            ModelCheckpoint(monitor="val_loss", mode="min"),
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
            RichProgressBar(),
            RichModelSummary(),
        ],
        logger=logger,
    )
    trainer.fit(model, dl)
