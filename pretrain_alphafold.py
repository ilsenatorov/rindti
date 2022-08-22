from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from rindti.data import ProteinDataModule
from rindti.models.pretrain.denoise import DenoiseModel
from rindti.utils import read_config

if __name__ == "__main__":
    import sys

    config = read_config(sys.argv[1])
    seed_everything(config["seed"])
    dm = ProteinDataModule(**config["datamodule"])
    model = DenoiseModel(**config["model"])
    wandb.init(config=config, project="pretrain_alphafold", name="overfit")
    logger = WandbLogger(log_model=True)
    trainer = Trainer(
        gpus=-1,
        gradient_clip_val=1,
        callbacks=[
            StochasticWeightAveraging(swa_lrs=1e-2),
            ModelCheckpoint(monitor="train_loss", mode="min"),
            RichProgressBar(),
            RichModelSummary(),
        ],
        logger=logger,
    )
    trainer.fit(model, dm)
