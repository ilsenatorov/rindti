from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger

from rindti.data import ProteinDataModule
from rindti.models.pretrain.denoise import DenoiseModel

if __name__ == "__main__":

    seed_everything(42)
    dm = ProteinDataModule("datasets/alphafold/resources/structures/", batch_sampling=True, max_num_nodes=5000)
    model = DenoiseModel(dropout=0.1, hidden_dim=64, num_layers=4, num_heads=4, weighted_loss=False)
    logger = WandbLogger(
        name="pretrain_alphafold",
        save_dir="wandb_logs",
        log_model=True,
    )
    trainer = Trainer(
        gpus=-1,
        gradient_clip_val=1,
        callbacks=[
            StochasticWeightAveraging(swa_lrs=1e-2),
            ModelCheckpoint(monitor="train_loss", mode="min"),
            EarlyStopping(monitor="train_loss", mode="min", patience=10),
            RichProgressBar(),
            RichModelSummary(),
        ],
        logger=logger,
    )
    trainer.fit(model, dm)
