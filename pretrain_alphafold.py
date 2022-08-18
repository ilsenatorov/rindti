import torch_geometric
import torch_geometric.transforms as T
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger

from rindti.data import DynamicBatchSampler, LargePreTrainDataset
from rindti.data.transforms import MaskType, PosNoise
from rindti.models.pretrain.denoise import DenoiseModel

if __name__ == "__main__":

    seed_everything(42)
    pre_transform = T.Compose(
        [
            T.Center(),
            T.NormalizeRotation(),
            T.RadiusGraph(r=7),
            T.ToUndirected(),
        ]
    )
    transform = T.Compose(
        [
            PosNoise(sigma=0.75),
            MaskType(prob=0.15),
        ]
    )

    ds = LargePreTrainDataset(
        "/scratch/SCRATCH_NVME/ilya/datasets/uniref50/resources/structures/",
        transform=transform,
        pre_transform=pre_transform,
        threads=128,
    )
    sampler = DynamicBatchSampler(ds, max_num=4000)
    model = DenoiseModel(dropout=0.1, hidden_dim=1024, num_layers=12, num_heads=4, weighted_loss=False)
    dl = torch_geometric.loader.DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=32,
    )
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
            RichProgressBar(),
            RichModelSummary(),
            ModelCheckpoint(monitor="train_loss", mode="min"),
        ],
        logger=logger,
    )
    trainer.fit(model, dl)
