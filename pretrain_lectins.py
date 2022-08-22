import os

import torch_geometric
import torch_geometric.transforms as T
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DynamicBatchSampler

from rindti.data.lectin_datasets import LectinPreTrainDataset
from rindti.data.transforms import MaskType, PosNoise, MaskTypeWeighted, WeightedESMasker
from rindti.models.pretrain.lectin_denoise import DenoiseModel

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
            # AACounts(),
            PosNoise(sigma=0.75),
            # MaskType(prob=0.15),
            MaskTypeWeighted(prob=0.15),
            # WeightedESMasker(prob=0.15),
            # T.RadiusGraph(r=7),
            # T.ToUndirected(),
            # T.RandomRotate(degrees=180, axis=0),
            # T.RandomRotate(degrees=180, axis=1),
            # T.RandomRotate(degrees=180, axis=2),
        ]
    )

    ds = LectinPreTrainDataset(
        "/home/rjo21/Desktop/rindti/data/pre_lectin/resources/structures/",
        "/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle/resources/structures/",
        transform=transform,
        pre_transform=pre_transform,
        threads=64,
    )

    folder = os.path.join("tb_logs", "pre_lectin", "noisy")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if len(os.listdir(folder)) == 0:
        next_version = 0
    else:
        next_version = str(
            int(
                [d for d in os.listdir(folder) if "version" in d and os.path.isdir(os.path.join(folder, d))][-1].split(
                    "_"
                )[1]
            )
            + 1
        )

    logger = TensorBoardLogger(
        save_dir=folder,
        name=f"version_{next_version}",
        default_hp_metric=False,
    )
    sampler = DynamicBatchSampler(ds, max_num=3000, mode="node")
    model = DenoiseModel(dropout=0.1, hidden_dim=512, num_layers=4, num_heads=2, weighted_loss=False)
    dl = torch_geometric.loader.DataLoader(ds, batch_sampler=sampler, num_workers=16)
    trainer = Trainer(
        gpus=1,
        logger=logger,
        # accumulate_grad_batches=4,
        log_every_n_steps=10,
        max_epochs=4500,
        gradient_clip_val=1,
        callbacks=[
            ModelCheckpoint(save_last=True, monitor='train_loss', save_top_k=3, mode='min'),
            # StochasticWeightAveraging(swa_lrs=1e-2),
        ],
    )
    trainer.fit(model, dl)
