import torch_geometric
import torch_geometric.transforms as T
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import StochasticWeightAveraging

from rindti.data import LargePreTrainDataset
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
            MaskType(prob=0.125),
            # T.RadiusGraph(r=7),
            # T.ToUndirected(),
            # T.RandomRotate(degrees=180, axis=0),
            # T.RandomRotate(degrees=180, axis=1),
            # T.RandomRotate(degrees=180, axis=2),
        ]
    )

    ds = LargePreTrainDataset(
        "datasets/alphafold/resources/structures/",
        transform=transform,
        pre_transform=pre_transform,
        threads=64,
    )[:10000]
    model = DenoiseModel(dropout=0.1, hidden_dim=128, num_layers=3, num_heads=4, weighted_loss=False)
    dl = torch_geometric.loader.DataLoader(ds, batch_size=32, shuffle=True, num_workers=16)
    trainer = Trainer(
        gpus=1,
        # accumulate_grad_batches=4,
        log_every_n_steps=100,
        max_epochs=6000,
        gradient_clip_val=5,
        callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
    )
    trainer.fit(model, dl)
