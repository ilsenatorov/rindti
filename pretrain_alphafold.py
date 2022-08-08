import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from graphgps.layer.gps_layer import GPSLayer
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import ConfusionMatrix
from torchmetrics.functional.classification import accuracy

from rindti.data import LargePreTrainDataset
from rindti.data.large_datasets import node_encode


class PosNoise:
    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma

    def __call__(self, batch) -> torch.Tensor:
        noise = torch.randn_like(batch.pos) * self.sigma
        batch.pos += noise
        batch.noise = noise
        return batch


class MaskType:
    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, batch) -> torch.Tensor:
        mask = torch.rand_like(batch.x, dtype=torch.float32) < self.prob
        batch.orig_x = batch.x[mask]
        batch.x[mask] = 20
        batch.mask = mask
        return batch


class GPSPosModel(LightningModule):
    def __init__(self, hidden_dim: int = 512, num_layers: int = 6, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        mid_dim = hidden_dim // 2
        self.feat_encode = torch.nn.Embedding(21, mid_dim)
        self.pos_encode = torch.nn.Linear(3, mid_dim)
        # self.edge_encode = torch.nn.Linear(3, hidden_dim)
        self.node_encode = torch.nn.Sequential(
            *[
                GPSLayer(
                    hidden_dim,
                    "GAT",
                    "Transformer",
                    num_heads,
                    dropout=dropout,
                    attn_dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.noise_pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 3),
        )
        self.type_pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 20),
        )
        # self.plddt_pred = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_dim, hidden_dim // 2),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(hidden_dim // 2, 1),
        # )
        self.confmat = ConfusionMatrix(num_classes=20, normalize="true")

    def forward(self, batch):
        feat_encode = self.feat_encode(batch.x)
        pos_encode = self.pos_encode(batch.pos)
        # batch.edge_attr = self.edge_encode(batch.edge_attr)
        batch.x = torch.cat([feat_encode, pos_encode], dim=1)
        batch = self.node_encode(batch)
        batch.noise_pred = self.noise_pred(batch.x)
        batch.type_pred = self.type_pred(batch.x[batch.mask])
        # batch.plddt_pred = self.plddt_pred(batch.x)
        return batch

    def shared_step(self, batch, step):
        batch = self.forward(batch)
        noise_loss = F.mse_loss(batch.noise_pred, batch.noise)
        pred_loss = F.cross_entropy(
            batch.type_pred,
            batch.orig_x,
            weight=torch.tensor(
                [
                    0.0189,
                    0.0299,
                    0.0479,
                    0.0311,
                    0.1520,
                    0.0449,
                    0.0254,
                    0.0232,
                    0.0780,
                    0.0291,
                    0.0180,
                    0.0316,
                    0.0698,
                    0.0469,
                    0.0383,
                    0.0301,
                    0.0332,
                    0.1637,
                    0.0638,
                    0.0243,
                ],
                device=self.device,
            ),
        )
        # plddt_loss = F.mse_loss(batch.plddt_pred * 0.01, batch.plddt.unsqueeze(1) * 0.01)
        loss = noise_loss + 0.5 * pred_loss
        acc = accuracy(batch.type_pred, batch.orig_x)
        self.log(f"{step}_loss", loss)
        self.log(f"{step}_noise_loss", noise_loss)
        self.log(f"{step}_pred_loss", pred_loss)
        # self.log(f"{step}_plddt_loss", plddt_loss)
        self.log(f"{step}_acc", acc)
        self.confmat.update(batch.type_pred, batch.orig_x)
        return dict(
            loss=loss,
            noise_loss=noise_loss.detach(),
            pred_loss=pred_loss.detach(),
            # plddt_loss=plddt_loss.detach(),
            acc=acc.detach(),
        )

    def training_step(self, batch):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs) -> None:
        confmat = self.confmat.compute().detach().cpu().numpy()
        confmat = pd.DataFrame(confmat, index=node_encode.keys(), columns=node_encode.keys())
        self.confmat.reset()
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(confmat, annot=True, fmt=".2f", cbar=False)
        self.logger.experiment.add_figure("confmat", fig, self.current_epoch)
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=1e-5, total_steps=1000)
        return optim


if __name__ == "__main__":
    from pytorch_lightning import seed_everything

    seed_everything(42)
    pre_transform = T.Compose(
        [
            T.Center(),
            T.NormalizeRotation(),
            T.RadiusGraph(r=7),
            T.ToUndirected(),
            # T.Spherical(),
        ]
    )
    transform = T.Compose(
        [
            PosNoise(sigma=0.5),
            MaskType(prob=0.1),
            # T.RadiusGraph(r=7),
            # T.ToUndirected(),
            # T.RandomRotate(degrees=180, axis=0),
            # T.RandomRotate(degrees=180, axis=1),
            # T.RandomRotate(degrees=180, axis=2),
        ]
    )

    ds = LargePreTrainDataset(
        "/scratch/SCRATCH_NVME/ilya/datasets/uniref50/resources/structures/",
        transform=transform,
        pre_transform=pre_transform,
        threads=64,
    )
    model = GPSPosModel(dropout=0)
    dl = torch_geometric.loader.DataLoader(ds, batch_size=4, shuffle=True, num_workers=16)
    trainer = Trainer(
        gpus=1,
        accumulate_grad_batches=8,
        log_every_n_steps=100,
        max_epochs=10000,
    )
    trainer.fit(model, dl)
