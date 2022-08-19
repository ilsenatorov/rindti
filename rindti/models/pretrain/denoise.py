import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
import wandb
from graphgps.layer.gps_layer import GPSLayer
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from torchmetrics import ConfusionMatrix
from torchmetrics.functional.classification import accuracy

from ...data.pdb_parser import node_encode
from ...utils.optim import LinearWarmupCosineAnnealingLR


class DenoiseModel(LightningModule):
    """Uses GraphGPS transformer layers to encode the graph and predict noise and node type."""

    def __init__(
        self,
        local_module: str = "GAT",
        global_module: str = "Performer",
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        optimizer: str = "Adam",
        max_lr: float = 1e-4,
        start_lr: float = 1e-5,
        min_lr: float = 1e-7,
        weighted_loss: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.weighted_loss = weighted_loss
        mid_dim = hidden_dim // 2
        self.feat_encode = torch.nn.Embedding(21, mid_dim)
        self.pos_encode = torch.nn.Linear(3, mid_dim)
        self.node_encode = torch.nn.Sequential(
            *[
                GPSLayer(
                    hidden_dim,
                    local_module,
                    global_module,
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
        self.confmat = ConfusionMatrix(num_classes=20, normalize="true")

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        feat_encode = self.feat_encode(batch.x)
        pos_encode = self.pos_encode(batch.pos)
        batch.x = torch.cat([feat_encode, pos_encode], dim=1)
        batch = self.node_encode(batch)
        batch.noise_pred = self.noise_pred(batch.x)
        batch.type_pred = self.type_pred(batch.x[batch.mask])
        return batch

    def shared_step(self, batch: Data, step: int) -> dict:
        """Shared step for training and validation."""
        batch = self.forward(batch)
        noise_loss = F.mse_loss(batch.noise_pred, batch.noise)
        pred_loss = F.cross_entropy(
            batch.type_pred,
            batch.orig_x,
        )
        loss = noise_loss + 0.5 * pred_loss
        acc = accuracy(batch.type_pred, batch.orig_x)
        self.log(f"{step}_loss", loss)
        self.log(f"{step}_noise_loss", noise_loss)
        self.log(f"{step}_pred_loss", pred_loss)
        self.log(f"{step}_acc", acc)
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        self.confmat.update(batch.type_pred, batch.orig_x)
        return dict(
            loss=loss,
            noise_loss=noise_loss.detach(),
            pred_loss=pred_loss.detach(),
            acc=acc.detach(),
        )

    def training_step(self, batch):
        """Just shared step"""
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs) -> None:
        """Log the confusion matrix and the histograms"""
        confmat = self.confmat.compute().detach().cpu().numpy()
        confmat = pd.DataFrame(confmat, index=node_encode.keys(), columns=node_encode.keys()).round(2)
        self.confmat.reset()
        fig = px.imshow(
            confmat,
            zmin=0,
            zmax=1,
            text_auto=True,
            width=400,
            height=400,
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        wandb.log({"chart": fig})

    def configure_optimizers(self):
        """Adam optimizer with linear warmup and cosine annealing."""
        if self.optimizer == "Adam":
            optim = torch.optim.AdamW(self.parameters(), lr=self.max_lr, betas=(0.9, 0.95), weight_decay=1e-5)
        else:
            optim = torch.optim.SGD(self.parameters(), lr=self.max_lr, momentum=0.9, weight_decay=1e-5)
        scheduler = LinearWarmupCosineAnnealingLR(
            optim,
            warmup_epochs=5,
            max_epochs=100,
            warmup_start_lr=self.start_lr,
            eta_min=self.min_lr,
        )
        return [optim], [scheduler]
