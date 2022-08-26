import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from graphgps.layer.gps_layer import GPSLayer
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from torchmetrics import ConfusionMatrix
from torchmetrics.functional.classification import accuracy

import wandb

from ...data.pdb_parser import node_encode
from ...utils import plot_aa_tsne, plot_confmat, plot_node_embeddings, plot_noise_pred


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
        attn_dropout: float = 0.1,
        alpha: float = 1.0,
        weighted_loss: bool = False,
    ):
        super().__init__()
        self.weighted_loss = weighted_loss
        self.alpha = alpha
        self.feat_encode = torch.nn.Embedding(21, hidden_dim)
        self.pos_encode = torch.nn.Linear(3, hidden_dim)
        self.node_encode = torch.nn.Sequential(
            *[
                GPSLayer(
                    hidden_dim,
                    local_module,
                    global_module,
                    num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.noise_pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 3),
        )
        self.type_pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 21),
        )
        self.confmat = ConfusionMatrix(num_classes=21, normalize="true")

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        feat_encode = self.feat_encode(batch.x)
        pos_encode = self.pos_encode(batch.pos)
        batch.x = ((feat_encode - pos_encode).pow(2) + 1e-8).sqrt()
        batch = self.node_encode(batch)
        batch.type_pred = self.type_pred(batch.x[batch.mask])
        batch.noise_pred = self.noise_pred(batch.x)
        return batch

    def log_confmat(self):
        """Log confusion matrix to wandb."""
        confmat_df = self.confmat.compute().detach().cpu().numpy()
        confmat_df = pd.DataFrame(confmat_df, index=node_encode.keys(), columns=node_encode.keys()).round(2)
        self.confmat.reset()
        return plot_confmat(confmat_df)

    def log_aa_embed(self):
        """Log t-SNE plot of amino acid embeddings."""
        aa = torch.tensor(range(21), dtype=torch.long, device=self.device)
        emb = self.feat_encode(aa).detach().cpu()
        return plot_aa_tsne(emb)

    def log_figs(self, step: str):
        """Log figures to wandb."""
        test_batch = self.forward(self.test_batch.clone())
        node_pca = plot_node_embeddings(
            test_batch.x, self.test_batch.x, [self.test_batch.uniprot_id[x] for x in self.test_batch.batch]
        )
        figs = {
            f"{step}/confmat": self.log_confmat(),
            f"{step}/aa_pca": self.log_aa_embed(),
            f"{step}/node_pca": node_pca,
        }
        for i in range(5):
            figs[f"{step}/noise_pred/{test_batch[i].uniprot_id}"] = plot_noise_pred(
                test_batch[i].pos - test_batch[i].noise,
                test_batch[i].pos - test_batch.noise_pred[test_batch.batch == i],
                test_batch[i].edge_index,
                test_batch[i].uniprot_id,
            )
        wandb.log(figs)

    def shared_step(self, batch: Data, step: int) -> dict:
        """Shared step for training and validation."""
        if self.global_step == 0:
            self.test_batch = batch.clone()
        batch = self.forward(batch)
        noise_loss = F.mse_loss(batch.noise_pred, batch.noise)
        pred_loss = F.cross_entropy(batch.type_pred, batch.orig_x)
        loss = noise_loss + self.alpha * pred_loss
        acc = accuracy(batch.type_pred, batch.orig_x)
        self.confmat.update(batch.type_pred, batch.orig_x)
        if self.global_step % 100 == 0:
            self.log(f"{step}/loss", loss)
            self.log(f"{step}/acc", acc)
            self.log(f"{step}/noise_loss", noise_loss)
            self.log(f"{step}/pred_loss", pred_loss)
        if self.global_step % 500 == 0:
            self.log_figs(step)
        return dict(
            loss=loss,
            noise_loss=noise_loss.detach(),
            pred_loss=pred_loss.detach(),
            acc=acc.detach(),
        )

    def training_step(self, batch):
        """Just shared step"""
        return self.shared_step(batch, "train")
