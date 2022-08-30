import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from umap import UMAP

from ..data.pdb_parser import node_encode

node_decode = {v: k for k, v in node_encode.items()}


def plot_loss_count_dist(losses: dict) -> Figure:
    """Plot distribution of times sampled vs avg loss of families."""
    fig = plt.figure()
    plt.xlabel("Times sampled")
    plt.ylabel("Avg loss")
    plt.title("Prot statistics")
    count = [len(x) for x in losses.values()]
    mean = [np.mean(x) for x in losses.values()]
    plt.scatter(x=count, y=mean)
    return fig


def plot_aa_tsne(emb: torch.Tensor):
    """Plot PCA of amino acid embeddings."""
    node_decode = {v: k for k, v in node_encode.items()}
    tsne = PCA(n_components=2)
    emb_trans = tsne.fit_transform(emb)
    emb_trans = pd.DataFrame(emb_trans)
    emb_trans["size"] = 50
    emb_trans["name"] = emb_trans.index.to_series().apply(lambda x: node_decode[x])
    emb_trans["type"] = [
        "hydrophobic",
        "positive",
        "polar",
        "negative",
        "special",
        "polar",
        "negative",
        "special",
        "positive",
        "hydrophobic",
        "hydrophobic",
        "positive",
        "hydrophobic",
        "hydrophobic",
        "special",
        "polar",
        "polar",
        "hydrophobic",
        "hydrophobic",
        "hydrophobic",
        "mask",
    ]
    return px.scatter(
        emb_trans,
        0,
        1,
        text="name",
        color="type",
        symbol="type",
        opacity=0.5,
        width=400,
        height=400,
        size="size",
        title="AA UMAP",
    )


def plot_noise_pred(
    pos: torch.Tensor,
    pred_pos: torch.Tensor,
    edge_index: torch.Tensor,
    uniprot_id: str,
    width: int = 800,
    height: int = 800,
):

    """Plot the predicted coords vs real coords."""
    loss = F.mse_loss(pos, pred_pos, reduction="none").mean(dim=1).detach().cpu().numpy()
    df = pd.DataFrame(pos.detach().cpu().numpy(), columns=["x", "y", "z"])
    pred_df = pd.DataFrame(pred_pos.detach().cpu().numpy(), columns=["x", "y", "z"])
    edges = pd.DataFrame(edge_index.t().detach().cpu().numpy(), columns=["source", "target"])
    edges["x"] = edges["source"].map(df.x)
    edges["y"] = edges["source"].map(df.y)
    edges["z"] = edges["source"].map(df.z)
    nodes = go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        mode="markers",
        name="nodes",
        marker=dict(size=7, color=loss, colorscale="Viridis", showscale=True, cmin=0, cmax=4),
        text=[i for i in range(len(df))],
        customdata=loss,
        hovertemplate="<b> x:</b> %{x:.2f}<b> y:</b> %{y:.2f}<b> z:</b> %{z:.2f}<br>"
        + "<b>Number: %{text}</b><br>"
        + "<b>Loss: %{customdata:.2f}</b>",
    )
    x_lines = list()
    y_lines = list()
    z_lines = list()
    for p in edge_index.t():
        for i in range(2):
            x_lines.append(df.x[p[i].item()])
            y_lines.append(df.y[p[i].item()])
            z_lines.append(df.z[p[i].item()])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    edges = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        name="edges",
        line=dict(color="gray", width=0.5),
        hoverinfo="skip",
    )
    predicted_nodes = go.Scatter3d(
        x=pred_df["x"],
        y=pred_df["y"],
        z=pred_df["z"],
        mode="markers",
        name="predicted nodes",
        marker=dict(
            size=7,
            color=loss,
            colorscale="Viridis",
            opacity=0.5,
            showscale=False,
        ),
        hoverinfo="skip",
    )
    fig = go.Figure(data=[nodes, predicted_nodes, edges])
    fig.update_layout(
        width=width,
        height=height,
        showlegend=False,
        title=f"{uniprot_id}, Loss: {loss.mean().item():.4f}",
    )
    return fig


def plot_confmat(confmat):
    """Plot confusion matrix."""
    return px.imshow(
        confmat,
        zmin=0,
        zmax=1,
        text_auto=True,
        width=400,
        height=400,
        color_continuous_scale=px.colors.sequential.Viridis,
    )


def plot_node_embeddings(embeds: torch.Tensor, labels: torch.LongTensor, uniprot_ids: list):
    """Plot UMAP embeddings of the node embeddings."""
    numbers = []
    cur_prot = uniprot_ids[0]
    i = 0
    for prot in uniprot_ids:
        if prot == cur_prot:
            numbers.append(i)
            i += 1
        else:
            numbers.append(0)
            i = 1
            cur_prot = prot
    pca = UMAP(n_components=2)
    embedded_nodes = pca.fit_transform(embeds.detach().cpu().numpy())
    embedded_nodes = pd.DataFrame(embedded_nodes, columns=["x", "y"])
    embedded_nodes["name"] = labels.detach().cpu().numpy()
    embedded_nodes["name"] = embedded_nodes["name"].apply(lambda x: node_decode[x])
    embedded_nodes["uniprot_id"] = uniprot_ids
    embedded_nodes["number"] = numbers
    return px.scatter(
        embedded_nodes,
        x="x",
        y="y",
        color="uniprot_id",
        hover_data=["uniprot_id", "name", "number"],
        width=400,
        height=400,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )


def plot_noise_loss_vs_plddt(noise: torch.Tensor, noise_pred: torch.Tensor, plddt: torch.Tensor):
    """Plot the noise loss vs PLDDT loss."""
    loss = F.mse_loss(noise, noise_pred, reduction="none").mean(dim=1).detach().cpu().numpy()
    return px.scatter(
        x=loss,
        y=plddt.detach().cpu().numpy(),
        width=400,
        height=400,
        title="Noise Loss vs PLDDT",
    )
