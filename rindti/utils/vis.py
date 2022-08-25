import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from ..data.pdb_parser import node_encode


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
    emb_trans["size"] = 20
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
    ]
    return px.scatter(
        emb_trans, 0, 1, text="name", color="type", opacity=0.5, width=400, height=400, size="size", title="AA PCA"
    )
