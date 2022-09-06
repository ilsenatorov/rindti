from statistics import NormalDist
from typing import Any

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchmetrics import Metric
import plotly.express as px


class DistOverlap(Metric):
    """A metric keeping track of the distributions of predicted values for positive and negative samples"""

    def __init__(self, prefix=""):
        super(DistOverlap, self).__init__()
        self.add_state("pos", default=[], dist_reduce_fx="cat")
        self.add_state("neg", default=[], dist_reduce_fx="cat")
        if prefix != "":
            self.prefix = ""
        else:
            self.prefix = prefix + "_"

    def __name__(self):
        return self.prefix + "DistOverlap"

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Store the predictions separated into those of positive samples and those of negative samples"""
        pos = preds[target == 1]
        neg = preds[target == 0]
        self.pos += pos
        self.neg += neg

    def compute(self) -> Any:
        """Calculate the metric based on the samples from the update rounds"""
        if len(self.pos) == 0 or len(self.neg) == 0:
            return torch.nan
        self.pos = torch.stack(self.pos)
        self.neg = torch.stack(self.neg)
        pos_mu, pos_sigma = torch.mean(self.pos), torch.std(self.pos)
        neg_mu, neg_sigma = torch.mean(self.neg), torch.std(self.neg)

        return torch.tensor(
            NormalDist(mu=pos_mu.item(), sigma=pos_sigma.item()).overlap(
                NormalDist(mu=neg_mu.item(), sigma=neg_sigma.item())
            )
        )


class EmbeddingMetric(Metric):
    def __init__(self, classes, prefix=""):
        super(EmbeddingMetric, self).__init__()
        self.classes = classes
        self.add_state("embeds", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        if prefix != "":
            self.prefix = ""
        else:
            self.prefix = prefix + "_"

    def __name__(self):
        return self.prefix + "DistOverlap"

    def update(self, embeddings, labels) -> None:
        self.embeds += embeddings
        self.labels += labels

    def compute(self) -> Any:
        self.embeds = torch.stack(self.embeds).cpu().numpy()
        self.labels = torch.stack(self.labels).cpu().numpy()
        tsne_embeds = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50).fit_transform(
            self.embeds
        )
        for i, c in enumerate(self.classes):
            plt.scatter(tsne_embeds[self.labels == i, 0], tsne_embeds[self.labels == i, 1], label=c, s=20)
        embed_df = pd.DataFrame(tsne_embeds)
        embed_df["size"] = 50
        embed_df["type"] = list(map(lambda x: self.classes[x], self.labels))
        return px.scatter(
            embed_df,
            0,
            1,
            color="type",
            symbol="type",
            opacity=0.5,
            width=400,
            height=400,
            size="size",
            title="Embedding tSNE",
        )
