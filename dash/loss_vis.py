# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import json
from typing import Union

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from rindti.losses import GeneralisedLiftedStructureLoss, SoftNearestNeighborLoss


def compress(dist, scale):
    centre = dist.mean(dim=0)
    displacement = (centre - dist).norm(dim=1, keepdim=True)
    return dist / (1 + displacement * scale)


class Plotter:
    """Plots the loss as a function of the distance between the two points"""

    def __init__(self, num: int = 100, max_dims: int = 1024):
        self.num = num
        self.dist = torch.normal(0, 1, (num, max_dims))

    def get_data(
        self,
        dims: int,
        distance: float,
        scale: float,
        dim_reducer: Union[PCA, UMAP, TSNE],
        loss: Union[SoftNearestNeighborLoss, GeneralisedLiftedStructureLoss],
        **loss_kwargs,
    ) -> pd.DataFrame:
        """Dataframe of all the points, with coordinates, loss and distance."""
        dist = self.dist[:, :dims]
        a = compress(dist, scale) + torch.ones(dims) * distance
        b = compress(dist, scale) - torch.ones(dims) * distance
        data = torch.cat((a, b))
        fam_idx = torch.tensor([0] * self.num + [1] * self.num).view(-1, 1)
        loss = loss(**loss_kwargs)
        losses = loss.forward(data, fam_idx).tolist()
        if data.shape[1] > 2:
            data = dim_reducer(n_components=2).fit_transform(data)
        else:
            data = data.numpy()
        data = pd.DataFrame(data, columns=["x", "y"])
        data["distance"] = distance
        data["scale"] = scale
        data["loss"] = losses
        data["symbol"] = fam_idx.view(-1).tolist()
        return data

    def get_loss(self, *args, **kwargs):
        """Calculate average loss for given entry"""
        data = self.get_data(*args, **kwargs)
        return data["loss"].mean()


app = dash.Dash(__name__)


app.layout = html.Div(
    style={"display": "flex", "flex-direction": "row"},
    children=[
        dcc.Store(id="plotter"),
        html.Div([dcc.Graph(id="fig1")]),
        html.Div(
            [
                html.H5("Distance"),
                dcc.Slider(id="distance", min=0, max=2, step=0.01, value=0),
                html.H5("Scale"),
                dcc.Slider(id="scale", min=0, max=2, step=0.01, value=0),
                html.H5("Temperature"),
                dcc.Slider(id="temperature", min=-3, max=3, step=0.1, value=0),
                html.H5("Positive Margin"),
                dcc.Slider(id="pos_margin", min=0, max=10, step=1, value=0),
                html.H5("Negative Margin"),
                dcc.Slider(id="neg_margin", min=0, max=10, step=1, value=0),
                html.H5("Loss"),
                dcc.Dropdown(
                    id="loss",
                    options=[
                        {"label": "Lifted Structure", "value": "lifted"},
                        {"label": "Soft Nearest Neighbor", "value": "snnl"},
                    ],
                    value="snnl",
                ),
                html.H5("Dimensions"),
                dcc.Dropdown(
                    id="dims",
                    options=[
                        {"label": "2", "value": 2},
                        {"label": "32", "value": 32},
                        {"label": "64", "value": 64},
                        {"label": "128", "value": 128},
                    ],
                    value=2,
                ),
                html.H5("Dimensionality Reduction"),
                dcc.Dropdown(
                    id="dim_reducer",
                    options=[
                        {"label": "PCA", "value": "pca"},
                        {"label": "UMAP", "value": "umap"},
                        {"label": "TSNE", "value": "tsne"},
                    ],
                    value="tsne",
                ),
            ],
            style={"padding": 10, "flex": 1},
        ),
    ],
)


p = Plotter(num=100)


@app.callback(
    Output("fig1", "figure"),
    [
        Input("dims", "value"),
        Input("distance", "value"),
        Input("scale", "value"),
        Input("dim_reducer", "value"),
        Input("loss", "value"),
        Input("temperature", "value"),
        Input("pos_margin", "value"),
        Input("neg_margin", "value"),
    ],
)
def output_fig(dims, distance, scale, dim_reducer, loss, temperature, pos_margin, neg_margin):
    """Update figure"""
    data = p.get_data(
        dims,
        distance,
        scale,
        {"pca": PCA, "umap": UMAP, "tsne": TSNE}[dim_reducer],
        {"lifted": GeneralisedLiftedStructureLoss, "snnl": SoftNearestNeighborLoss}[loss],
        temperature=10 ** temperature,
        pos_margin=pos_margin,
        neg_margin=neg_margin,
    )
    fig = px.scatter(
        data,
        x="x",
        y="y",
        color="loss",
        symbol="symbol",
        hover_name="distance",
        hover_data=["loss"],
        width=1000,
        height=800,
        symbol_sequence=["circle", "cross"],
        title="Loss = {}".format(data["loss"].mean()),
    )
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(showlegend=False)
    fig.update_coloraxes(showscale=False)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
