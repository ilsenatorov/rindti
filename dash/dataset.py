import os
import pickle
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html

app = dash.Dash(__name__)


def plot_prot_structs(prots: pd.DataFrame) -> go.Figure:
    """Plots protein structure information."""
    return px.scatter(
        prots,
        x="nnodes",
        y="nedges",
        color="plddt",
        width=800,
        height=600,
        hover_name=prots.index,
    )


def plot_prot_dist(prots: pd.DataFrame) -> go.Figure:
    """Distribtion of proteins."""
    return px.scatter(prots, "label", "count", hover_name=prots.index, width=800, height=600)


def update_data(data: dict, plddt: pd.Series) -> dict:
    """Takes original pickle and plddt file and combines them"""
    prots = data["prots"].join(plddt)
    prots["nnodes"] = prots["data"].apply(lambda x: x["x"].size(0))
    prots["nedges"] = prots["data"].apply(lambda x: x["edge_index"].size(1))
    inter = pd.DataFrame(data["data"])
    prots = prots.join(inter.groupby("prot_id").agg("mean"))
    prots["count"] = prots["data"].apply(lambda x: x["count"])
    return {"prots": prots}


def file_options(folder: str) -> Iterable[str]:
    """All files in given folder"""
    return [os.path.join(folder, i) for i in os.listdir(folder)]


with open("datasets/glass/results/prepare_all/tdlnpnclnr_9dc97485.pkl", "rb") as f:
    data = pickle.load(f)

plddt = pd.read_csv("datasets/glass/results/structure_info/t_c4e04594.tsv", sep="\t", index_col=0).squeeze("columns")
data = update_data(data, plddt)

app.layout = html.Div(
    children=[dcc.Graph(figure=plot_prot_structs(data["prots"])), dcc.Graph(figure=plot_prot_dist(data["prots"]))],
)

if __name__ == "__main__":
    app.run_server(debug=True)
