import os
import pickle
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html

app = dash.Dash(__name__)


def plot_prots(prot: pd.DataFrame) -> go.Figure:
    prot["nnodes"] = prot["data"].apply(lambda x: x["x"].size(0))
    prot["nedges"] = prot["data"].apply(lambda x: x["edge_index"].size(1))
    return px.scatter(
        x=prot["nnodes"],
        y=prot["nedges"],
        color=prot["plddt"],
        width=800,
        height=600,
        hover_name=prot.index,
    )


def file_options(folder: str) -> Iterable[str]:
    return [os.path.join(folder, i) for i in os.listdir(folder)]


with open("datasets/glass/results/prepare_all/tdlnpnclnr_9dc97485.pkl", "rb") as f:
    data = pickle.load(f)

plddt = pd.read_csv("datasets/glass/results/structure_info/t_c4e04594.tsv", sep="\t", index_col=0).squeeze("columns")
prot = data["prots"].join(plddt)

app.layout = html.Div(
    [
        dcc.Graph(figure=plot_prots(prot)),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
