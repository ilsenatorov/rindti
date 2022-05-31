import os
import pickle
from typing import Iterable, Tuple

import dash_bio as dashbio
import dash_bio.utils.ngl_parser as ngl_parser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate

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
    """Takes original pickle and plddt file and combines them."""
    prots = data["prots"].join(plddt)
    prots["nnodes"] = prots["data"].apply(lambda x: x["x"].size(0))
    prots["nedges"] = prots["data"].apply(lambda x: x["edge_index"].size(1))
    inter = pd.DataFrame(data["data"])
    prots = prots.join(inter.groupby("prot_id").agg("mean"))
    prots["count"] = prots["data"].apply(lambda x: x["count"])
    return {"prots": prots}


def file_options(folder: str) -> Iterable[str]:
    """All files in given folder."""
    return [os.path.join(folder, i) for i in os.listdir(folder)]


@app.callback(
    Output("highlighted_prot", "data"),
    Input("prot_struct", "clickData"),
    Input("prot_dist", "clickData"),
)
def highlight_prot(prot_struct: str, prot_dist: str) -> str:
    """Save the protein that is highlighted, clean other stores"""
    print(prot_struct, prot_dist)
    if prot_struct is None and prot_dist is None:
        raise PreventUpdate
    if prot_struct is not None:
        return prot_struct["points"][0]["hovertext"]
    if prot_dist is not None:
        return prot_dist["points"][0]["hovertext"]


@app.callback(Output("molecule", "data"), Output("molecule", "molStyles"), Input("highlighted_prot", "data"))
def plot_molecule(prot_id: str) -> Tuple[dict, dict]:
    """Get molecular visualisation on click."""
    if not prot_id:
        raise PreventUpdate
    molstyles_dict = {
        "representations": ["cartoon"],
        "chosenAtomsColor": "white",
        "chosenAtomsRadius": 1,
        "molSpacingXaxis": 100,
    }
    data_list = [
        ngl_parser.get_data(
            data_path="file:datasets/glass/results/parsed_structs/t_c4e04594/",
            pdb_id=prot_id,
            color="blue",
            reset_view=True,
            local=False,
        )
    ]
    return data_list, molstyles_dict


with open("datasets/glass/results/prepare_all/tdlnpnclnr_9dc97485.pkl", "rb") as f:
    data = pickle.load(f)

plddt = pd.read_csv("datasets/glass/results/structure_info/t_c4e04594.tsv", sep="\t", index_col=0).squeeze("columns")
data = update_data(data, plddt)

app.layout = html.Div(
    children=[
        dcc.Store(id="highlighted_prot"),
        dcc.Graph(id="prot_struct", figure=plot_prot_structs(data["prots"])),
        dcc.Graph(id="prot_dist", figure=plot_prot_dist(data["prots"])),
        dashbio.NglMoleculeViewer(id="molecule"),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
