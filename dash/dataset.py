import os
import pickle
from typing import Iterable, Tuple

import dash_bio as dashbio
import dash_bio.utils.ngl_parser as ngl_parser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

import dash
from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate
from rindti.models.dti.baseline import ProtDrugMax

app = dash.Dash(__name__)

folder = "datasets/glass/results/summary"
files = [x for x in os.listdir(folder)]
columns = ["nnodes", "nedges", "plddt", "count", "label"]


@app.callback(
    Output("prot_struct", "figure"),
    Input("data", "data"),
    Input("x-col", "value"),
    Input("y-col", "value"),
    Input("color-col", "value"),
)
def plot_prot_structs(data: dict, x: str, y: str, color: str) -> go.Figure:
    """Plots protein structure information."""
    prots = pd.DataFrame(data["prots"])
    return px.scatter(
        prots,
        x=x,
        y=y,
        color=color,
        width=800,
        height=600,
        hover_name=prots.index,
    )


@app.callback(Output("baselines", "figure"), Input("data", "data"), Input("prob", "value"))
def plot_baselines(data: dict, prob: bool) -> go.Figure:
    """Plots the baselines."""
    inter = pd.DataFrame(data["inter"]).rename({"prot_id": "Target_ID", "drug_id": "Drug_ID", "label": "Y"}, axis=1)
    train = inter[inter["split"] == "train"]
    test = inter[inter["split"] == "test"]
    models = {k: ProtDrugMax(k, prob=prob) for k in ["prot", "drug", "both", "none"]}
    metrics = {k: v.assess_dataset(train, test) for k, v in models.items()}
    metrics = pd.DataFrame(metrics)
    metrics.to_csv("test.csv")
    return px.bar(metrics.T.round(3), barmode="group", text_auto=True, width=800, height=600)


@app.callback(
    Output("molecule", "data"),
    Output("molecule", "molStyles"),
    Input("prot_struct", "clickData"),
    Input("data", "data"),
)
def plot_molecule(prot_id: str, data: dict) -> Tuple[list, dict]:
    """Get molecular visualisation on click."""
    if not prot_id:
        raise PreventUpdate
    prot_id = prot_id["points"][0]["hovertext"]
    molstyles_dict = {
        "representations": ["cartoon"],
        "chosenAtomsColor": "white",
        "chosenAtomsRadius": 1,
        "molSpacingXaxis": 100,
    }
    data_list = [
        ngl_parser.get_data(
            data_path=f"file:{data['struct_dir']}/",
            pdb_id=prot_id,
            color="blue",
            reset_view=True,
            local=False,
        )
    ]
    return data_list, molstyles_dict


@app.callback(Output("data", "data"), Input("dropdown", "value"))
def read_data(filename: str) -> dict:
    """Read pickle file."""
    with open(os.path.join(folder, filename), "rb") as file:
        data = pickle.load(file)
    return {k: v.to_dict() if isinstance(v, pd.DataFrame) else v for k, v in data.items()}


@app.callback(Output("config", "children"), Input("data", "data"))
def get_config(data: dict) -> str:
    """Return config as yaml."""
    return yaml.dump({k: v for k, v in data["config"].items() if k != "data"})


flex_style = {"display": "flex", "flex-direction": "row", "justify-content": "left"}


def build_col_option(colname: str, col_id: str, value: str) -> html.Div:
    """Options for the prot plotter."""
    return html.Div(
        style=flex_style,
        children=[
            html.Div(colname + "     "),
            dcc.Dropdown(id=col_id, options=columns, value=value),
        ],
    )


app.layout = html.Div(
    children=[
        dcc.Store(id="data"),
        html.Div(
            [
                html.H4("Summary file"),
                dcc.Dropdown(id="dropdown", options=files, value=files[0], style={"width": "50%"}),
                html.Div(
                    children=[
                        build_col_option(colname, col_id, value)
                        for colname, col_id, value in [
                            ("X-axis", "x-col", "nnodes"),
                            ("Y-axis", "y-col", "nedges"),
                            ("Color", "color-col", "label"),
                        ]
                    ],
                ),
            ]
        ),
        html.Div(
            style={"display": "flex", "flex-direction": "row"},
            children=[
                dcc.Graph(id="prot_struct"),
                dashbio.NglMoleculeViewer(id="molecule"),
            ],
        ),
        html.Div(
            style={"display": "flex", "flex-direction": "row"},
            children=[
                dcc.Graph(id="baselines"),
                html.Plaintext(id="config"),
            ],
        ),
        html.Div(
            [
                html.H4("Probabilistic"),
                dcc.RadioItems(
                    id="prob",
                    options=[
                        {"label": "true", "value": True},
                        {"label": "false", "value": False},
                    ],
                    value=False,
                ),
            ]
        ),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
