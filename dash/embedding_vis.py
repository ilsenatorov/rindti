import json

import dash_bio as dashbio
import dash_bio_utils.ngl_parser as ngl_parser
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__)

df = pd.read_csv("data/embeddings.tsv", sep="\t")
df["select"] = "grey"


app.layout = html.Div(
    style={"display": "flex", "flex-direction": "row"},
    children=[
        html.Div(
            [
                dashbio.NglMoleculeViewer(id="molecule"),
                html.A("UniProt Link", id="uniprot_link"),
            ]
        ),
        html.Div([dcc.Graph(id="fig1")]),
        html.Div(
            [
                html.H5("Opacity"),
                dcc.Slider(id="opacity", min=0, max=1, step=0.01, value=0.5),
                html.H5("Size"),
                dcc.Slider(id="size", min=1, max=20, step=1, value=8, marks={k: str(k) for k in range(1, 21)}),
                html.H5("Family Search"),
                dcc.Input(
                    id="fam_search",
                    type="text",
                    placeholder="Family name",
                ),
                html.H5("Protein Search"),
                dcc.Input(
                    id="prot_search",
                    type="text",
                    placeholder="Protein name",
                ),
            ],
            style={"padding": 10, "flex": 1},
        ),
    ],
)


@app.callback(Output("uniprot_link", "href"), Input("fig1", "clickData"))
def get_uniprot_link(clickData):
    """show link to uniprot"""
    if clickData is None:
        raise PreventUpdate
    prot_id, fams = clickData["points"][0]["customdata"]
    return f"https://www.uniprot.org/uniprot/{prot_id}"


@app.callback(Output("molecule", "data"), Output("molecule", "molStyles"), Input("fig1", "clickData"))
def return_molecule(prot_dict):
    """Get molecular visualisation on click"""
    if prot_dict is None:
        raise PreventUpdate
    prot_id = prot_dict["points"][0]["customdata"][0]

    molstyles_dict = {
        "representations": ["cartoon"],
        "chosenAtomsColor": "white",
        "chosenAtomsRadius": 1,
        "molSpacingXaxis": 100,
    }

    data_list = [
        ngl_parser.get_data(
            data_path="https://alphafold.ebi.ac.uk/files/",
            pdb_id=f"AF-{prot_id}-F1-model_v1",
            color="blue",
            reset_view=True,
            local=False,
        )
    ]
    return data_list, molstyles_dict


@app.callback(
    Output("fig1", "figure"),
    [
        Input("opacity", "value"),
        Input("fam_search", "value"),
        Input("prot_search", "value"),
        Input("size", "value"),
    ],
)
def output_fig(opacity, fam_search, prot_search, size):
    """Update figure"""
    data = df.copy()
    if fam_search:
        data.loc[data["fam"].str.contains(fam_search), "select"] = "red"
    if prot_search:
        data.loc[data["id"].str.contains(prot_search), "select"] = "blue"
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=data["x"],
                y=data["y"],
                mode="markers",
                marker=dict(size=size, opacity=opacity, color=data["select"]),
                customdata=data[["id", "fam"]].values,
                hovertemplate="UniProt ID: <b>%{customdata[0]}</b><br>Pfam IDs: <b>%{customdata[1]}</b>",
            )
        ]
    )
    fig.update_layout(
        title="Embeddings",
        xaxis_title="x",
        yaxis_title="y",
        width=800,
        height=800,
        clickmode="event+select",
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
