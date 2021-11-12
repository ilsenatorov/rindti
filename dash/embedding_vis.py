from collections import defaultdict

import dash_bio as dashbio
import dash_bio_utils.ngl_parser as ngl_parser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__)

df = pd.read_csv("data/embeddings.tsv", sep="\t").iloc[:1000]
fam_counts = defaultdict(int)
for fams in df["fam"]:
    for fam in fams.split(";"):
        fam_counts[fam] += 1
fam_counts = pd.Series(fam_counts).sort_values(ascending=False)
df["color"] = "grey"
df["symbol"] = "circle"


app.layout = html.Div(
    style={"display": "flex", "flex-direction": "row"},
    children=[
        html.Div(
            [
                dcc.Store(id="highlighted_prot"),
                dashbio.NglMoleculeViewer(id="molecule"),
                html.Table(id="prot_table", children=[]),
            ],
            style={"width": "30%"},
        ),
        html.Div([dcc.Graph(id="embedding-fig")]),
        html.Div(
            [
                html.H5("Opacity"),
                dcc.Slider(id="opacity", min=0, max=1, step=0.01, value=0.5),
                html.H5("Size"),
                dcc.Slider(id="size", min=1, max=20, step=1, value=8, marks={k: str(k) for k in range(1, 21)}),
                html.H5("Family Search"),
                dcc.Dropdown(
                    id="fam_search",
                    multi=True,
                    options=[{"label": x, "value": x} for x in fam_counts.index],
                    value=[],
                ),
                dcc.RadioItems(
                    id="fam_search_highlight",
                    options=[{"label": "Highlight", "value": "on"}, {"label": "Show Only", "value": "off"}],
                    value="on",
                    labelStyle={"display": "inline-block"},
                ),
                html.H5("UniProt ID Search"),
                dcc.Dropdown(
                    id="prot_search",
                    options=[{"label": x, "value": x} for x in df["id"].value_counts().index],
                    placeholder="UniProt ID",
                ),
                html.H5("Species Search"),
                dcc.Dropdown(
                    id="species_search",
                    options=[{"label": x, "value": x} for x in df["organism"].unique()],
                    placeholder="Species",
                ),
            ],
            style={"padding": 10, "flex": 1},
        ),
    ],
)


@app.callback(
    Output("highlighted_prot", "data"),
    Output("embedding-fig", "clickData"),
    Output("prot_search", "value"),
    Input("prot_search", "value"),
    Input("embedding-fig", "clickData"),
)
def highlight_prot(prot_search, clickData):
    """Save the protein that is highlighted, clean other stores"""
    if prot_search is None and clickData is None:
        raise PreventUpdate
    if prot_search is not None:
        return prot_search, None, None
    prot_search = clickData["points"][0]["customdata"][0]
    return prot_search, None, None


@app.callback(Output("molecule", "data"), Output("molecule", "molStyles"), Input("highlighted_prot", "data"))
def plot_molecule(highlight):
    """Get molecular visualisation on click"""
    if highlight is None:
        raise PreventUpdate
    prot_id = highlight

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
    Output("embedding-fig", "figure"),
    [
        Input("opacity", "value"),
        Input("fam_search", "value"),
        Input("fam_search_highlight", "value"),
        Input("highlighted_prot", "data"),
        Input("size", "value"),
        Input("species_search", "value"),
        Input("embedding-fig", "relayoutData"),
    ],
)
def update_figure(
    opacity: float,
    fam_search: list,
    highlight: str,
    highlighted_prot: str,
    size: int,
    species: str,
    relayoutData: dict,
) -> go.Figure:
    """Update figure"""
    data = df.copy()
    data["size"] = size
    data["opacity"] = opacity
    if species:
        data.loc[data["organism"] == species, "color"] = "teal"
        data.loc[data["organism"] == species, "symbol"] = "diamond"
        if highlight == "off":
            data = data[data["organism"] == species]

    if fam_search:
        for i, fam in enumerate(fam_search):
            data.loc[data["fam"] == fam, "color"] = px.colors.qualitative.G10[i]
        if highlight == "off":
            data = data[data["fam"].str.contains("|".join(fam_search))]
    if highlighted_prot:  # highlight clicked protein
        data.loc[data["id"] == highlighted_prot, "symbol"] = "star"
        data.loc[data["id"] == highlighted_prot, "size"] = size * 2.5
        data.loc[data["id"] == highlighted_prot, "opacity"] = 1
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=data["x"],
                y=data["y"],
                mode="markers",
                marker=dict(size=data["size"], opacity=data["opacity"], color=data["color"], symbol=data["symbol"]),
                customdata=data[["id", "fam"]].values,
                hovertemplate="UniProt ID: <b>%{customdata[0]}</b><br>Pfam IDs: <b>%{customdata[1]}</b>",
            )
        ]
    )
    fig.update_layout(
        title="Embeddings", xaxis_title="x", yaxis_title="y", width=800, height=800,
    )
    if relayoutData and "xaxis.range[0]" in relayoutData:
        fig.update_xaxes(range=[relayoutData["xaxis.range[0]"], relayoutData["xaxis.range[1]"]])
        fig.update_yaxes(range=[relayoutData["yaxis.range[0]"], relayoutData["yaxis.range[1]"]])
    return fig


@app.callback(Output("prot_table", "children"), Input("highlighted_prot", "data"))
def update_prot_table(highlighted_prot: str) -> list:
    """Creates table with info for a protein"""
    if highlighted_prot is None:
        return []
    series = df.set_index("id").loc[highlighted_prot]
    renames = {"organism": "Organism", "name": "Protein name"}
    table = [html.Tr([html.Th(renames[key]), value]) for key, value in series.items() if key in renames]
    table.append(
        html.Tr(
            [
                html.Th("UniProt ID"),
                html.A(series.name, href="https://www.uniprot.org/uniprot/{}".format(series.name)),
            ]
        )
    )
    pfams = [html.Th("Pfam IDs")]
    for x in series.fam.split(";"):
        pfams.append(html.A(x, href=f"https://pfam.xfam.org/family/{x}"))
        pfams.append(html.Br())
    table.append(html.Tr(pfams,))
    return table


if __name__ == "__main__":
    app.run_server(debug=True)
