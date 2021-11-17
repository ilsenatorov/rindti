import random
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

df = pd.read_csv("data/fragments_embeddings.tsv", sep="\t")
fam_counts = defaultdict(int)
for fams in df["fam"]:
    for fam in fams.split(";"):
        fam_counts[fam] += 1
fam_counts = pd.Series(fam_counts).sort_values(ascending=False)
df["color"] = "grey"
df["symbol"] = "circle"

symbols = ["square", "diamond", "triangle-up", "x", "star", "hourglass"]

app.layout = html.Div(
    style={"display": "flex", "flex-direction": "row"},
    children=[
        dcc.Store(id="highlighted_prot"),  # store highlighted protein
        dcc.Store(id="sample_index"),  # store highlighted family
        html.Div(  # left side, visualising protein structure and info
            [dashbio.NglMoleculeViewer(id="molecule"), html.Table(id="prot_table", children=[])],
            style={"width": "30%"},
        ),
        html.Div([dcc.Graph(id="embedding-fig")]),  # center, embedding plot
        html.Div(  # right side, with all the sliders and dropdowns
            [
                html.H4("Opacity"),
                dcc.Slider(
                    id="opacity",
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.25,
                    marks={i / 100: str(i) + "%" for i in range(0, 100, 10)},
                ),
                html.H4("Sample"),
                dcc.Slider(
                    id="sample",
                    min=10000,
                    max=len(df),
                    step=1,
                    value=10000,
                    marks={i: str(i)[:-3] + "K" for i in range(10000, len(df) - 10000, 10000)},
                ),
                html.H4("Size"),
                dcc.Slider(id="size", min=1, max=20, step=1, value=10, marks={k: str(k) for k in range(1, 21)}),
                html.H4("Family Search"),
                dcc.Dropdown(
                    id="fam_search",
                    multi=True,
                    options=[{"label": x, "value": x} for x in fam_counts.index],
                    value=[],
                    placeholder="Pfam families",
                ),
                dcc.RadioItems(
                    id="fam_search_highlight",
                    options=[{"label": "Highlight", "value": "on"}, {"label": "Show Only", "value": "off"}],
                    value="on",
                    labelStyle={"display": "inline-block"},
                ),
                html.H4("Organism Search"),
                dcc.Dropdown(
                    id="organism_search",
                    options=[{"label": x, "value": x} for x in df["organism"].unique()],
                    placeholder="Organism",
                    multi=True,
                ),
                dcc.RadioItems(
                    id="organism_search_highlight",
                    options=[{"label": "Highlight", "value": "on"}, {"label": "Show Only", "value": "off"}],
                    value="on",
                    labelStyle={"display": "inline-block"},
                ),
                html.H4("UniProt ID Search"),
                dcc.Dropdown(
                    id="prot_search",
                    options=[{"label": x, "value": x} for x in df["id"].value_counts().index],
                    placeholder="UniProt ID",
                ),
                html.H4("Name search"),
                dcc.Input(id="name_search", placeholder="Keywords...", value=""),
                dcc.RadioItems(
                    id="name_search_highlight",
                    options=[{"label": "Highlight", "value": "on"}, {"label": "Show Only", "value": "off"}],
                    value="on",
                    labelStyle={"display": "inline-block"},
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


@app.callback(Output("sample_index", "data"), Input("sample", "value"))
def update_sample_idx(sample):
    """Save the sample index"""
    return random.sample(range(len(df)), sample)


@app.callback(Output("molecule", "data"), Output("molecule", "molStyles"), Input("highlighted_prot", "data"))
def plot_molecule(fam_highlight):
    """Get molecular visualisation on click"""
    if fam_highlight is None:
        raise PreventUpdate
    prot_id = fam_highlight
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
        Input("highlighted_prot", "data"),
        Input("fam_search", "value"),
        Input("fam_search_highlight", "value"),
        Input("organism_search", "value"),
        Input("organism_search_highlight", "value"),
        Input("name_search", "value"),
        Input("name_search_highlight", "value"),
        Input("size", "value"),
        Input("opacity", "value"),
        Input("sample_index", "data"),
        Input("embedding-fig", "relayoutData"),
    ],
)
def update_figure(
    highlighted_prot: str,
    fam_search: list,
    fam_highlight: str,
    organism_search: list,
    organism_highlight: str,
    name_search: str,
    name_highlight: str,
    size: int,
    opacity: float,
    sample: int,
    relayoutData: dict,
) -> go.Figure:
    """Update figure"""
    data = df.copy().iloc[sample]
    if highlighted_prot and highlighted_prot not in data["id"]:
        data.loc["highlight"] = df[df["id"] == highlighted_prot].iloc[0]
    data["size"] = size
    data["opacity"] = opacity
    if organism_search:  # highlight an organism
        for i, organism in enumerate(organism_search):
            data.loc[data["organism"] == organism, "color"] = px.colors.qualitative.Dark24[i]
        if organism_highlight == "off":
            data = data[data["organism"].isin(organism_search)]
    if fam_search:  # many families can be chosen
        for i, fam in enumerate(fam_search):
            data.loc[data["fam"].str.contains(fam), "color"] = px.colors.qualitative.Light24[i]
        if fam_highlight == "off":
            data = data[data["fam"].str.contains("|".join(fam_search))]
    if name_search:  # search by name
        if name_highlight == "on":
            data.loc[data["name"].str.contains(name_search, case=False), "color"] = "red"
        else:
            data = data[data["name"].str.contains(name_search, case=False)]
    if highlighted_prot:
        data.loc[data["id"] == highlighted_prot, "symbol"] = "star"
        data.loc[data["id"] == highlighted_prot, "size"] = size * 2.5
        data.loc[data["id"] == highlighted_prot, "opacity"] = 1
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=data["x"],
                y=data["y"],
                mode="markers",
                marker=dict(
                    size=data["size"],
                    opacity=data["opacity"],
                    color=data["color"],
                    symbol=data["symbol"],
                    line=dict(width=0),
                ),
                customdata=data[["id", "fam", "name"]].values,
                hovertemplate="<b>UniProt ID:</b> %{customdata[0]}<br>"
                + "<b>Pfam IDs:</b> %{customdata[1]}</b><br>"
                + "<b>Name:</b> %{customdata[2]}",
            )
        ]
    )
    fig.update_layout(
        width=800,
        height=800,
        margin=dict(l=5, r=5, t=5, b=5),
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
    table.append(
        html.Tr(
            pfams,
        )
    )
    return table


if __name__ == "__main__":
    app.run_server(debug=True)
