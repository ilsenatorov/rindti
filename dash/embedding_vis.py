import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)


app.layout = html.Div(
    style={"display": "flex", "flex-direction": "row"},
    children=[
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

df = pd.read_csv("data/embeddings.tsv", sep="\t").iloc[:1000]
df["select"] = "grey"


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
        data.loc[data["id"].str.contains(prot_search), "select"] = "red"
    print(data)
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
        width=1000,
        height=1000,
        clickmode="event+select",
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
