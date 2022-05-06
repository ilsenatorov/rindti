import dash_cytoscape as cyto
import pandas as pd

import dash
from dash import html

app = dash.Dash(__name__)


df = pd.read_csv("results/parse_dataset/posneg_none_inter.csv", sep="\t")
edges = []
drugs = []
prots = []
for i, row in df.iterrows():
    if row["Drug_ID"] not in drugs:
        drugs.append({"data": {"id": row["Drug_ID"]}, "classes": "drug"})
    if row["Target_ID"] not in prots:
        prots.append({"data": {"id": row["Target_ID"]}, "classes": "prot"})
    edges.append({"data": {"source": row["Drug_ID"], "target": row["Target_ID"]}})

print("Calculations done, loading the server!\n")

app.layout = html.Div(
    [
        cyto.Cytoscape(
            id="cytoscape",
            layout={"name": "cose"},
            style={"width": "1800px", "height": "900px"},
            elements=edges + drugs + prots,
            stylesheet=[
                # Group selectors
                {
                    "selector": "node",
                    "style": {
                        "content": "data(label)",
                        # 'width' : 20,
                        # 'height' : 10,
                    },
                },
                # Class selectors
                {"selector": ".prot", "style": {"background-color": "red", "line-color": "red"}},
                {"selector": ".drug", "style": {"background-color": "blue", "line-color": "blue"}},
            ],
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
