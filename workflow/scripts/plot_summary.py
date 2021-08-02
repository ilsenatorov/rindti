import pickle

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_nnodes_nedges(df):
    df["nnodes"] = df["data"].apply(lambda x: x["x"].size(0))
    df["nedges"] = df["data"].apply(lambda x: x["edge_index"].size(1))
    return df


if __name__ == "__main__":

    with open(snakemake.input.pickle, "rb") as file:
        all_data = pickle.load(file)

    prot = all_data["prots"]
    drug = all_data["drugs"]
    inter = pd.DataFrame(all_data["data"])

    prot = calculate_nnodes_nedges(prot)
    drug = calculate_nnodes_nedges(drug)

    drug_agg = inter.groupby("drug_id").agg("mean")
    drug_agg["count"] = drug["count"]

    prot_agg = inter.groupby("prot_id").agg("mean")
    prot_agg["count"] = prot["count"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Prot nodes/edge distribution",
            "Drug node/edge distribution",
            "Prot label/count distribution",
            "Drug label/count distribution",
        ),
    )

    fig.add_trace(
        go.Scatter(x=prot["nnodes"], y=prot["nedges"], mode="markers", name="prots", text=prot.index), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=drug["nnodes"], y=drug["nedges"], mode="markers", name="drugs", text=drug.index), row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=prot_agg["label"], y=prot_agg["count"], mode="markers", name="drugs", text=prot_agg.index),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=drug_agg["label"], y=drug_agg["count"], mode="markers", name="drugs", text=drug_agg.index),
        row=2,
        col=2,
    )

    fig["layout"]["xaxis"]["title"] = "Number of nodes"
    fig["layout"]["yaxis"]["title"] = "Number of edges"
    fig["layout"]["xaxis2"]["title"] = "Number of nodes"
    fig["layout"]["yaxis2"]["title"] = "Number of edges"
    fig["layout"]["xaxis3"]["title"] = "Mean label"
    fig["layout"]["yaxis3"]["title"] = "Popularity"
    fig["layout"]["xaxis4"]["title"] = "Mean label"
    fig["layout"]["yaxis4"]["title"] = "Popularity"
    fig.update_layout(height=1080, width=1920, title_text="Data on", showlegend=False)
    fig.write_html(snakemake.output.html)
