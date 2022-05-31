import pickle

import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.subplots import make_subplots


def calculate_nnodes_nedges(df: DataFrame, esm=False) -> DataFrame:
    """Add columns for number of nodes and number of edges

    Args:
        df (DataFrame): [description]
        esm (bool): [description]

    Returns:
        DataFrame: [description]
    """
    df["nnodes"] = 0 if esm else df["data"].apply(lambda x: x["x"].size(0))
    df["nedges"] = 0 if esm else df["data"].apply(lambda x: x["edge_index"].size(1))
    return df


if __name__ == "__main__":

    with open(snakemake.input.pickle, "rb") as file:
        all_data = pickle.load(file)

    prot = all_data["prots"]
    drug = all_data["drugs"]
    inter = pd.DataFrame(all_data["data"])
    struct_info = pd.read_csv(snakemake.input.struct_info, sep="\t", index_col=0).squeeze("columns")

    prot = calculate_nnodes_nedges(prot, snakemake.config["prepare_prots"]["node_feats"] == "esm")
    prot["plddt"] = struct_info
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
            "Drug node distribution",
            "Prot label/count distribution",
            "Drug label/count distribution",
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=prot["nnodes"],
            y=prot["nedges"],
            mode="markers",
            name="prots",
            text=prot.index.to_series() + ", Mean pLDDT: " + prot["plddt"].round().astype(str),
            marker_color=prot["plddt"],
            marker=dict(colorbar=dict(title="mean pLDDT", len=0.45, y=0.8, x=0.45)),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=drug["nnodes"],
            name="drugs_hist",
            # text=drug.index,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=prot_agg["label"],
            y=prot_agg["count"],
            mode="markers",
            name="drugs",
            text=prot_agg.index,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=drug_agg["count"],
            name="drugs",
        ),
        row=2,
        col=2,
    )

    fig["layout"]["xaxis"]["title"] = "Number of nodes"
    fig["layout"]["yaxis"]["title"] = "Number of edges"
    fig["layout"]["xaxis2"]["title"] = "Number of nodes"
    fig["layout"]["yaxis2"]["title"] = "Count"
    fig["layout"]["xaxis3"]["title"] = "Mean label"
    fig["layout"]["yaxis3"]["title"] = "Popularity"
    fig["layout"]["xaxis4"]["title"] = "Drug popularity"
    fig["layout"]["yaxis4"]["title"] = "Count"
    fig.update_layout(
        height=900,
        width=1800,
        title={
            "text": "Data summary. Structure: {struct}. Filtering: {filt}.".format(
                struct=snakemake.config["structures"], filt=snakemake.config["parse_dataset"]["filtering"]
            ),
            "y": 1,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(
                size=24,
            ),
        },
        showlegend=False,
        margin=dict(t=70, b=0, l=0, r=0),
    )
    fig.write_html(snakemake.output.html)
