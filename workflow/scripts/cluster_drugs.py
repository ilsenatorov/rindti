"""Cluster ligands by ECFP4/Tanimoto similarity into a Drug_ID -> cluster table."""

import pandas as pd
from cluster import cluster_drugs

if __name__ == "__main__":
    ligs = pd.read_csv(snakemake.input.smiles, sep="\t").set_index("Drug_ID")
    assignment = cluster_drugs(ligs["Drug"].to_dict(), cutoff=snakemake.params.cutoff)
    out = pd.Series(assignment, name="cluster").rename_axis("Drug_ID").sort_index()
    out.to_csv(snakemake.output.clusters, sep="\t")
