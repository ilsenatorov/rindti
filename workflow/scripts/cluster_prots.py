"""Turn MMseqs2 output into a Target_ID -> cluster-representative table."""

import pandas as pd
from cluster import parse_mmseqs_clusters

if __name__ == "__main__":
    mapping = pd.read_csv(snakemake.input.mapping, sep="\t", index_col="surrogate")["Target_ID"].to_dict()
    assignment = parse_mmseqs_clusters(snakemake.input.clusters, mapping)
    out = pd.Series(assignment, name="cluster").rename_axis("Target_ID").sort_index()
    out.to_csv(snakemake.output.clusters, sep="\t")
