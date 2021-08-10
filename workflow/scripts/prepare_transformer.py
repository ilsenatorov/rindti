import pickle

import pandas as pd
from prepare_proteins import aa_encoding

with open(snakemake.input.prots, "rb") as file:
    prots = pickle.load(file)
index_mapping = {i: row["data"]["index_mapping"] for i, row in prots.iterrows()}

gnomad = pd.read_csv(snakemake.input.gnomad)
gnomad_dict = {}
for prot in gnomad["UniProt ID"].unique():
    subset = gnomad[gnomad["UniProt ID"] == prot]
    subset = subset[subset["mut_pos"].isin(index_mapping[prot].keys())]
    gnomad_dict[prot] = subset
gnomad = gnomad_dict

final_data = {
    "index_mapping": index_mapping,
    "gnomad": gnomad,
    "encoded_residues": aa_encoding,
}

with open(snakemake.output.transformer_pickle, "wb") as file:
    pickle.dump(final_data, file, protocol=-1)
