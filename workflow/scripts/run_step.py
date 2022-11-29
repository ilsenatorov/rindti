if "snakemake" in globals():
    all_structures = snakemake.input.pdbs

    prots = pd.Series(list(snakemake.input.rins), name="sif")
    prots = pd.DataFrame(prots)
    prots["ID"] = prots["sif"].apply(extract_name)
    prots.set_index("ID", inplace=True)
    prot_encoder = ProteinEncoder(snakemake.params.node_feats, snakemake.params.edge_feats)
    prots["data"] = prots["sif"].apply(prot_encoder)
    prots.to_pickle(snakemake.output.pickle)