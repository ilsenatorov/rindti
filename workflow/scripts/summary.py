import pickle

import numpy as np
import pandas as pd


def update_data(data: dict, plddt: pd.Series) -> dict:
    """Take original pickle and plddt file and combine them."""
    prots = data["prots"].join(plddt)
    prots["nnodes"] = prots["data"].apply(lambda x: x["x"].size(0))
    prots["nedges"] = prots["data"].apply(lambda x: x["edge_index"].size(1))
    inter = pd.DataFrame(data["data"])
    prots = prots.join(inter.groupby("prot_id").agg("mean"))
    prots["count"] = prots["data"].apply(lambda x: x["count"])
    return {"prots": prots.drop("data", axis=1), "config": data["config"], "inter": inter}


def get_plddt_scores(structs: list) -> pd.Series:
    """From pdb structures get the average plddt score."""
    plddt_scores = {}
    for structure in structs:
        calphas_plddt = []
        with open(structure, "r") as file:
            for line in file.readlines():
                if line.startswith("ATOM") and line[13:15] == "CA":
                    calphas_plddt.append(float(line[61:66]))
            structure_id = structure.split("/")[-1].split(".")[0]
            plddt_scores[structure_id] = np.mean(calphas_plddt)
    return pd.Series(plddt_scores, name="plddt").sort_values()


if __name__ == "__main__":
    plddt = get_plddt_scores(snakemake.input.structs)
    with open(snakemake.input.pickle, "rb") as file:
        data = pickle.load(file)
    new_data = update_data(data, plddt)
    new_data["struct_dir"] = snakemake.params.struct_dir
    with open(snakemake.output.summary, "wb") as file:
        pickle.dump(new_data, file)
