import numpy as np
import pandas as pd

plddt_scores = {}
for structure in snakemake.input.structs:
    calphas_plddt = []
    with open(structure, "r") as file:
        for line in file.readlines():
            if line.startswith("ATOM") and line[13:15] == "CA":
                calphas_plddt.append(float(line[61:66]))
        structure_id = structure.split("/")[-1].split(".")[0]
        plddt_scores[structure_id] = np.mean(calphas_plddt)

plddt_scores = pd.Series(plddt_scores, name="plddt").sort_values()
plddt_scores.to_csv(snakemake.output.struct_info, sep="\t")
