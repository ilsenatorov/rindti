import pickle

import pandas as pd
from utils import get_config

prot_table = pd.read_csv(snakemake.input.prot_table, sep="\t")
prot_data = pd.read_pickle(snakemake.input.prot_data)
prot_y = prot_table.set_index("Target_ID")["Y"].to_dict()

dims_config = get_config(prot_data, "prot")
dims_config["num_classes"] = len(prot_y)
snakemake.config["prots"]["data"] = dims_config

y_encoder = {v: k for k, v in enumerate(sorted(set(prot_y.values())))}

result = []
for k, v in prot_data["data"].items():
    v["y"] = y_encoder[prot_y[k]]
    v["id"] = k
    result.append(v)

with open(snakemake.output.pretrain_prot_data, "wb") as file:
    pickle.dump(
        {
            "data": result,
            "config": snakemake.config["prots"],
            "decoder": {v: k for k, v in y_encoder.items()},
        },
        file,
    )
