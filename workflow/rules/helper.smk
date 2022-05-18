import os
import os.path as osp
import hashlib
import json
import pandas as pd


def flatten_config(config: dict) -> dict:
    """Flatten a config dictionary."""
    return pd.json_normalize(config).T[0].to_dict()


class Namer:
    def __init__(self, cutoff: int = None):
        self.cutoff = cutoff

    def hash_config(self, config: dict) -> str:
        """Hash a config dictionary."""
        as_json = json.dumps(config, sort_keys=True).encode("utf-8")
        return hashlib.md5(as_json).hexdigest()[: self.cutoff]

    def get_name(self, config: dict) -> str:
        """Get the name of a config."""
        flat = flatten_config(config)
        res = ""
        for k, v in flat.items():
            if k in ["source", "target"]:
                continue
            elif isinstance(v, str):
                res += v[0]
        return res + "_" + self.hash_config(config)

    def __call__(self, config: dict) -> str:
        return self.get_name(config)


class SnakemakeHelper:
    def __init__(self, config: dict, namer_cutoff: int = None):
        self.namer = Namer(namer_cutoff)
        self.source = config["source"]
        self.config = config
        self.target = "/".join(self.source.split("/")[:-1] + ["results"])
        self.prot_ids = [x.split(".")[0] for x in os.listdir(self.source + "/structures") if x.endswith(".pdb")]
        self.raw_structs = [os.path.join(self.source, "structures", x + ".pdb") for x in self.prot_ids]

    def _source(self, *args) -> str:
        return os.path.join(self.source, *args)

    def _target(self, *args) -> str:
        return os.path.join(self.target, *args)


# rinerator_output = f"{target}/rinerator_{config['structures']}/{{protein}}/{{protein}}_h.sif"
# rinerator_protein_output = f"{target}/rinerator_based/protein_data_{config['structures']}_{prot_settings}.pkl"

# distance_protein_output = f"{target}/distance_based/protein_data_{config['structures']}_{prot_settings}.pkl"
# if config["graph"] == "distance":
#     protein_output = distance_protein_output
# else:
#     protein_output = rinerator_protein_output

# if config["only_prots"]:  # Only calculate the data for the prots
#     output = [protein_output]
#     drug_output = ""
#     final_output = ""
#     transformer_output = ""
#     plot_output = ""
# else:

#     drug_output = target + f"/prepare_drugs/drug_data_{drug_settings}.pkl"

#     split_data = (
#         "{target}/split_data/{split}_{filtering}_{sampling}_{task}_split.csv".format(
#             target=target,
#             split=config["split"]["method"],
#             filtering=config["parse_dataset"]["filtering"],
#             sampling=config["parse_dataset"]["sampling"],
#             task=config["parse_dataset"]["task"],
#         )
#         if not config["only_prots"]
#         else ""
#     )

#     transformer_output = "{target}/prepare_transformer/{node_feats}_transformer.pkl".format(
#         target=target, node_feats=config["prepare_prots"]["node_feats"]
#     )

#     final_output = "{target}/prepare_all/{split}_{filtering}_{sampling}_{task}_{structures}_{graph}_{prot_settings}_{drug_settings}.pkl".format(
#         target=target,
#         split=config["split"]["method"],
#         filtering=config["parse_dataset"]["filtering"],
#         sampling=config["parse_dataset"]["sampling"],
#         task=config["parse_dataset"]["task"],
#         structures=config["structures"],
#         graph=config["graph"],
#         prot_settings=prot_settings,
#         drug_settings=drug_settings,
#     )

#     plot_output = final_output.split("/")[-1].split(".")[0]
#     plot_output = f"{target}/report/plot_summary/{plot_output}.html"

#     output = [final_output, plot_output]


# ### CHECK IF templates ARE PRESENT ###
# if osp.isdir(os.path.join(source, "templates")):
#     templates = expand(
#         "{resources}/templates/{template}",
#         resources=source,
#         template=os.listdir(f"{source}/templates"),
#     )
# else:
#     if not config["only_prots"] and config["structures"] not in ["whole", "plddt"]:
#         raise ValueError("No templates available")
#     templates = []
# ### CHECK IF gnomad is available ###
# if osp.isfile(os.path.join(source, "gnomad.csv")):
#     gnomad = source + "/gnomad.csv"
#     output.append(transformer_output)
# else:
#     gnomad = []
# ### CHECK IF drug data is available ###
# if osp.isdir(os.path.join(source, "drugs")):
#     drugs = {x: source + "/drugs/{x}.tsv".format(x=x) for x in ["inter", "lig", "prots"]}
# else:
#     if not config["only_prots"]:
#         raise ValueError("No drug interaction data available, can't calculate final data!")
#     drugs = {x: [] for x in ["inter", "lig"]}
