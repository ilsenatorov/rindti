from argparse import ArgumentParser

import torch
import copy
from pytorch_lightning import seed_everything
import os

from torch_geometric.loader import DataLoader

from rindti.data import DTIDataModule, TwoGraphData

from rindti.utils import read_config, remove_arg_prefix
from train import models
import prettytable as pt
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


def modify_protein(prot: dict, mode: str):
    """
    mode in ["obfuscate", "delete", or "skip"]
    """
    for i in range(-1, len(prot["x"])):
        prot_data = copy.deepcopy(prot)
        if i != -1:
            if mode == "obfuscate":
                prot_data["x"][i] = 0
            elif mode == "delete":
                prot_data["x"] = torch.cat([prot_data["x"][:i], prot_data["x"][(i + 1):]])
                prot_data["batch"] = torch.cat([prot_data["batch"][:i], prot_data["batch"][(i + 1):]])
                e_idx = prot_data["edge_index"].T
                del_indices = [j for j, e in enumerate(e_idx) if i in e]
                for d in reversed(del_indices):
                    e_idx = torch.cat([e_idx[:d], e_idx[(d + 1):]])
                e_idx[e_idx > i] -= 1
                prot_data["edge_index"] = e_idx.T
            elif mode == "skip":
                pass
            else:
                raise ValueError(f"Unknown mode of explainability investigation: {mode}")
        yield prot_data


def combine(prot, drug, x):
    comb_dict = {}
    for k in prot.keys():
        comb_dict["prot_" + k] = prot[k]
    for k in drug.keys():
        comb_dict["drug_" + k] = drug[k]
    for k, v in x.items():
        if not k.startswith("drug") and not k.startswith("prot"):
            comb_dict[k] = x[k]
    if "drug_IUPAC" in comb_dict:
        comb_dict["drug_IUPAC"] = comb_dict["drug_IUPAC"][0]
    return TwoGraphData(**comb_dict)


def fit_dist(**kwargs):
    for data in kwargs["models"]:
        seed_everything(42)
        kwargs["model"]["drug"]["method"] = data["drug_method"]
        kwargs["model"]["prot"]["method"] = data["prot_method"]

        datamodule = DTIDataModule(filename=data["dataset"], exp_name="expl", batch_size=128, shuffle=False, num_workers=16)
        datamodule.setup(split="train")
        datamodule.update_config(kwargs)
        dataloader = datamodule.train_dataloader()
        print("Data loaded")

        model_list = []
        model_data = {}
        for m in data["models"]:
            for method in kwargs["methods"]:
                os.makedirs(f"{kwargs['output_dir']}/{m['name']}/{method}/", exist_ok=True)
            os.makedirs(f"{kwargs['output_dir']}/{m['name']}/counts/", exist_ok=True)
            os.makedirs(f"{kwargs['output_dir']}/{m['name']}/distance/", exist_ok=True)

            model = models[kwargs["model"]["module"]](**kwargs)
            model = model.load_from_checkpoint(m['checkpoint'])
            model.eval()
            model_list.append((m['name'], model))
            model_data[m["name"]] = []
        print("Models loaded")

        for i, batch in enumerate(dataloader):
            print(f"\r{i} / {len(dataloader)}", end="")
            if i >= 5:
                break
            for n, model in model_list:
                results = model.shared_step(batch)
                model_data[n].append(torch.sigmoid(results["preds"][(results["labels"] == 1).squeeze(), :]))
        print()
        for n, v in model_data.items():
            m_data = torch.cat(v).numpy()
            norm_loc, norm_scale = stats.norm.fit(np.array(m_data))
            print(n, ":\n\tLoc:", norm_loc, "\tScale:", norm_scale)


def explain(**kwargs):
    for data in kwargs["models"]:
        seed_everything(42)
        kwargs["model"]["drug"]["method"] = data["drug_method"]
        kwargs["model"]["prot"]["method"] = data["prot_method"]

        datamodule = DTIDataModule(filename=data["dataset"], exp_name="expl", batch_size=1, shuffle=False)
        datamodule.setup(split="train")
        datamodule.update_config(kwargs)
        dataloader = datamodule.train_dataloader()
        print("Data loaded")

        model_list = []
        for m in data["models"]:
            for method in kwargs["methods"]:
                os.makedirs(f"{kwargs['output_dir']}/{m['name']}/{method}/", exist_ok=True)
            os.makedirs(f"{kwargs['output_dir']}/{m['name']}/counts/", exist_ok=True)
            os.makedirs(f"{kwargs['output_dir']}/{m['name']}/distance/", exist_ok=True)

            model = models[kwargs["model"]["module"]](**kwargs)
            model = model.load_from_checkpoint(m['checkpoint'])
            model.eval()
            model_list.append((m['name'], model, m["params"]))
        print("Models loaded")

        res_table = pt.PrettyTable()
        res_table.field_names = ["Sample", "Label"] + [n for n, _, _ in model_list]

        for i, x in enumerate(dataloader):
            res_line = [f"{x['prot_id'][0]}-{x['drug_id'][0]}", x["label"][0].item()]
            print("\tSample", i + 1, "|", x["prot_id"][0], "-", x["drug_id"][0], "| Label:", x["label"][0].item())
            with open(kwargs["sequences"], "r") as dist_file:
                aa_seq, dists = [], []
                for line in dist_file.readlines()[1:]:
                    line = line.strip().split("\t")
                    if line[0][:4] == x["prot_id"][0]:
                        aa_seq = line[1]
                    # dists.append(float(y.strip().split(" ")[1]) // 10)

            drug_data = remove_arg_prefix("drug_", x)
            prot_data = remove_arg_prefix("prot_", x)

            for method in kwargs["methods"]:
                print("\t\tMethod", method)
                comb_data = [combine(prot, drug_data, x) for prot in modify_protein(prot_data, method)]
                loader = DataLoader(comb_data, batch_size=len(comb_data))
                for name, model, (loc, scale) in model_list:
                    print("\t\t\tModel", name)
                    results = model.shared_step(next(iter(loader)))
                    predictions = torch.sigmoid(results["preds"])
                    print("\t\t\t\tAvg  :", torch.mean(predictions).item())
                    print("\t\t\t\tFirst:", predictions[0].item())
                    p_val = stats.norm.cdf(predictions[0].item(), loc=loc, scale=scale)
                    print("\t\t\t\tp-val:", p_val)
                    res_line.append(round(p_val, 5))
                    predictions = predictions[0].expand(len(predictions) - 1) - torch.tensor(predictions[1:].squeeze())
                    predictions = predictions.detach().numpy()

                    aa_counter = {}

                    for aa, p in zip(aa_seq, list(predictions)):
                        if aa not in aa_counter:
                            aa_counter[aa] = [0, 0]
    
                        aa_counter[aa][0] += 1
                        aa_counter[aa][1] += p

                    with open(f"{kwargs['output_dir']}/{name}/counts/aa_count_{method}.tsv", "w") as out:
                        for k in aa_counter.keys():
                            print(f"{k}\t{aa_counter[k][0]}\t{aa_counter[k][1]}", file=out)
                    """'''with open(f"{kwargs['output_dir']}/{name}/distance/aa_dists_{method}.tsv", "w") as out:
                        for k in aa_dist.keys():
                            print(f"{k}\t{aa_dist[k][0]}\t{aa_dist[k][1]}", file=out)'''"""
                    del predictions
                    del results
                del loader
                del comb_data
            del prot_data
            del drug_data
            res_table.add_row(res_line)
        print("\n".join(f"\t{line.strip()}" for line in res_table.get_string().split("\n")))
        exit(0)


def main():
    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    config = read_config(args.config)
    explain(**config)
    # fit_dist(**config)


if __name__ == '__main__':
    main()
