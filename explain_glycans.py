import copy
import pickle

import torch
import viewer
from pytorch_lightning import seed_everything
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
import torch.nn.functional as F

from rindti.data import DTIDataModule, TwoGraphData
from rindti.utils import read_config, get_git_hash, remove_arg_prefix
from train import models
import numpy as np
import pandas as pd
import os


def modify_protein(prot: dict, mode: str):
    """
    mode in ["obfuscate", "delete", or "skip"
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


def explain(ckpt, model_prefix, split, folder, mode, aa_counter, aa_dist, **kwargs):
    if not os.path.exists(f"exp/{mode}/{model_prefix}/"):
        os.makedirs(f"exp/{folder}/{mode}/{model_prefix}/", exist_ok=True)
        os.makedirs(f"exp/{folder}/counts/", exist_ok=True)
        os.makedirs(f"exp/{folder}/distance/", exist_ok=True)

    seed_everything(kwargs["seed"])
    datamodule = DTIDataModule(**kwargs["datamodule"])
    datamodule.setup(split=split)
    dataloader = datamodule.test_dataloader() if split == "test" else datamodule.train_dataloader()
    datamodule.update_config(kwargs)
    print("Data loaded")

    """pdb_ids = list()
    pdbs = set()
    data = pd.read_csv("datasets/raw/oracle/model_summary.tsv", sep="\t")

    for i, x in enumerate(datamodule.train_dataloader()):
        print(f"\rProtein {i}", end="")
        record = data[data["Input ID"] == x["prot_id"][0]]
        if record["Template PDB ID"].values[0] in pdbs:
            continue
        pdbs.add(record["Template PDB ID"].values[0])
        pdb_ids.append((record["Template PDB ID"].values[0], record["Sequence identity"].values[0],
                        record["Coverage"].values[0], x["prot_id"][0], record["Template chain"].values[0]))
    print()
    print("\n".join(
        f"{x[0]}_{x[4]} ({x[3]}):\tI: {x[1]:.4f}\tC: {x[2]:.4f}" for x in sorted(pdb_ids, key=lambda r: r[1] + r[2])))
    exit(0)"""

    model = models[kwargs["model"]["module"]](**kwargs)
    model = model.load_from_checkpoint(ckpt)
    model.eval()
    print("Model loaded")
    all_pred = []
    dist_map = {
        "Lec00259": "exp/structs/3llz_dist.txt",
        "Lec01242": "exp/structs/1jzn_dist.txt",
        "Lec01359": "exp/structs/1k12_dist.txt",
        "Lec00212": "exp/structs/4x5p_dist.txt",
    }
    pdb_map = {
        "Lec00259": "exp/structs/3llz.pdb",
        "Lec01242": "exp/structs/1jzn.pdb",
        "Lec01359": "exp/structs/1k12.pdb",
        "Lec00212": "exp/structs/4x5p.pdb",
    }
    seen = set()
    stack = {}

    for i, x in enumerate(dataloader):
        print(f"\r{i}", end="")
        if len(seen) == 2:
            break
        if x["prot_id"][0] not in pdb_map.keys() or x["prot_id"][0] in seen or x["drug_id"][0] != "Gly00015":  # 16, 19
            continue
        seen.add(x["prot_id"][0])
        print(f"\n{x['prot_id'][0]} ({pdb_map[x['prot_id'][0]].split('.')[0][-4:]}) + {x['drug_id'][0]}")
        drug_data = remove_arg_prefix("drug_", x)
        prot_data = remove_arg_prefix("prot_", x)
        data = [combine(prot, drug_data, x) for prot in modify_protein(prot_data, mode)]
        with open(dist_map[x["prot_id"][0]], "r") as dist_file:
            aas, dists = [], []
            for y in dist_file.readlines():
                aas.append(y.strip().split(" ")[0].split("_")[0])
                dists.append(float(y.strip().split(" ")[1]) // 10)
        loader = DataLoader(data, batch_size=len(data))
        results = model.shared_step(next(iter(loader)))
        predictions = F.sigmoid(results["preds"])
        predictions = predictions[0].expand(len(predictions) - 1) - torch.tensor(predictions[1:].squeeze())
        predictions = predictions.detach().numpy()
        for aa, d, p in zip(aas, dists, list(predictions)):
            if aa not in aa_counter:
                aa_counter[aa] = [0, 0]
            if d not in aa_dist:
                aa_dist[d] = [0, 0]

            aa_counter[aa][0] += 1

            # aa_counter[aa][1] += abs(p)
            aa_counter[aa][1] += p

            aa_dist[d][0] += 1

            # aa_dist[d][1] += abs(p)
            aa_dist[d][1] += p

        if x['prot_id'][0] not in stack:
            stack[x['prot_id'][0]] = [1, predictions]
        else:
            stack[x['prot_id'][0]][1] += predictions
            stack[x['prot_id'][0]][0] += 1

    for lectin in stack.keys():
        with open(dist_map[lectin], "r") as dist_file:
            dist = [float(x.strip().split(" ")[1]) for x in dist_file.readlines()]
        count, predictions = stack[lectin]
        predictions /= count
        np_dist = np.array(dist)
        print(f"Lectin: {lectin} ({pdb_map[lectin].split('.')[0][-4:]}) Count:", count)

        # print("PCC:", np.corrcoef(np_dist, np.abs(predictions))[1, 0])
        print("PCC:", np.corrcoef(np_dist, predictions)[1, 0])

        # viewer.view(pdb_map[lectin], f"exp/{mode}/{model_prefix}/", [abs(x) for x in predictions], dist)
        viewer.view(pdb_map[lectin], f"exp/{folder}/{mode}/{model_prefix}/", [x for x in predictions], dist)

        plt.hist(predictions)
        plt.savefig(f"exp/{folder}/{mode}/{model_prefix}/expl_o_zero_{pdb_map[lectin].split('.')[0][-4:]}.png")
        plt.clf()


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code

    aa_counter = {
        "obfuscate": {},
        "delete": {},
    }
    aa_dist = {
        "obfuscate": {},
        "delete": {},
    }

    for ckpt, folder in [
        # ("tb_logs/dti_glylec/rlnwgntanc_5e01134f/version_1/version_82/checkpoints/epoch=58-step=152396.ckpt", "Gly82"),
        # ("tb_logs/dti_glylec/rlnwlntanc_1dc421f8/version_1/version_82/checkpoints/epoch=81-step=211805.ckpt", "Lab82"),
        ("tb_logs/dti_glylec/rlnwlntanc_1dc421f8/version_1/version_0/checkpoints/epoch=43-step=113651.ckpt", "Lab82")
        # ("tb_logs/dti_glylec/rlnwIntanc_4e03bb8f/version_1/version_82/checkpoints/epoch=30-step=80072.ckpt", "Swe82"),
    ]:
        for split in ["train", "test"]:
            for mode in ["obfuscate", "delete"]:
                print(f"{folder} - {split} - {mode}")
                explain(ckpt, folder, split, "glycans", mode, aa_counter[mode], aa_dist[mode], **orig_config)

    with open(f"exp/glycans/counts/aa_count_obfuscate.tsv", "w") as out:
        for k in aa_counter["obfuscate"].keys():
            print(f"{k}\t{aa_counter['obfuscate'][k][0]}\t{aa_counter['obfuscate'][k][1]}", file=out)

    with open(f"exp/glycans/counts/aa_count_delete.tsv", "w") as out:
        for k in aa_counter['delete'].keys():
            print(f"{k}\t{aa_counter['delete'][k][0]}\t{aa_counter['delete'][k][1]}", file=out)

    with open(f"exp/glycans/distance/aa_dist_obfuscate.tsv", "w") as out:
        for k in aa_dist['obfuscate'].keys():
            print(f"{k}\t{aa_dist['obfuscate'][k][0]}\t{aa_dist['obfuscate'][k][1]}", file=out)

    with open(f"exp/glycans/distance/aa_dist_delete.tsv", "w") as out:
        for k in aa_dist['delete'].keys():
            print(f"{k}\t{aa_dist['delete'][k][0]}\t{aa_dist['delete'][k][1]}", file=out)


if __name__ == "__main__":
    main()
