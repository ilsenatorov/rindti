import copy
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import prettytable as pt
import scipy.stats as stats
import torch
import wandb
import yaml
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

from rindti.data import DTIDataModule, TwoGraphData
from rindti.utils import read_config, remove_arg_prefix
from train import models, transformers


# run = wandb.init("glylec_dti")


def concat_dataloaders(loaders):
    """Concatenate dataloaders"""
    for loader in loaders:
        if loader.dataset is None:
            continue
        for x in loader:
            yield x


def sorted_dataloader(loader):
    return DataLoader(sorted([x for x in loader], key=lambda x: x["prot_id", "drug_id"]))


def modify_protein(prot: dict, mode: str):
    """
    mode in ["obfuscate", "delete", or "skip"]
    """
    for i in range(-1, len(prot["x"])):
        prot_data = copy.deepcopy(prot)
        if i != -1:
            if mode == "obfuscate":  # mask a single node (suppress all information spreading but keep message flows)
                prot_data["x"][i] = 0
            elif mode == "delete":  # delete a node completely
                prot_data["x"] = torch.cat([prot_data["x"][:i], prot_data["x"][(i + 1) :]])
                prot_data["batch"] = torch.cat([prot_data["batch"][:i], prot_data["batch"][(i + 1) :]])
                e_idx = prot_data["edge_index"].T
                del_indices = [j for j, e in enumerate(e_idx) if i in e]
                for d in reversed(del_indices):
                    e_idx = torch.cat([e_idx[:d], e_idx[(d + 1) :]])
                e_idx[e_idx > i] -= 1
                prot_data["edge_index"] = e_idx.T
            elif mode == "skip":  # delete a node and connect all neighbors with each other
                pass
            else:
                raise ValueError(f"Unknown mode of explainability investigation: {mode}")
        yield prot_data


def clean_yaml(config: dict) -> dict:
    """Clear types in a dictionary to be stored in a readable yaml file"""
    output = {}
    for k, v in config.items():
        if isinstance(v, dict):
            v = clean_yaml(v)
        elif isinstance(v, list) or isinstance(v, tuple):
            if isinstance(v[0], dict):
                v = [clean_yaml(val) for val in v]
            elif isinstance(v[0], str):
                v = v
            elif np.issubdtype(type(v[0]), np.integer):
                v = [int(val) for val in v]
            elif np.issubdtype(type(v[0]), np.floating):
                v = [float(val) for val in v]
        elif isinstance(v, np.ndarray):
            v = list(v)
        elif isinstance(v, str):
            v = v
        elif np.issubdtype(type(v), np.integer):
            v = int(v)
        elif np.issubdtype(type(v), np.floating):
            v = float(v)
        output[k] = v
    return output


def load_data(data, dataset_name, split, batch_size, **kwargs):
    """Load the specified data into a dataloader"""
    print("\tLoading the data...")
    datamodule = DTIDataModule(
        filename=data[dataset_name], exp_name="glylec_mbb", batch_size=batch_size, shuffle=False, num_workers=16
    )
    datamodule.setup(transform=transformers[kwargs["transform"]["mode"]](**kwargs["transform"]), split=split)
    print("\t\tDone")
    return datamodule


def load_models(data, filtering=lambda x: False):
    """Load the models and create a list to store data in"""
    print("\tLoading the models...")
    model_list = []
    model_data = {}
    for m in data["models"]:
        if filtering(m):
            continue
        artifact = run.use_artifact(m["checkpoint"], type="model")
        artifact_dir = artifact.download()
        model = models[m["arch"]].load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
        model.eval()
        model = model.cuda()
        model_list.append((m["name"], model, m["neg_params"], m["pos_params"]))
        model_data[m["name"]] = [], [], ([], []), ([], [])
    print("\r\t\tDone")
    return model_list, model_data


def fit_dist(filename, **kwargs):
    """Fit normal distributions to the predicted values of the model on a subset of the training data"""
    print("Fit normal distributions of predictions on training data")
    for data in kwargs["models"]:
        print(f"\t{data['dataset']}")
        seed_everything(42)

        datamodule = load_data(data, dataset_name="dist_data", split="train", batch_size=128, **kwargs)
        dataloader = datamodule.train_dataloader()

        model_list, model_data = load_models(
            data, filtering=lambda x: x["pos_params"][1] != -1 and x["neg_params"][1] != -1
        )

        print("\tCalculate distributions...")
        for i, batch in enumerate(dataloader):
            print(f"\r\tBatch: {i} / {len(dataloader)}", end="")
            if i >= 5:
                break
            for n, model, _, _ in model_list:
                batch = batch.cuda()
                results = model.forward(remove_arg_prefix("prot_", batch), remove_arg_prefix("drug_", batch))
                model_data[n][0].append(torch.sigmoid(results["pred"][(batch["label"] == 0).squeeze(), :]))
                model_data[n][1].append(torch.sigmoid(results["pred"][(batch["label"] == 1).squeeze(), :]))
        print("\t\tDone")

        print("\tStore parameters...")
        for i, (n, (v_pos, v_neg, _, _)) in enumerate(model_data.items()):
            m_data_pos = torch.cat(v_pos).cpu().numpy()
            norm_loc_pos, norm_scale_pos = stats.norm.fit(np.array(m_data_pos))
            m_data_neg = torch.cat(v_neg).cpu().numpy()
            norm_loc_neg, norm_scale_neg = stats.norm.fit(np.array(m_data_neg))
            data["models"][i]["pos_params"] = [norm_loc_pos, norm_scale_pos]
            data["models"][i]["neg_params"] = [norm_loc_neg, norm_scale_neg]
        print("\t\tDone")

    config = clean_yaml(kwargs)
    with open(filename, "w") as out:
        yaml.dump(config, out)
    print("Finished\n")
    return config


def plot(output_dir, **kwargs):
    """Plot the distributions estimated in fit_dist()"""
    print("Plot distributions of predictions")
    x = np.linspace(0, 1, 100)
    for data in kwargs["models"]:
        for i, m in enumerate(data["models"]):
            (neg_loc, neg_scale), (pos_loc, pos_scale) = m["neg_params"], m["pos_params"]
            plt.plot(x, stats.norm.pdf(x, pos_loc, pos_scale), color="g", label="binding")
            plt.plot(x, stats.norm.pdf(x, neg_loc, neg_scale), color="r", label="non binding")
            plt.title(m["name"])
            plt.xlabel("classification value")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"dist_pred_{m['name']}.png"))
            plt.clf()
    print("Finished\n")


def run_tsne(output_dir, **kwargs):
    """Create some tSNE plots for protein, drug and prediction embeddings"""
    print("Calculate tSNE plots for embeddings for all data")
    for data in kwargs["models"]:
        print(f"\t{data['dataset']}")
        seed_everything(42)

        datamodule = load_data(data, dataset_name="dist_data", split=None, batch_size=1, **kwargs)
        dataloader = concat_dataloaders(
            [
                datamodule.train_dataloader(),
                datamodule.val_dataloader(),
                datamodule.test_dataloader(),
            ]
        )

        model_list, model_data = load_models(data)

        print("\tCalculate embeddings...")
        seen_prots, seen_drugs = set(), set()
        for i, batch in enumerate(dataloader):
            batch = batch.cuda()
            print(f"\r\tBatch: {i}", end="")
            if batch["prot_id"][0] not in seen_prots:
                for name, model, _, _ in model_list:
                    graph, _ = model.prot_encoder.forward(remove_arg_prefix("prot_", batch))
                    model_data[name][0].append(graph.cpu().numpy()[0])
                seen_prots.add(batch["prot_id"][0])
            if batch["drug_id"][0] not in seen_drugs:
                for name, model, _, _ in model_list:
                    graph, _ = model.drug_encoder.forward(remove_arg_prefix("drug_", batch))
                    model_data[name][1].append(graph.cpu().numpy()[0])
                seen_drugs.add(batch["drug_id"][0])
            if i < 1_000:
                for name, model, _, _ in model_list:
                    pred = model.forward(remove_arg_prefix("prot_", batch), remove_arg_prefix("drug_", batch))
                    if batch["label"][0] == 0:  # true binding
                        model_data[name][2][0 if pred["pred"] <= 0 else 1].append(pred["embed"].cpu().numpy()[0])
                    if batch["label"][0] == 1:  # true non-binding
                        model_data[name][3][0 if pred["pred"] <= 0 else 1].append(pred["embed"].cpu().numpy()[0])
        print("\r\t\tDone")

        print("\tPlot tSNE embeddings")
        for name, (prot_embeds, drug_embeds, (tp_be, fn_be), (fp_nbe, tn_nbe)) in model_data.items():
            prot_embeds = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50).fit_transform(
                np.array(prot_embeds)
            )
            drug_embeds = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50).fit_transform(
                np.array(drug_embeds)
            )
            embeds = tp_be + fn_be + fp_nbe + tn_nbe
            embeds = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50).fit_transform(
                np.array(embeds)
            )
            fig, ax = plt.subplots(1, 2, figsize=(12, 8))
            ax[0].scatter(prot_embeds[:, 0], prot_embeds[:, 1], color="b", marker="o", s=10)
            ax[1].scatter(drug_embeds[:, 0], drug_embeds[:, 1], color="b", marker="o", s=10)
            fig.suptitle(f"tSNE Plots for Protein and Drug Embeddings of Model {name}")
            plt.savefig(os.path.join(output_dir, f"tsne_prot_drug_{name}.png"))
            plt.clf()
            plt.scatter(embeds[: len(tp_be), 0], embeds[: len(tp_be), 1], color="g", marker="o", label="tp", s=20)
            plt.scatter(
                embeds[len(tp_be) : len(tp_be) + len(fn_be), 0],
                embeds[len(tp_be) : len(tp_be) + len(fn_be), 1],
                color="r",
                marker="o",
                label="fn",
                s=20,
            )
            plt.scatter(
                embeds[len(tp_be) + len(fn_be) : -len(tn_nbe), 0],
                embeds[len(tp_be) + len(fn_be) : -len(tn_nbe), 1],
                color="r",
                marker="s",
                label="fp",
                s=20,
            )
            plt.scatter(embeds[-len(tn_nbe) :, 0], embeds[-len(tn_nbe) :, 1], color="g", marker="s", label="tn", s=20)
            plt.suptitle(f"tSNE Plot for Combined Embeddings of Model {name}")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"tsne_pred_{name}.png"))
            plt.clf()
        print("\t\tDone")
    print("Finished\n")


def explain(output_dir, **kwargs):
    """Calculate predictions and p-values for some handcrafted dataset"""
    print("Calculate predictions and p-values for prediction dataset")
    for data in kwargs["models"]:
        print(f"\t{data['dataset']}")
        seed_everything(42)

        exp_datamodule = load_data(data, dataset_name="dataset", split="train", batch_size=1, **kwargs)
        exp_dataloader = exp_datamodule.train_dataloader()

        train_datamodule = load_data(data, dataset_name="dist_data", split="train", batch_size=1, **kwargs)
        train_dataloader = train_datamodule.train_dataloader()
        # train_dataloader = sorted_dataloader(train_datamodule.train_dataloader())

        model_list, _ = load_models(data)

        print("\tCalculate predictions...")
        res_table = pt.PrettyTable()
        res_table.field_names = ["Sample", "Label"] + [n for n, _, _, _ in model_list]
        for j, (set_name, loader) in enumerate([("Expl.", exp_dataloader), ("Train", train_dataloader)]):
            res_table.add_row([set_name, "Dataset"] + ["" for _ in model_list])

            for i, x in enumerate(loader):
                if j == 1 and i >= kwargs["control_count"]:
                    break
                p_line = [f"{x['prot_id'][0]}-{x['drug_id'][0]}", x["label"][0].item()]
                pos_line, neg_line = ["", "p+"], ["", "p-"]
                print("\tSample", i + 1, "|", x["prot_id"][0], "-", x["drug_id"][0], "| Label:", x["label"][0].item())
                for name, model, (neg_loc, neg_scale), (pos_loc, pos_scale) in model_list:
                    x = x.cuda()
                    results = model.forward(remove_arg_prefix("prot_", x), remove_arg_prefix("drug_", x))
                    predictions = torch.sigmoid(results["pred"])
                    neg_p_val = stats.norm.cdf(predictions[0].item(), loc=neg_loc, scale=neg_scale)
                    pos_p_val = stats.norm.cdf(predictions[0].item(), loc=pos_loc, scale=pos_scale)
                    p_line.append(predictions[0].item())
                    neg_line.append(round(neg_p_val, 5))
                    pos_line.append(round(1 - pos_p_val, 5))
                res_table.add_row(p_line)
                res_table.add_row(pos_line)
                res_table.add_row(neg_line)

        print(res_table.get_csv_string(), file=open(os.path.join(output_dir, f"{data['name']}_model_predictions.csv"), "w"))
        print(res_table.get_string(), file=open(os.path.join(output_dir, f"{data['name']}_model_predictions.txt"), "w"))
        print("Done")
    print("Finished\n")


def main():
    """Run the main routine"""
    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = read_config(args.config)
    os.makedirs(config["output_dir"], exist_ok=True)

    with torch.no_grad():
        config = fit_dist(args.config, **config)
        if "plot" in config["steps"]:
            plot(**config)
        if "tsne" in config["steps"]:
            run_tsne(**config)
        if "explain" in config["steps"]:
            explain(**config)


def test():
    gnn_datamodule = DTIDataModule(
        filename="/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle_full/results/prepare_all/rlnwgnrbnc_fe97f132.pkl",
        exp_name="glylec_mbb",
        batch_size=1,
        shuffle=False,
        num_workers=16
    )
    gnn_datamodule.setup(transform=transformers["none"](), split="train")
    gnn_loader = gnn_datamodule.train_dataloader()
    for i, x in enumerate(gnn_loader):
        if i > 5:
            break
        print(x["prot_id"], "<>", x["drug_id"])

    lo_datamodule = DTIDataModule(
        filename="/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle_esm/results/prepare_all/elnwInrbnc_8b492165.pkl",
        exp_name="glylec_mbb",
        batch_size=1,
        shuffle=False,
        num_workers=16
    )
    lo_datamodule.setup(transform=transformers["none"](), split="train")
    lo_loader = lo_datamodule.train_dataloader()
    for i, x in enumerate(lo_loader):
        if i > 5:
            break
        print(x["prot_id"], "<>", x["drug_id"])


if __name__ == "__main__":
    test()
    # main()
