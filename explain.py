from argparse import ArgumentParser

import torch
import copy

from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
import os
import yaml
from sklearn.manifold import TSNE

from torch_geometric.loader import DataLoader

from rindti.data import DTIDataModule, TwoGraphData

from rindti.utils import read_config, remove_arg_prefix
from train import models, transformers
import prettytable as pt
import scipy.stats as stats
import numpy as np


def concat_dataloaders(loaders):
    for loader in loaders:
        if loader.dataset is None:
            continue
        for x in loader:
            yield x


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


def clean_yaml(config: dict) -> dict:
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


def fit_dist(filename, **kwargs):
    print("Fit normal distributions of predictions on training data")
    for data in kwargs["models"]:
        print(f"\t{data['dataset']}")
        seed_everything(42)
        kwargs["model"]["drug"]["method"] = data["drug_method"]
        kwargs["model"]["prot"]["method"] = data["prot_method"]

        print("\tLoading the data...")
        datamodule = DTIDataModule(filename=data["dist_data"], exp_name="glylec_mbb", batch_size=128, shuffle=False, num_workers=16)
        datamodule.setup(transform=transformers[kwargs["transform"]["mode"]](**kwargs["transform"]), split="train")
        datamodule.update_config(kwargs)
        dataloader = datamodule.train_dataloader()
        print("\t\tDone")

        print("\tLoading the models...")
        model_list = []
        model_data = {}
        for m in data["models"]:
            if m["pos_params"][1] != -1 and m["neg_params"][1] != -1:
                continue
            kwargs["model"]["module"] = m["arch"]
            # model = models[kwargs["model"]["module"]](**kwargs)
            # model = model.load_from_checkpoint(m['checkpoint'])
            model = models[kwargs["model"]["module"]].load_from_checkpoint(m['checkpoint'])
            model.eval()
            model_list.append((m["name"], model))
            model_data[m["name"]] = [], []
        print("\r\t\tDone")

        print("\tCalculate distributions...")
        for i, batch in enumerate(dataloader):
            print(f"\r\tBatch: {i} / {len(dataloader)}", end="")
            if i >= 5:
                break
            for n, model in model_list:
                results = model.forward(remove_arg_prefix("prot_", batch), remove_arg_prefix("drug_", batch))
                model_data[n][0].append(torch.sigmoid(results["pred"][(batch["label"] == 0).squeeze(), :]))
                model_data[n][1].append(torch.sigmoid(results["pred"][(batch["label"] == 1).squeeze(), :]))
        print("\t\tDone")

        print("\tStore parameters...")
        for i, (n, (v_pos, v_neg)) in enumerate(model_data.items()):
            m_data_pos = torch.cat(v_pos).numpy()
            norm_loc_pos, norm_scale_pos = stats.norm.fit(np.array(m_data_pos))
            m_data_neg = torch.cat(v_neg).numpy()
            norm_loc_neg, norm_scale_neg = stats.norm.fit(np.array(m_data_neg))
            data["models"][i]["pos_params"] = [norm_loc_pos, norm_scale_pos]
            data["models"][i]["neg_params"] = [norm_loc_neg, norm_scale_neg]
        print("\t\tDone")

    config = clean_yaml(kwargs)
    with open(filename, "w") as out:
        yaml.dump(config, out)
    print("Finished")
    return config


def plot(output_dir, **kwargs):
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
    print("Finished")


def run_tsne(output_dir, **kwargs):
    print("Calculate tSNE plots for embeddings for all data")
    for data in kwargs["models"]:
        print(f"\t{data['dataset']}")
        seed_everything(42)
        kwargs["model"]["drug"]["method"] = data["drug_method"]
        kwargs["model"]["prot"]["method"] = data["prot_method"]

        print("\tLoading the data...")
        datamodule = DTIDataModule(filename=data["dist_data"], exp_name="glylec_mbb", batch_size=1, shuffle=False,
                                   num_workers=16)
        datamodule.setup(transform=transformers[kwargs["transform"]["mode"]](**kwargs["transform"]), split="train")
        datamodule.update_config(kwargs)
        dataloader = concat_dataloaders([
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            datamodule.test_dataloader(),
        ])
        print("\t\tDone")

        print("\tLoading the models...")
        model_list = []
        model_data = {}
        for m in data["models"]:
            kwargs["model"]["module"] = m["arch"]
            model = models[kwargs["model"]["module"]](**kwargs)
            model = model.load_from_checkpoint(m['checkpoint'])
            model.eval()
            model_list.append((m["name"], model))
            model_data[m["name"]] = [], [], [], []
        print("\t\tDone")

        print("\tCalculate embeddings...")
        seen_prots, seen_drugs = set(), set()
        for i, batch in enumerate(dataloader):
            print(f"\r\tBatch: {i}", end="")
            if batch["prot_id"][0] not in seen_prots:
                for name, model in model_list:
                    graph, _ = model.prot_encoder.forward(remove_arg_prefix("prot_", batch))
                    model_data[name][0].append(graph.cpu().numpy()[0])
                seen_prots.add(batch["prot_id"][0])
            if batch["drug_id"][0] not in seen_drugs:
                for name, model in model_list:
                    graph, _ = model.drug_encoder.forward(remove_arg_prefix("drug_", batch))
                    model_data[name][1].append(graph.cpu().numpy()[0])
                seen_drugs.add(batch["drug_id"][0])
            if i < 1_000:
                for name, model in model_list:
                    if batch["label"][0] == 0:
                        model_data[name][2].append(model.forward(remove_arg_prefix("prot_", batch), remove_arg_prefix("drug_", batch))["embed"].cpu().numpy()[0])
                    if batch["label"][0] == 1:
                        model_data[name][3].append(model.forward(remove_arg_prefix("prot_", batch), remove_arg_prefix("drug_", batch))["embed"].cpu().numpy()[0])
        print("\r\t\tDone")

        print("\tPlot tSNE embeddings")
        for name, (prot_embeds, drug_embeds, bind_embeds, non_bind_embeds) in model_data.items():
            prot_embeds = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50).fit_transform(np.array(prot_embeds))
            drug_embeds = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50).fit_transform(np.array(drug_embeds))
            embeds = bind_embeds + non_bind_embeds
            embeds = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50).fit_transform(np.array(embeds))
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].scatter(prot_embeds[:, 0], prot_embeds[:, 1], color="b", marker="o", s=10)
            ax[1].scatter(drug_embeds[:, 0], drug_embeds[:, 1], color="b", marker="o", s=10)
            ax[2].scatter(embeds[:len(bind_embeds), 0], embeds[:len(bind_embeds), 1], color="g", marker="o", label="binding", s=1)
            ax[2].scatter(embeds[len(bind_embeds):, 0], embeds[len(bind_embeds):, 1], color="r", marker="o", label="non-binding", s=1)
            fig.suptitle(f"tSNE Plots for Protein, Drug, and Combined Embeddings of Model {name}")
            plt.savefig(os.path.join(output_dir, f"tsne_prot_drug_{name}.png"))
        print("\t\tDone")
    print("Finished")


def explain(output_dir, **kwargs):
    print("Calculate predictions and p-values for prediction dataset")
    for data in kwargs["models"]:
        print(f"\t{data['dataset']}")
        seed_everything(42)
        kwargs["model"]["drug"]["method"] = data["drug_method"]
        kwargs["model"]["prot"]["method"] = data["prot_method"]

        print("\tLoading the data...")
        datamodule = DTIDataModule(filename=data["dataset"], exp_name="glylec_mbb", batch_size=1, shuffle=False)
        datamodule.setup(transform=transformers[kwargs["transform"]["mode"]](**kwargs["transform"]), split="train")
        datamodule.update_config(kwargs)
        dataloader = datamodule.train_dataloader()
        print("\t\tDone")

        print("\tLoading the models...")
        model_list = []
        for m in data["models"]:
            # for method in kwargs["methods"]:
            #     os.makedirs(f"{kwargs['output_dir']}/{m['name']}/{method}/", exist_ok=True)
            # os.makedirs(f"{kwargs['output_dir']}/{m['name']}/counts/", exist_ok=True)
            # os.makedirs(f"{kwargs['output_dir']}/{m['name']}/distance/", exist_ok=True)

            kwargs["model"]["module"] = m["arch"]
            model = models[kwargs["model"]["module"]](**kwargs)
            model = model.load_from_checkpoint(m['checkpoint'])
            model.eval()
            model_list.append((m['name'], model, m["neg_params"], m["pos_params"]))
        print("\t\tDone")

        print("\tCalculate predictions...")
        res_table = pt.PrettyTable()
        res_table.field_names = ["Sample", "Label"] + [n for n, _, _, _ in model_list]

        for i, x in enumerate(dataloader):
            p_line = [f"{x['prot_id'][0]}-{x['drug_id'][0]}", x["label"][0].item()]
            pos_line, neg_line = ["", "p+"], ["", "p-"]
            print("\tSample", i + 1, "|", x["prot_id"][0], "-", x["drug_id"][0], "| Label:", x["label"][0].item())
            # with open(kwargs["sequences"], "r") as dist_file:
            #     aa_seq, dists = [], []
            #     for line in dist_file.readlines()[1:]:
            #         line = line.strip().split("\t")
            #         if line[0][:4] == x["prot_id"][0]:
            #             aa_seq = line[1]
            #         dists.append(float(y.strip().split(" ")[1]) // 10)
            #
            # drug_data = remove_arg_prefix("drug_", x)
            # prot_data = remove_arg_prefix("prot_", x)
            #
            # for method in kwargs["methods"]:
            # print("\t\tMethod", method)
            #     comb_data = [combine(prot, drug_data, x) for prot in modify_protein(prot_data, method)]
            #     loader = DataLoader(comb_data, batch_size=len(comb_data))
            #     for name, model, (neg_loc, neg_scale), (pos_loc, pos_scale) in model_list:
            #         print("\t\t\tModel", name)
            #         results = model.shared_step(next(iter(loader)))
            #         results = model.shared_step(x)
            #         predictions = torch.sigmoid(results["preds"])
            #         print("\t\t\t\tAvg  :", torch.mean(predictions).item())
            #         print("\t\t\t\tPred:", predictions[0].item())
            #         neg_p_val = stats.norm.cdf(predictions[0].item(), loc=neg_loc, scale=neg_scale)
            #         pos_p_val = stats.norm.cdf(predictions[0].item(), loc=pos_loc, scale=pos_scale)
            #         print("\t\t\t\tneg. p-val:", neg_p_val)
            #         print("\t\t\t\tpos. p-val:", pos_p_val)
            #         p_line.append(predictions[0].item())
            #         neg_line.append(round(neg_p_val, 5))
            #         pos_line.append(round(1 - pos_p_val, 5))
            #         predictions = predictions[0].expand(len(predictions) - 1) -
            #                   torch.tensor(predictions[1:].squeeze())
            #         predictions = predictions.detach().numpy()

            #         aa_counter = {}
            #
            #         for aa, p in zip(aa_seq, list(predictions)):
            #             if aa not in aa_counter:
            #                 aa_counter[aa] = [0, 0]
            #             aa_counter[aa][0] += 1
            #             aa_counter[aa][1] += p

            #         with open(f"{kwargs['output_dir']}/{name}/counts/aa_count_{method}.tsv", "w") as out:
            #             for k in aa_counter.keys():
            #                 print(f"{k}\t{aa_counter[k][0]}\t{aa_counter[k][1]}", file=out)
            #         with open(f"{kwargs['output_dir']}/{name}/distance/aa_dists_{method}.tsv", "w") as out:
            #             for k in aa_dist.keys():
            #                 print(f"{k}\t{aa_dist[k][0]}\t{aa_dist[k][1]}", file=out)'''"""
            #         del predictions
            #         del results
            #     del loader
            #     del comb_data
            for name, model, (neg_loc, neg_scale), (pos_loc, pos_scale) in model_list:
                results = model.forward(remove_arg_prefix("prot_", x), remove_arg_prefix("drug_", x))
                predictions = torch.sigmoid(results["pred"])
                neg_p_val = stats.norm.cdf(predictions[0].item(), loc=neg_loc, scale=neg_scale)
                pos_p_val = stats.norm.cdf(predictions[0].item(), loc=pos_loc, scale=pos_scale)
                p_line.append(predictions[0].item())
                neg_line.append(round(neg_p_val, 5))
                pos_line.append(round(1-pos_p_val, 5))
            res_table.add_row(p_line)
            res_table.add_row(pos_line)
            res_table.add_row(neg_line)

        print(res_table.get_csv_string(), file=open(os.path.join(output_dir, "model_predictions.csv"), "w"))
        print(res_table.get_string(), file=open(os.path.join(output_dir, "model_predictions.txt"), "w"))
        print("Done")
    print("Finished")


def main():
    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = read_config(args.config)
    os.makedirs(config["output_dir"], exist_ok=True)

    with torch.no_grad():
        # config = fit_dist(args.config, **config)
        plot(**config)
        run_tsne(**config)
        # explain(**config)


if __name__ == '__main__':
    main()
