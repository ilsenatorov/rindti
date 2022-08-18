import pickle
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from rindti.data import DTIDataModule
from rindti.data.transforms import NullTransformer
from rindti.models import MultitaskClassification
from rindti.utils import read_config


def run(model, dataloader, p_name, d_name):
    d, p = None, None
    for i, batch in enumerate(dataloader):
        print(f"\r{i}", end="")
        results = model.shared_step(batch)
        if p is None:
            p = results["prot_embed"]
        else:
            p = torch.cat((p, results["prot_embed"]), dim=0)
        if d is None:
            d = results["drug_embed"]
        else:
            d = torch.cat((d, results["drug_embed"]), dim=0)
    pickle.dump(np.unique(p.numpy(), axis=0), open(p_name, "wb"))
    pickle.dump(np.unique(d.numpy(), axis=0), open(d_name, "wb"))
    print()


def compute_embeddings(config):
    config = read_config(config)

    dti_datamodule = DTIDataModule(
        filename="/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle/results/prepare_all/rlnwgntanc_5e01134f.pkl",
        exp_name="glylec_mbb",
        batch_size=128,
        shuffle=False,
    )
    dti_datamodule.setup(transform=NullTransformer(**config["transform"]))
    dti_datamodule.update_config(config)
    ricin_datamodule = DTIDataModule(
        filename="/scratch/SCRATCH_SAS/roman/rindti/datasets/glylex/Ricin/results/prepare_all/rlnwgntanc_a40936e4.pkl",
        exp_name="glylec_mbb",
        batch_size=12,
        shuffle=False,
    )
    ricin_datamodule.setup(transform=NullTransformer(**config["transform"]), split="train")
    ricin_datamodule.update_config(config)

    model = MultitaskClassification(**config)
    model = model.load_from_checkpoint(
        "tb_logs/dti_glylec_mbb/rlnwgntanc_5e01134f/version_22/version_82/checkpoints/epoch=1348-step=232027.ckpt"
    )
    model.eval()

    run(model, dti_datamodule.train_dataloader(), "p_train.pkl", "d_train.pkl")
    run(model, dti_datamodule.val_dataloader(), "p_val.pkl", "d_val.pkl")
    run(model, ricin_datamodule.train_dataloader(), "p_ricin.pkl", "d_ricin.pkl")


def show(prefix, title, name):
    train_embeds = pickle.load(open(f"{prefix}_train.pkl", "rb"))
    val_embeds = pickle.load(open(f"{prefix}_val.pkl", "rb"))
    ricin_embeds = pickle.load(open(f"{prefix}_ricin.pkl", "rb"))

    embeds = np.concatenate((train_embeds, val_embeds, ricin_embeds), axis=0)
    embeds = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50).fit_transform(embeds)
    # embeds = PCA(n_components=2).fit_transform(embeds)

    t_x, t_y = zip(*(embeds[: len(train_embeds)]))
    v_x, v_y = zip(*(embeds[len(train_embeds) : (len(train_embeds) + len(val_embeds))]))
    r_x, r_y = zip(*(embeds[(len(train_embeds) + len(val_embeds)) :]))
    plt.scatter(t_x, t_y, color="g", marker="x", label="train")
    plt.scatter(v_x, v_y, color="r", marker="+", label="val")
    plt.scatter(r_x, r_y, color="b", marker="o", label="ricin")
    plt.title(title)
    plt.legend(loc=2)
    plt.savefig(name)
    plt.show()


# with torch.no_grad():
#     compute_embeddings("config/dti/glylec_lr.yaml")

show("p", "Proteins", "prots_tsne.png")
show("d", "Drugs", "drugs_tsne.png")

print("Done")
