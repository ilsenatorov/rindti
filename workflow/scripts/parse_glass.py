import pickle

import numpy as np
import pandas as pd
from rdkit import Chem

inter = pd.read_csv(snakemake.input.inter, sep="\t")
lig = pd.read_csv(snakemake.input.lig)
targ = pd.read_csv(snakemake.input.targ, sep="\t")
lig.drop_duplicates("InChI Key", inplace=True)


def process_value(entry):
    try:
        return float(entry)
    except Exception as e:
        return np.nan


def get_numatoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms()


counts = {"inter": {}, "lig": {}, "targ": {}}


def update_counts(operation):
    counts["inter"][operation] = inter.shape[0]
    counts["lig"][operation] = len(inter["InChI Key"].unique())
    counts["targ"][operation] = len(inter["UniProt ID"].unique())


update_counts("GLASS dataset")

### First apply filters that are always applied ###

inter = inter[inter["Unit"] == "nM"]
update_counts("Activity units")

inter["Value"] = inter["Value"].apply(process_value)
inter = inter[inter["Value"].notna()]
update_counts("Numeric activity value")

inter["comb"] = inter["InChI Key"] + "_" + inter["UniProt ID"]
inter = inter.groupby("comb").agg("median")

inter.reset_index(inplace=True)
inter["InChI Key"] = inter["comb"].apply(lambda x: x.split("_")[0])
inter["UniProt ID"] = inter["comb"].apply(lambda x: x.split("_")[1])
update_counts("Removed duplicates")

# Then apply filters that are defined in config.yaml

if snakemake.config["parse_glass"]["species"] == "human":
    targ = targ[targ["Species"] == "Homo sapiens (Human)"]
    inter = inter[inter["UniProt ID"].isin(targ["UniProt ID"])]
    update_counts("Species")

if snakemake.config["parse_glass"]["druglikeness"] != 0:
    lig = lig[lig["druglikeness"] > snakemake.config["parse_glass"]["druglikeness"]]
    inter = inter[inter["InChI Key"].isin(lig["InChI Key"])]
    update_counts("Druglikeness")

if snakemake.config["parse_glass"]["max_atoms"]:
    lig["numatoms"] = lig["Canonical SMILES"].apply(get_numatoms)
    lig = lig[lig["numatoms"] < snakemake.config["parse_glass"]["max_atoms"]]
    inter = inter[inter["InChI Key"].isin(lig["InChI Key"])]
    update_counts("Num atoms")

if snakemake.config["parse_glass"]["filtering"] == "balanced":
    inter["y"] = inter["Value"].apply(lambda x: 1 if x < 1000 else 0)
    vc = inter["UniProt ID"].value_counts()
    vc = pd.DataFrame(vc)
    vc = vc.reset_index()
    vc.columns = ["UniProt ID", "count"]
    inter = inter.merge(vc, left_on="UniProt ID", right_on="UniProt ID")
    inter["weight"] = inter["count"].apply(lambda x: 1 / (x ** 2))
    pos = inter[inter["y"] == 1].sample(30000, weights="weight")
    neg = inter[inter["y"] == 0].sample(30000, weights="weight")
    inter = pd.concat([pos, neg]).drop(["y", "weight", "count"], axis=1)
    update_counts("Balanced")

valmax = np.inf
valmin = -np.inf
if "valmax" in snakemake.config["parse_glass"]:
    valmax = snakemake.config["parse_glass"]["valmax"]
if "valmin" in snakemake.config["parse_glass"]:
    valmin = snakemake.config["parse_glass"]["valmin"]

inter = inter[inter["Value"].between(valmin, valmax)]
inter["UniProt ID"] = inter["comb"].apply(lambda x: x.split("_")[1])
update_counts("Outlier values")

if snakemake.config["parse_glass"]["filtering"] == "posneg":
    threshold = snakemake.config["prepare_all"]["threshold"]
    pos = inter[inter["Value"] > threshold]["InChI Key"].unique()
    neg = inter[inter["Value"] <= threshold]["InChI Key"].unique()
    both = set(pos).intersection(set(neg))
    inter = inter[inter["InChI Key"].isin(both)]
    update_counts("Pos and neg presence")

targ = targ[targ["UniProt ID"].isin(inter["UniProt ID"])]
lig = lig[lig["InChI Key"].isin(inter["InChI Key"])]
inter = inter[inter["InChI Key"].isin(lig["InChI Key"].unique())]

inter.to_csv(snakemake.output.inter, index=False)
lig.to_csv(snakemake.output.lig, index=False)
targ.to_csv(snakemake.output.targ, index=False)
with open(snakemake.output.log, "wb") as file:
    pickle.dump(counts, file)
