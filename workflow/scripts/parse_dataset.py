import pandas as pd

inter = pd.read_csv(snakemake.input.inter, sep="\t")
lig = pd.read_csv(snakemake.input.lig, sep="\t")
lig.drop_duplicates("Drug_ID", inplace=True)


# If duplicates, take median of entries
inter = inter.groupby(["Drug_ID", "Target_ID"]).agg("median").reset_index()
inter["Y"] = inter["Y"].apply(lambda x: 1 if x < snakemake.config["parse_dataset"]["threshold"] else 0)

if snakemake.config["parse_dataset"]["filtering"] == "balanced":
    num_pos = inter[inter["Y"] == 1]
    num_neg = inter[inter["Y"] == 0]
    vc = inter["Target_ID"].value_counts()
    vc = pd.DataFrame(vc)
    vc = vc.reset_index()
    vc.columns = ["Target_ID", "count"]
    inter = inter.merge(vc, left_on="Target_ID", right_on="Target_ID")
    inter["weight"] = inter["count"].apply(lambda x: 1 / (x ** 2))
    pos = inter[inter["y"] == 1].sample(min(num_pos, num_neg), weights="weight")
    neg = inter[inter["y"] == 0].sample(min(num_pos, num_neg), weights="weight")
    inter = pd.concat([pos, neg]).drop(["y", "weight", "count"], axis=1)
elif snakemake.config["parse_dataset"]["filtering"] == "posneg":
    pos = inter[inter["Y"] == 1]["Drug_ID"].unique()
    neg = inter[inter["Y"] == 0]["Drug_ID"].unique()
    both = set(pos).intersection(set(neg))
    inter = inter[inter["Drug_ID"].isin(both)]
elif snakemake.config["parse_dataset"]["filtering"] != "none":
    raise ValueError("No such type of filtering!")

inter.to_csv(snakemake.output.inter, index=False, sep="\t")
lig.to_csv(snakemake.output.lig, index=False, sep="\t")
