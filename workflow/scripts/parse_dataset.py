import numpy as np
import pandas as pd


def posneg_filter(inter: pd.DataFrame) -> pd.DataFrame:
    """Only keep drugs that have at least 1 positive and negative interaction"""
    pos = inter[inter["Y"] == 1]["Drug_ID"].unique()
    neg = inter[inter["Y"] == 0]["Drug_ID"].unique()
    both = set(pos).intersection(set(neg))
    inter = inter[inter["Drug_ID"].isin(both)]
    return inter


def sample(inter: pd.DataFrame, how: str = "under") -> pd.DataFrame:
    """Sample the interactions dataset

    Args:
        inter (pd.DataFrame): whole data, has to be binary class
        how (str, optional): over or undersample.
        Oversample adds fake negatives, undersample removed extra positives. Defaults to "under".
    """
    if how == "none":
        return inter

    total = []
    pos = inter[inter["Y"] == 1]
    neg = inter[inter["Y"] == 0]
    for prot in inter["Target_ID"].unique():
        possample = pos[pos["Target_ID"] == prot]
        negsample = neg[neg["Target_ID"] == prot]
        poscount = possample.shape[0]
        negcount = negsample.shape[0]
        if poscount == 0:
            continue
        if poscount >= negcount:
            if how == "under":
                total.append(possample.sample(negcount))
                total.append(negsample)
            elif how == "over":
                total.append(possample)
                total.append(negsample)
                subsample = inter[inter["Target_ID"] != prot].sample(poscount - negcount)
                subsample["Target_ID"] = prot
                subsample["Y"] = 0
                total.append(subsample)
            else:
                raise ValueError("Unknown sampling method!")
        else:
            total.append(possample)
            total.append(negsample.sample(poscount))
    return pd.concat(total)


if __name__ == "__main__":

    inter = pd.read_csv(snakemake.input.inter, sep="\t")

    config = snakemake.config["parse_dataset"]
    # If duplicates, take median of entries
    inter = inter.groupby(["Drug_ID", "Target_ID"]).agg("median").reset_index()
    if config["task"] == "class":
        inter["Y"] = inter["Y"].apply(lambda x: int(x < config["threshold"]))
    elif config["task"] == "reg":
        if config["log"]:
            inter["Y"] = inter["Y"].apply(np.log10)
    else:
        raise ValueError("Unknown task!")

    if config["filtering"] != "all" and config["sampling"] != "none" and config["task"] == "reg":
        raise ValueError(
            "Can't use filtering {filter} with task {task}!".format(filter=config["filtering"], task=config["task"])
        )

    if config["filtering"] == "posneg":
        inter = posneg_filter(inter)
    elif config["filtering"] != "all":
        raise ValueError("No such type of filtering!")

    inter = sample(inter, how=config["sampling"])

    inter.to_csv(snakemake.output.inter, index=False, sep="\t")
