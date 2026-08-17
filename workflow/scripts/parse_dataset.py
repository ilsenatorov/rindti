import numpy as np
import pandas as pd

# Affinities outside this range are measurement artefacts rather than real values.
# GLASS in particular carries negative affinities, exact zeros and values up to 1e28.
MIN_VALID_AFFINITY = 1e-3
MAX_VALID_AFFINITY = 1e6


def drop_invalid_affinities(inter: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose affinity cannot be a real measurement.

    Only applied to value-based units (``nM``); unitless scores such as KIBA are
    left alone.
    """
    valid = inter["Y"].between(MIN_VALID_AFFINITY, MAX_VALID_AFFINITY)
    dropped = (~valid).sum()
    if dropped:
        print(f"Dropping {dropped} of {len(inter)} interactions with out-of-range affinities")
    return inter[valid]


def binarize(inter: pd.DataFrame, threshold: float, unit: str) -> pd.DataFrame:
    """Turn continuous affinities into binary interaction labels.

    The direction depends on the unit. For ``nM`` a *lower* value means a stronger
    interaction, so positives are ``Y < threshold``. For a unitless score such as
    KIBA a *higher* value means stronger, so positives are ``Y >= threshold``.
    """
    if unit == "nM":
        inter["Y"] = (inter["Y"] < threshold).astype(int)
    elif unit == "score":
        inter["Y"] = (inter["Y"] >= threshold).astype(int)
    else:
        raise ValueError(f"Unknown affinity unit {unit!r}, expected 'nM' or 'score'")
    return inter


def posneg_filter(inter: pd.DataFrame) -> pd.DataFrame:
    """Only keep drugs that have at least 1 positive and negative interaction"""
    pos = inter[inter["Y"] == 1]["Drug_ID"].unique()
    neg = inter[inter["Y"] == 0]["Drug_ID"].unique()
    both = set(pos).intersection(set(neg))
    inter = inter[inter["Drug_ID"].isin(both)]
    return inter


def balanced_filter(inter: pd.DataFrame) -> pd.DataFrame:
    """Globally downsample the majority class to match the minority class."""
    pos = inter[inter["Y"] == 1]
    neg = inter[inter["Y"] == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        return inter.iloc[0:0]
    return pd.concat([pos.sample(n), neg.sample(n)])


def sample(inter: pd.DataFrame, how: str = "under") -> pd.DataFrame:
    """Sample the interactions dataset, per target.

    Args:
        inter (pd.DataFrame): whole data, has to be binary class
        how (str, optional): over or undersample. Defaults to "under".

    Warning:
        ``over`` does not resample existing negatives. It takes interactions
        belonging to *other* targets, reassigns them to this one and relabels them
        as negative - i.e. it fabricates decoys on the assumption that a random
        drug does not bind a given target. That assumption injects false negatives
        whenever the drug does in fact bind. Prefer ``under`` or ``none`` unless
        you specifically want decoy-based negatives.
    """
    if how == "none":
        return inter
    if how not in ("over", "under"):
        raise ValueError(f"Unknown sampling method {how!r}")
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
            else:  # over - see the warning in the docstring
                total.append(possample)
                total.append(negsample)
                subsample = inter[inter["Target_ID"] != prot].sample(poscount - negcount)
                subsample["Target_ID"] = prot
                subsample["Y"] = 0
                total.append(subsample)
        else:
            total.append(possample)
            total.append(negsample.sample(poscount))
    if not total:
        return inter.iloc[0:0]
    return pd.concat(total)


def check_not_degenerate(inter: pd.DataFrame, task: str, stage: str) -> None:
    """Fail loudly rather than writing an empty or single-class dataset.

    An empty result used to be written out as a header-only TSV and only surfaced
    much later as an unexplained pipeline failure.
    """
    if inter.empty:
        raise ValueError(
            f"{stage} produced an empty dataset. If you changed parse_dataset.threshold "
            "or unit, check they match the affinity scale of this dataset."
        )
    if task == "class":
        counts = inter["Y"].value_counts().to_dict()
        if len(counts) < 2:
            raise ValueError(
                f"{stage} produced a single-class dataset ({counts}). The threshold is "
                "almost certainly on the wrong scale for this dataset's affinity unit."
            )


if __name__ == "__main__":
    from lightning.pytorch import seed_everything

    seed_everything(snakemake.config["seed"])

    inter = pd.read_csv(snakemake.input.inter, sep="\t")

    config = snakemake.config["parse_dataset"]
    unit = config["unit"]

    if unit == "nM":
        inter = drop_invalid_affinities(inter)

    # If duplicates, take median of entries
    inter = inter.groupby(["Drug_ID", "Target_ID"]).agg("median").reset_index()

    if config["task"] == "class":
        inter = binarize(inter, config["threshold"], unit)
    elif config["task"] == "reg":
        if config["log"]:
            inter["Y"] = inter["Y"].apply(np.log10)
    else:
        raise ValueError("Unknown task!")

    check_not_degenerate(inter, config["task"], "Binarization")

    if config["task"] == "reg" and (config["filtering"] != "all" or config["sampling"] != "none"):
        raise ValueError(f"Can't use filtering {config['filtering']} / sampling {config['sampling']} with task reg!")

    if config["filtering"] == "posneg":
        inter = posneg_filter(inter)
    elif config["filtering"] == "balanced":
        inter = balanced_filter(inter)
    elif config["filtering"] != "all":
        raise ValueError("No such type of filtering!")

    check_not_degenerate(inter, config["task"], f"Filtering ({config['filtering']})")

    inter = sample(inter, how=config["sampling"])

    check_not_degenerate(inter, config["task"], f"Sampling ({config['sampling']})")

    inter.to_csv(snakemake.output.inter, index=False, sep="\t")
