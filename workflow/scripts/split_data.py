import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_groups(
    inter: pd.DataFrame,
    col_name: str = "Target_ID",
    bin_size: int = 10,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
) -> pd.DataFrame:
    """Split data by protein (cold-target)
    Tries to ensure good size of all sets by sorting the prots by number of interactions
    and performing splits within bins of 10

    Args:
        inter (pd.DataFrame): interaction DataFrame
        col_name (str): Which column to split on (col_name or 'Drug_ID' usually)
        bin_size (int, optional): Size of the bins to perform individual splits in. Defaults to 10.
        train_frac (float, optional): value from 0 to 1, how much of the data goes into train
        val_frac (float, optional): value from 0 to 1, how much of the data goes into validation

    Returns:
        pd.DataFrame: DataFrame with a new 'split' column
    """
    sorted_index = [x for x in inter[col_name].value_counts().index]
    train_prop = int(bin_size * train_frac)
    val_prop = int(bin_size * val_frac)
    train = []
    val = []
    test = []
    for i in range(0, len(sorted_index), bin_size):
        subset = sorted_index[i : i + bin_size]
        train_bin = list(np.random.choice(subset, min(len(subset), train_prop), replace=False))
        train += train_bin
        subset = [x for x in subset if x not in train_bin]
        val_bin = list(np.random.choice(subset, min(len(subset), val_prop), replace=False))
        val += val_bin
        subset = [x for x in subset if x not in val_bin]
        test += subset
    train_idx = inter[inter[col_name].isin(train)].index
    val_idx = inter[inter[col_name].isin(val)].index
    test_idx = inter[inter[col_name].isin(test)].index
    inter.loc[train_idx, "split"] = "train"
    inter.loc[val_idx, "split"] = "val"
    inter.loc[test_idx, "split"] = "test"
    return inter


def split_random(inter: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.2) -> pd.DataFrame:
    """Split the dataset in a completely random fashion

    Args:
        inter (pd.DataFrame): interaction DataFrame
        train_frac (float, optional): value from 0 to 1, how much of the data goes into train
        val_frac (float, optional): value from 0 to 1, how much of the data goes into validation

    Returns:
        pd.DataFrame: DataFrame with a new 'split' column
    """
    train, valtest = train_test_split(inter, train_size=train_frac)
    # val_frac is a fraction of the whole dataset, but is applied to what is left
    # after the train split, so it has to be rescaled. Without this, train=0.7 /
    # val=0.2 silently produced 70/6/24 instead of 70/20/10.
    val, test = train_test_split(valtest, train_size=val_frac / (1 - train_frac))
    train.loc[:, "split"] = "train"
    val.loc[:, "split"] = "val"
    test.loc[:, "split"] = "test"
    inter = pd.concat([train, val, test])
    return inter


GROUP_COL = "_group"


def add_group_column(inter: pd.DataFrame, id_col: str, assignment: dict) -> pd.DataFrame:
    """Attach the grouping column that the split is then performed on.

    An ID missing from ``assignment`` becomes its own group rather than an error or a
    dropped row: ``prepare_all.py`` filters interactions against the protein and drug
    pickles, so losing rows here would silently shrink the dataset. Giving an unknown
    ID its own group is also the conservative choice - it cannot leak into another
    group's split.

    Args:
        inter (pd.DataFrame): interaction DataFrame
        id_col (str): 'Target_ID' or 'Drug_ID'
        assignment (dict): ID -> group representative

    Returns:
        pd.DataFrame: the same frame with a group column added
    """
    inter = inter.copy()
    inter[GROUP_COL] = inter[id_col].map(assignment).fillna(inter[id_col])
    n_groups = inter[GROUP_COL].nunique()
    print(f"{inter[id_col].nunique()} distinct {id_col} -> {n_groups} groups")
    return inter


def read_assignment(path: str, id_col: str) -> dict:
    """Read an ``ID<tab>cluster`` table written by one of the clustering rules."""
    return pd.read_csv(path, sep="\t", index_col=id_col)["cluster"].to_dict()


def read_sequences(path: str) -> dict:
    """Read the ``Target_ID -> sequence`` table straight from the source tables."""
    return pd.read_csv(path, sep="\t", index_col="Target_ID")["Target"].to_dict()


if __name__ == "__main__":
    from cluster import dedup_sequences
    from lightning.pytorch import seed_everything

    seed_everything(snakemake.config["seed"])
    inter = pd.read_csv(snakemake.input.inter, sep="\t")
    fracs = {"train_frac": snakemake.params.train, "val_frac": snakemake.params.val}
    method = snakemake.params.method

    if method == "target":
        # Cold-target, deduplicated. Davis carries exact duplicate sequences under
        # different Target_IDs, so splitting on the raw ID puts the same protein in
        # train and test and inflates every reported number.
        assignment = dedup_sequences(read_sequences(snakemake.input.seqs))
        inter = split_groups(add_group_column(inter, "Target_ID", assignment), col_name=GROUP_COL, **fracs)
    elif method == "drug":
        inter = split_groups(inter, col_name="Drug_ID", **fracs)
    elif method == "cluster_target":
        assignment = read_assignment(snakemake.input.clusters, "Target_ID")
        inter = split_groups(add_group_column(inter, "Target_ID", assignment), col_name=GROUP_COL, **fracs)
    elif method == "cluster_drug":
        assignment = read_assignment(snakemake.input.clusters, "Drug_ID")
        inter = split_groups(add_group_column(inter, "Drug_ID", assignment), col_name=GROUP_COL, **fracs)
    elif method == "random":
        inter = split_random(inter, **fracs)
    else:
        raise NotImplementedError(f"Unknown split type {method!r}!")
    inter.drop(columns=GROUP_COL, errors="ignore").to_csv(snakemake.output.split_data, sep="\t")
