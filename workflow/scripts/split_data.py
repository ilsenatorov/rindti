import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def allocation_pattern(bin_size: int, train_frac: float, val_frac: float) -> list[str]:
    """One split label per slot in a bin, e.g. 7 train + 2 val + 1 test for the defaults.

    Test takes the remainder of the bin rather than its own rounded share, so the three
    always sum to ``bin_size`` and no slot is left unassigned.
    """
    pattern = ["train"] * round(bin_size * train_frac) + ["val"] * round(bin_size * val_frac)
    if len(pattern) > bin_size:
        raise ValueError(f"train_frac + val_frac exceed 1 for bin_size {bin_size}")
    return pattern + ["test"] * (bin_size - len(pattern))


def split_groups(
    inter: pd.DataFrame,
    col_name: str = "Target_ID",
    bin_size: int = 10,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
) -> pd.DataFrame:
    """Split by entity (cold-target, cold-drug, or a cluster of either).

    Entities are sorted by interaction count and allocated within bins of ``bin_size``,
    so each split gets a comparable mix of well- and sparsely-measured entities rather
    than test inheriting only the rare ones.

    Allocation deals a shuffled pattern of split labels across each bin. It used to take
    ``min(len(subset), int(bin_size * train_frac))`` for train and then the same for val,
    which is correct for a full bin but not for a short one: the proportions were
    relative to ``bin_size`` instead of to the bin actually in hand, so the final partial
    bin was always biased toward train, and a dataset with fewer than ``bin_size``
    entities went *entirely* to train - leaving val and test empty, and training then
    early-stopping on a validation set that did not exist. Dealing from a shuffled
    pattern is unbiased for a partial bin and identical for a full one.

    Small datasets can still end up with an empty split - five groups cannot be divided
    70/20/10 - but that is now a property of the arithmetic rather than a systematic
    bias, and ``workflow/scripts/dataset_stats.py`` reports it.

    Args:
        inter (pd.DataFrame): interaction DataFrame
        col_name (str): which column to split on ('Target_ID', 'Drug_ID' or a group column)
        bin_size (int): size of the bins to allocate within. Defaults to 10.
        train_frac (float): value from 0 to 1, how much of the data goes into train
        val_frac (float): value from 0 to 1, how much of the data goes into validation

    Returns:
        pd.DataFrame: DataFrame with a new 'split' column
    """
    sorted_index = list(inter[col_name].value_counts().index)
    pattern = allocation_pattern(bin_size, train_frac, val_frac)

    assignment = {}
    for start in range(0, len(sorted_index), bin_size):
        subset = sorted_index[start : start + bin_size]
        # Shuffled per bin, so which slot a given rank lands in is not fixed across bins.
        for entity, split in zip(subset, np.random.permutation(pattern), strict=False):
            assignment[entity] = split

    inter["split"] = inter[col_name].map(assignment)
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


def read_smiles(path: str) -> dict:
    """Read the ``Drug_ID -> SMILES`` table straight from the source tables."""
    return pd.read_csv(path, sep="\t", index_col="Drug_ID")["Drug"].to_dict()


if __name__ == "__main__":
    from cluster import dedup_sequences, dedup_smiles
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
        # Cold-drug, deduplicated, for the same reason `target` is: the same molecule
        # under two Drug_IDs would otherwise be in train and test at once.
        assignment = dedup_smiles(read_smiles(snakemake.input.smiles))
        inter = split_groups(add_group_column(inter, "Drug_ID", assignment), col_name=GROUP_COL, **fracs)
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
