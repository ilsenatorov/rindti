import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_groups(inter: pd.DataFrame,
                 bin_size: int = 10,
                 train_frac: float = 0.7,
                 val_frac: float = 0.2) -> pd.DataFrame:
    """Split data by protein (cold-target)
    Tries to ensure good size of all sets by sorting the proteins by number of interactions
    and performing splits within bins of 10

    Args:
        inter (pd.DataFrame): interaction DataFrame
        bin_size (int, optional): Size of the bins to perform individual splits in. Defaults to 10.
        train_frac (float, optional): value from 0 to 1, how much of the data goes into train
        val_frac (float, optional): value from 0 to 1, how much of the data goes into validation

    Returns:
        pd.DataFrame: DataFrame with a new 'split' column
    """
    sorted_index = [x for x in inter['UniProt ID'].value_counts().index]
    train_prop = int(bin_size * train_frac)
    val_prop = int(bin_size * val_frac)
    train = []
    val = []
    test = []
    for i in range(0, len(sorted_index), bin_size):
        subset = sorted_index[i:i+bin_size]
        train_bin = list(np.random.choice(subset, min(len(subset), train_prop), replace=False))
        train += train_bin
        subset = [x for x in subset if x not in train_bin]
        val_bin = list(np.random.choice(subset, min(len(subset), val_prop), replace=False))
        val += val_bin
        subset = [x for x in subset if x not in val_bin]
        test += subset
    train_idx = inter[inter['UniProt ID'].isin(train)].index
    val_idx = inter[inter['UniProt ID'].isin(val)].index
    test_idx = inter[inter['UniProt ID'].isin(test)].index
    inter.loc[train_idx, 'split'] = 'train'
    inter.loc[val_idx, 'split'] = 'val'
    inter.loc[test_idx, 'split'] = 'test'
    return inter


def split_random(inter: pd.DataFrame,
                 train_frac: float = 0.7,
                 val_frac: float = 0.2) -> pd.DataFrame:
    """Split the dataset in a completely random fashion

    Args:
        inter (pd.DataFrame): interaction DataFrame
        train_frac (float, optional): value from 0 to 1, how much of the data goes into train
        val_frac (float, optional): value from 0 to 1, how much of the data goes into validation

    Returns:
        pd.DataFrame: DataFrame with a new 'split' column
    """
    train, valtest = train_test_split(
        inter,
        train_size=train_frac
    )
    val, test = train_test_split(valtest, train_size=val_frac)
    train.loc[:, 'split'] = 'train'
    val.loc[:, 'split'] = 'val'
    test.loc[:, 'split'] = 'test'
    inter = pd.concat([train, val, test])
    return inter


if __name__ == "__main__":
    np.random.seed(snakemake.config['seed'])

    threshold = snakemake.config['prepare_all']['threshold']

    lig = pd.read_csv(snakemake.input.lig).set_index('InChI Key')
    inter = pd.read_csv(snakemake.input.inter)

    inter['y'] = inter['Value'].apply(lambda x: int(x < threshold))
    inter['Canonical SMILES'] = inter['InChI Key'].apply(lambda x: lig.loc[x, 'Canonical SMILES'])

    if snakemake.config['split']['method'] == 'coldtarget':
        inter = split_groups(inter,
                             train_frac=snakemake.config['split']['train'],
                             val_frac=snakemake.config['split']['val'])
    elif snakemake.config['split']['method'] == 'random':
        inter = split_random(inter)
    else:
        raise NotImplementedError('Unknown split type!')
    inter.to_csv(snakemake.output.split_data)
