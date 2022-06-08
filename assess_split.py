import time

import pandas as pd

from workflow.scripts.split_data import get_communities, split_groups


def split(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Split the data.
    Args:
        df (pd.DataFrame): DataFrame "Drug_ID", "Target_ID" and "Y" columns.
        train_frac (float, optional): value from 0 to 1, how much of the data goes into train.
        val_frac (float, optional): value from 0 to 1, how much of the data goes into validation.
        Returns:
            pd.DataFrame: DataFrame with a new 'split' column.
    """
    df = get_communities(df)
    df = split_groups(df, "community", 10, train_frac, val_frac)
    return df


def assess_split(df: pd.DataFrame, orig_num: int, train_frac: float, val_frac: float) -> dict:
    """Assess the split.
    Args:
        df (pd.DataFrame): DataFrame "Drug_ID", "Target_ID", "split" and "Y" columns.
        orig_num (int): Original number of interactions.
        train_frac (float, optional): value from 0 to 1, how much of the data should go into train.
        val_frac (float, optional): value from 0 to 1, how much of the data should into validation.
        Returns:
            dict: Dictionary with the following keys:
                "train_diff": Change in number of train interactions.
                "val_diff": Change in number of val interactions.
                "test_diff": Change in number of test interactions.
                "total_diff": Change in number of total interactions.
    """
    assert "split" in df.columns, "The dataframe must contain a column named 'split'"
    assert train_frac + val_frac <= 1, "The sum of train_frac and val_frac must be less than 1"
    test_frac = 1 - train_frac - val_frac
    train_diff = df[df["split"] == "train"].shape[0] / df.shape[0] - train_frac
    val_diff = df[df["split"] == "val"].shape[0] / df.shape[0] - val_frac
    test_diff = df[df["split"] == "test"].shape[0] / df.shape[0] - test_frac
    total_diff = (df.shape[0] - orig_num) / orig_num
    return dict(train_diff=train_diff, val_diff=val_diff, test_diff=test_diff, total_diff=total_diff)


def main(filename: str, num_iter: int = 10, train_frac: float = 0.7, val_frac: float = 0.2):
    """Main function.
    Args:
        filename (str): Path to the dataset file to be split, should be a tsv with necessary columns.
        num_iter (int, optional): Number of iterations.
        train_frac (float, optional): value from 0 to 1, how much of the data should go into train.
        val_frac (float, optional): value from 0 to 1, how much of the data should into validation.
    """
    df = pd.read_csv(filename, sep="\t")
    results = []
    orig_num = df.shape[0]
    for i in range(num_iter):
        start = time.time()
        df = split(df, train_frac, val_frac)
        metrics = assess_split(df, orig_num, train_frac, val_frac)
        metrics["time"] = time.time() - start
        results.append(metrics)
    results = pd.DataFrame(results)
    print("Mean")
    print(results.mean())
    print("Std")
    print(results.std())


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
