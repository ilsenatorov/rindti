import pickle
import random

import numpy as np
import torch

from .data import TwoGraphData


class BaseTransformer(object):
    def __init__(self, **kwargs):
        raise NotImplementedError()

    def __call__(self, data: TwoGraphData) -> TwoGraphData:
        raise NotImplementedError()


class GnomadTransformer(BaseTransformer):
    """Transformer of TwoGraphData entries

    Args:
        gnomad (dict): Dict of gnomad DataFrames for each protein
        index_mapping (dict): Dict of index mappings for each protein
        encoded_residues (dict): Dict of encoded tensors for each residue
        max_num_mut (int): Upper limit of number of mutations
    """

    def __init__(
        self,
        gnomad: dict,
        index_mapping: dict,
        encoded_residues: dict,
        max_num_mut: int = 50,
    ):
        self.gnomad = gnomad
        self.index_mapping = index_mapping
        self.encoded_residues = encoded_residues
        self.max_num_mut = max_num_mut

    def __call__(self, data: TwoGraphData) -> TwoGraphData:
        """Called within the Dataset class during loading the data

        Args:
            data (TwoGraphData): Prot + Drug entry

        Returns:
            (TwoGraphData): entry with modified protein features
        """
        prot_id = data["prot_id"]
        if prot_id not in self.gnomad:
            return data
        x = data["prot_x"]
        mutations = self.gnomad[prot_id]
        mutations = mutations.sample(frac=1).drop_duplicates("mut_pos")
        num_mut = min(np.random.randint(0, self.max_num_mut), mutations.shape[0])
        if num_mut == 0:
            return data
        mutations = mutations.sample(num_mut)

        new_x = x.detach().clone()
        for i, row in mutations.iterrows():
            position_index = self.index_mapping[prot_id][row["mut_pos"]]
            new_x_row = self.encoded_residues[row["mut_to"].lower()].detach().clone()
            new_x[position_index, :] = new_x_row
        data["prot_x"] = new_x
        return data

    @staticmethod
    def from_pickle(filename: str, max_num_mut=50) -> BaseTransformer:
        """Load transformer from pickle

        Args:
            filename (str): Pickle file location
            max_num_mut (int, optional): Maximal number of mutation to applyt. Defaults to 50.

        Returns:
            GnomadTransformer: Transformer
        """
        with open(filename, "rb") as file:
            all_data = pickle.load(file)
        return GnomadTransformer(
            all_data["gnomad"],
            all_data["index_mapping"],
            all_data["encoded_residues"],
            max_num_mut=max_num_mut,
        )


class RandomTransformer(BaseTransformer):
    """Random Transformer of TwoGraphData entries

    Args:
        encoded_residues (dict): Dict of encoded tensors for each residue
        max_num_mut (int): Upper limit of number of mutations
    """

    def __init__(self, encoded_residues: dict, max_num_mut: int = 50):
        self.encoded_residues = list(encoded_residues.values())
        self.max_num_mut = max_num_mut

    def __call__(self, data: TwoGraphData) -> TwoGraphData:
        """Called within the Dataset class during loading the data

        Args:
            data (TwoGraphData): Prot + Drug entry

        Returns:
            (TwoGraphData): entry with modified protein features
        """
        x = data["prot_x"]
        num_mut = min(np.random.randint(0, self.max_num_mut), x.size(1))
        positions_to_mutate = np.random.choice(range(x.size(1)), size=num_mut, replace=False)
        for pos in positions_to_mutate:
            new_residue_feature = random.choice(self.encoded_residues)
            x[pos, :] = new_residue_feature.detach().clone()
        data["prot_x"] = x
        return data

    @staticmethod
    def from_pickle(filename: str, max_num_mut=50) -> BaseTransformer:
        """Load transformer from pickle

        Args:
            filename (str): Pickle file location
            max_num_mut (int, optional): Maximal number of mutation to applyt. Defaults to 50.

        Returns:
            RandomTransformer: Transformer
        """
        with open(filename, "rb") as file:
            all_data = pickle.load(file)
        return RandomTransformer(all_data["encoded_residues"], max_num_mut=max_num_mut)
