import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .data import TwoGraphData
from .utils import add_arg_prefix


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


class PfamTransformer(BaseTransformer):
    """For given protein uniprot id, find a negative or positive match

    Args:
        merged_df (pd.DataFrame): Has to contain 'fam' that defines the family and 'data' with normal torch_geometric data
        pos_balance (float, optional): How often to pick positive matches. Defaults to 0.5.
        min_fam_entries (int, optional): Lower cutoff for a family 'too small'. Defaults to 5.
    """

    def __init__(self, merged_df: pd.DataFrame, pos_balance: float = 0.7, min_fam_entries: int = 5):
        assert pos_balance >= 0 and pos_balance <= 1, "pos_balance not between 0 and 1!"
        self.pos_balance = pos_balance
        self.merged_df = merged_df
        if min_fam_entries:
            vc = self.merged_df["fam"].value_counts()
            small_families = vc[vc < min_fam_entries].index
            prot_in_small_families = self.merged_df[self.merged_df["fam"].isin(small_families)].index
            self.merged_df.loc[prot_in_small_families, "fam"] = "Other"

    def __call__(self, data: Data) -> TwoGraphData:
        """Find a matching graph pair for this protein

        Args:
            data (Data): Protein data (has to contain id field for this to work)

        Returns:
            TwoGraphData: all features of original graph become 'a_<feature>', all features of the pair are 'b_<feature>'
        """
        family = self.merged_df.loc[data.id, "fam"]
        new_data = add_arg_prefix("a_", data)
        if family == "Other":  # If from a small family only negative samples are allowed
            label = 0
        else:
            label = np.random.choice([True, False], size=1, p=[self.pos_balance, 1 - self.pos_balance])
        if label == 1:
            new_data.update(add_arg_prefix("b_", self._get_pos_sample(family)))
        else:
            new_data.update(add_arg_prefix("b_", self._get_neg_sample(family)))
        new_data["label"] = torch.tensor(label, dtype=torch.long)
        return TwoGraphData(**new_data)

    def _process_sampled_row(self, sampled_row: pd.Series) -> dict:
        """Helper function for other processes, given a sampled row extract data from it"""
        data = sampled_row["data"]
        data["id"] = sampled_row.name
        return data

    def _get_pos_sample(self, family: str) -> dict:
        """Get a positive match for given family (protein from same family)

        Args:
            family (str): Family name

        Returns:
            dict: data of protein from same family (can be the same protein!)
        """
        sampled_row = self.merged_df[self.merged_df["fam"] == family].sample().iloc[0]
        return self._process_sampled_row(sampled_row)

    def _get_neg_sample(self, family: str) -> dict:
        """Get a negative match for given family (protein from another family)

        Args:
            family (str): Family name

        Returns:
            dict: data of protein from another family
        """
        sampled_row = self.merged_df[self.merged_df["fam"] != family].sample().iloc[0]
        return self._process_sampled_row(sampled_row)

    def _filter(self, data: Data) -> bool:
        """Returns True if graph in self.merged_df else False"""
        return data.id in self.merged_df.index

    @staticmethod
    def from_pickle(filename: str):
        merged_df = pd.read_pickle(filename)
        return PfamTransformer(merged_df)


class SizeFilter(object):
    def __init__(self, max_nnodes: int, min_nnodes: int = 0):
        """Filters out graph that are too big/small"""
        self.max_nnodes = max_nnodes
        self.min_nnodes = min_nnodes

    def __call__(self, data: Data) -> bool:
        """Returns True if number of nodes in given graph is within required values else False"""
        nnodes = data.x.size(0)
        return nnodes > self.min_nnodes and nnodes < self.max_nnodes
