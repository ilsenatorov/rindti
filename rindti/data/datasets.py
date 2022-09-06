import os
import pickle
import random
from typing import Callable, Iterable

import pandas as pd
import torch
from deprecate import deprecated
from torch_geometric.data import Data, Dataset, InMemoryDataset

from .data import TwoGraphData


@deprecated
def balance_data(data_list):
    """Method to balance the dataset after its creation by snakemake, this is a hot fix"""
    count = {}
    for s in data_list:
        if s["label"] not in count:
            count[s["label"]] = 0
        count[s["label"]] += 1
    majority_class = max(count.keys(), key=lambda k: count[k])
    minority_class = min(count.keys(), key=lambda k: count[k])
    keep_ratio = count[minority_class] / count[majority_class]

    output = []
    for s in data_list:
        if s["label"] == minority_class or random.random() < keep_ratio:
            output.append(s)
    return output


class DTIDataset(InMemoryDataset):
    """Dataset class for prots and drugs.

    Args:
        filename (str): Pickle file that stores the data
        split (str, optional): Split type ('train', 'val', 'test). Defaults to "train".
        transform (Callable, optional): transformer to apply on each access. Defaults to None.
        pre_transform (Callable, optional): pre-transformer to apply once before. Defaults to None.
    """

    splits = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        filename: str,
        exp_name: str,
        split: str = "train",
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
    ):
        root = self._set_filenames(filename, exp_name)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.config = torch.load(self.processed_paths[self.splits[split]])

    def _set_filenames(self, filename: str, exp_name: str) -> str:
        basefilename = os.path.basename(filename)
        basefilename = os.path.splitext(basefilename)[0]
        self.filename = filename
        return os.path.join("data", exp_name, basefilename)

    def process_(self, data_list: list, split: str):
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, self.config), self.processed_paths[self.splits[split]])

    def _get_datum(self, all_data: dict, id: str, which: str, **kwargs) -> dict:
        """Get either prot or drug data."""
        # MEME comment to see if test difference works
        graph = all_data[which].loc[id, "data"]
        if graph != graph:
            graph = {}
        graph["id"] = id
        if (
            which == "drugs"
            and "drugs" in kwargs["snakemake"]
            and kwargs["snakemake"]["drugs"]["node_feats"] == "IUPAC"
        ):
            graph["IUPAC"] = all_data[which].loc[id, "IUPAC"]
        return {which.rstrip("s") + "_" + k: v for k, v in graph.items()}

    @property
    def processed_file_names(self) -> Iterable[str]:
        """Files that are created."""
        return [k + ".pt" for k in self.splits.keys()]

    def process(self):
        """If the dataset was not seen before, process everything."""
        with open(self.filename, "rb") as file:
            all_data = pickle.load(file)
            self.config = {}
            self.config["snakemake"] = all_data["config"]
            for split in self.splits.keys():
                data_list = []
                for i in all_data["data"]:
                    if i["split"] != split:
                        continue
                    data = self._get_datum(all_data, i["prot_id"], "prots", **self.config)
                    data.update(self._get_datum(all_data, i["drug_id"], "drugs", **self.config))
                    data["label"] = i["label"]
                    two_graph_data = TwoGraphData(**data)
                    two_graph_data.num_nodes = 1  # supresses the warning
                    data_list.append(two_graph_data)
                if data_list:
                    # data_list = balance_data(data_list)
                    self.process_(data_list, split)


class PreTrainDataset(InMemoryDataset):
    """Dataset class for pre-training.

    Args:
        filename (str): Pickle file that stores the data
        split (str, optional): Split type ('train', 'val', 'test). Defaults to "train".
        transform (Callable, optional): transformer to apply on each access. Defaults to None.
        pre_transform (Callable, optional): pre-transformer to apply once before. Defaults to None.
    """

    def __init__(self, filename: str, transform: Callable = None, pre_transform: Callable = None):
        basefilename = os.path.basename(filename)
        basefilename = os.path.splitext(basefilename)[0]
        root = os.path.join("data", basefilename)
        self.filename = filename
        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.config = torch.load(self.processed_paths[0])

    def index(self, id: str):
        """Find protein by id."""
        return self[self.data.id.index(id)]

    @property
    def processed_file_names(self) -> Iterable[str]:
        """Which files have to be in the dir to consider dataset processed.

        Returns:
            Iterable[str]: list of files
        """
        return ["data.pt"]

    def process(self):
        """If the dataset was not seen before, process everything."""
        data = pd.read_pickle(self.filename)
        data_list = []
        for i in data["data"]:
            data_list.append(Data(**i))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.config = data["config"]
        data, slices = self.collate(data_list)
        torch.save((data, slices, self.config), self.processed_paths[0])
