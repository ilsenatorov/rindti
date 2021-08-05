"""
Just a collection of different useful functions, data structures and helpers.
"""

import os
import pickle
from copy import deepcopy
from math import ceil
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class TwoGraphData(Data):
    """
    Subclass of torch_geometric.data.Data for protein and drug data.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __inc__(self, key: str, value: Any) -> dict:
        """How to increment values during batching


        Args:
            key (str): [description]
            value (Any): [description]

        Returns:
            dict: [description]
        """
        if not key.endswith("edge_index"):
            return super().__inc__(key, value)

        lenedg = len("edge_index")
        prefix = key[:-lenedg]
        return self.__dict__[prefix + "x"].size(0)

    def nnodes(self, prefix: str) -> int:
        return self.__dict__[prefix + "x"].size(0)

    def numfeats(self, prefix: str) -> int:
        """
        Calculate the feature dimension of one of the graphs.
        If the features are index-encoded (dtype long, single number for each node, for use with Embedding),
        then return the max. Otherwise return size(1)
        :param prefix: str for prefix "drug_", "prot_" or else
        """
        x = self.__dict__[prefix + "x"]
        if len(x.size()) == 1:
            return x.max().item() + 1
        if len(x.size()) == 2:
            return x.size(1)
        raise ValueError("Too many dimensions in input features")


class Dataset(InMemoryDataset):
    """Dataset class for proteins and drugs

    Args:
        filename (str): Pickle file that stores the data
        split (str, optional): Split type ('train', 'val', 'test). Defaults to "train".
        transform (Callable, optional): transformer to apply on each access. Defaults to None.
        pre_transform (Callable, optional): pre-transformer to apply once before. Defaults to None.
    """

    def __init__(
        self,
        filename: str,
        split: str = "train",
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        basefilename = os.path.basename(filename)
        basefilename = os.path.splitext(basefilename)[0]
        root = os.path.join("data", basefilename)
        self.filename = filename
        super().__init__(root, transform, pre_transform)
        if split == "train":
            self.data, self.slices, self.info = torch.load(self.processed_paths[0])
        elif split == "val":
            self.data, self.slices, self.info = torch.load(self.processed_paths[1])
        elif split == "test":
            self.data, self.slices, self.info = torch.load(self.processed_paths[2])
        else:
            raise ValueError("Unknown split!")

    @property
    def processed_file_names(self) -> Iterable[str]:
        """Which files have to be in the dir to consider dataset processed

        Returns:
            Iterable[str]: list of files
        """
        return ["train.pt", "val.pt", "test.pt"]

    def process_(self, data_list: list, s: int):
        """Process the datalist

        Args:
            data_list (list): List of TwoGraphData entries
            s (int): index of train, val or test
        """
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, self.info), self.processed_paths[s])

    def process(self):
        """If the dataset was not seen before, process everything"""
        self.info = dict(prot_max_nodes=0, drug_max_nodes=0, prot_feat_dim=0, drug_feat_dim=0)
        with open(self.filename, "rb") as file:
            all_data = pickle.load(file)
            for s, split in enumerate(["train", "val", "test"]):
                data_list = []
                for i in all_data["data"]:
                    if i["split"] != split:
                        continue
                    prot_id = i["prot_id"]
                    drug_id = i["drug_id"]
                    prot_data = all_data["prots"].loc[prot_id, "data"]
                    drug_data = all_data["drugs"].loc[drug_id, "data"]
                    new_i = {
                        "prot_count": float(all_data["prots"].loc[prot_id, "count"]),
                        "drug_count": float(all_data["drugs"].loc[drug_id, "count"]),
                        "prot_id": prot_id,
                        "drug_id": drug_id,
                        "label": i["label"],
                    }
                    new_i.update({"prot_" + k: v for (k, v) in prot_data.items()})
                    new_i.update({"drug_" + k: v for (k, v) in drug_data.items()})
                    two_graph_data = TwoGraphData(**new_i)
                    self.info["prot_max_nodes"] = max(two_graph_data.nnodes("prot_"), self.info["prot_max_nodes"])
                    self.info["drug_max_nodes"] = max(two_graph_data.nnodes("drug_"), self.info["drug_max_nodes"])
                    self.info["prot_feat_dim"] = max(int(two_graph_data.prot_x.max()), self.info["prot_feat_dim"])
                    self.info["drug_feat_dim"] = max(int(two_graph_data.drug_x.max()), self.info["drug_feat_dim"])
                    data_list.append(two_graph_data)
                self.info["prot_feat_dim"] = data_list[0].numfeats("prot_")
                self.info["drug_feat_dim"] = data_list[0].numfeats("drug_")
                self.process_(data_list, s)


class PreTrainDataset(InMemoryDataset):
    """Dataset class for pre-training

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
        self.data, self.slices, self.info = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> Iterable[str]:
        """Which files have to be in the dir to consider dataset processed

        Returns:
            Iterable[str]: list of files
        """
        return ["data.pt"]

    def process(self):
        """If the dataset was not seen before, process everything"""
        info = {"max_nodes": 0, "feat_dim": 0}
        with open(self.filename, "rb") as file:
            df = pickle.load(file)
            data_list = []
            for x in df["data"]:
                info["max_nodes"] = max(info["max_nodes"], x["x"].size(0))
                info["feat_dim"] = max(info["feat_dim"], int(x["x"].max()))
                del x["index_mapping"]
                data_list.append(Data(**x))

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices, info), self.processed_paths[0])
