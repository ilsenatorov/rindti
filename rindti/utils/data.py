"""
Just a collection of different useful functions, data structures and helpers.
"""

import os
import pickle
from typing import Any, Callable, Iterable

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import degree


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

    def n_nodes(self, prefix: str) -> int:
        """Number of nodes

        Args:
            prefix (str): prefix for which to count ('prot_', 'drug_')

        Returns:
            int: Number of nodes
        """
        return self.__dict__[prefix + "x"].size(0)

    def n_edges(self, prefix: str) -> int:
        """Returns number of edges for graph with prefix"""
        return self.__dict__[prefix + "edge_index"].size(1)

    def n_node_feats(self, prefix: str) -> int:
        """Calculate the feature dimension of one of the graphs.
        If the features are index-encoded (dtype long, single number for each node, for use with Embedding),
        then return the max. Otherwise return size(1)
        """
        x = self.__dict__[prefix + "x"]
        if len(x.size()) == 1:
            return x.max().item() + 1
        if len(x.size()) == 2:
            return x.size(1)
        raise ValueError("Too many dimensions in input features")

    def n_edge_feats(self, prefix: str) -> int:
        """Returns number of different edges for graph with prefix"""
        if prefix + "edge_feats" not in self.__dict__:
            return 1
        edge_feats = self.__dict__[prefix + "edge_feats"]
        if len(edge_feats.size()) == 1:
            return edge_feats.max().item() + 1
        if len(edge_feats.size()) == 2:
            return edge_feats.size(1)
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
        pre_filter: Callable = None,
    ):
        pre_transform_tag = "" if pre_transform is None else str(pre_transform)
        pre_filter_tag = "" if pre_transform is None else str(pre_filter)
        basefilename = os.path.basename(filename)
        basefilename = os.path.splitext(basefilename)[0]
        root = os.path.join("data", basefilename + pre_transform_tag + pre_filter_tag)
        self.filename = filename
        super().__init__(root, transform, pre_transform, pre_filter)
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
        """Files that are created"""
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
        self.info["prot_deg"] = torch.zeros(100, dtype=torch.long)
        self.info["drug_deg"] = torch.zeros(10, dtype=torch.long)
        for data in data_list:
            d = degree(data["prot_edge_index"][1], num_nodes=data.n_nodes("prot_"), dtype=torch.long)
            self.info["prot_deg"] += torch.bincount(d, minlength=self.info["prot_deg"].numel())
            d = degree(data["drug_edge_index"][1], num_nodes=data.n_nodes("drug_"), dtype=torch.long)
            self.info["drug_deg"] += torch.bincount(d, minlength=self.info["drug_deg"].numel())

        data, slices = self.collate(data_list)
        torch.save((data, slices, self.info), self.processed_paths[s])

    def _update_info(self, key: str, value: int):
        self.info[key] = max(self.info[key], value)

    def process(self):
        """If the dataset was not seen before, process everything"""
        self.info = dict(
            prot_max_nodes=0, drug_max_nodes=0, prot_feat_dim=0, drug_feat_dim=0, prot_edge_dim=0, drug_edge_dim=0
        )
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
                    self._update_info("prot_max_nodes", two_graph_data.n_nodes("prot_"))
                    self._update_info("drug_max_nodes", two_graph_data.n_nodes("drug_"))
                    self._update_info("prot_feat_dim", two_graph_data.n_node_feats("prot_"))
                    self._update_info("drug_feat_dim", two_graph_data.n_node_feats("drug_"))
                    self._update_info("prot_edge_dim", two_graph_data.n_edge_feats("prot_"))
                    self._update_info("drug_edge_dim", two_graph_data.n_edge_feats("drug_"))
                    data_list.append(two_graph_data)
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
