import os
import pickle
import random
from collections import defaultdict
from typing import Any, Callable, Iterable, List

import pandas as pd
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset


class TwoGraphData(Data):
    """
    Subclass of torch_geometric.data.Data for protein and drug data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> dict:
        """How to increment values during batching"""
        if not key.endswith("edge_index"):
            return super().__inc__(key, value, *args, **kwargs)
        lenedg = len("edge_index")
        prefix = key[:-lenedg]
        return self[prefix + "x"].size(0)

    def n_nodes(self, prefix: str) -> int:
        """Number of nodes for graph with prefix"""
        return self[prefix + "x"].size(0)

    def n_edges(self, prefix: str) -> int:
        """Returns number of edges for graph with prefix"""
        return self[prefix + "edge_index"].size(1)

    def n_node_feats(self, prefix: str) -> int:
        """Calculate the feature dimension of one of the graphs.
        If the features are index-encoded (dtype long, single number for each node, for use with Embedding),
        then return the max. Otherwise return size(1)
        """
        x = self[prefix + "x"]
        if len(x.size()) == 1:
            return x.max().item() + 1
        if len(x.size()) == 2:
            return x.size(1)
        raise ValueError("Too many dimensions in input features")

    def n_edge_feats(self, prefix: str) -> int:
        """Returns number of different edges for graph with prefix"""
        if prefix + "edge_feats" not in self._store:
            return 1
        if self[prefix + "edge_feats"] is None:
            return 1
        edge_feats = self._store[prefix + "edge_feats"]
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
            self.data, self.slices, self.config = torch.load(self.processed_paths[0])
        elif split == "val":
            self.data, self.slices, self.config = torch.load(self.processed_paths[1])
        elif split == "test":
            self.data, self.slices, self.config = torch.load(self.processed_paths[2])
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

        data, slices = self.collate(data_list)
        torch.save((data, slices, self.config), self.processed_paths[s])

    def process(self):
        """If the dataset was not seen before, process everything"""
        with open(self.filename, "rb") as file:
            all_data = pickle.load(file)
            self.config = all_data["config"]
            self.config["prot_max_nodes"] = 0
            self.config["drug_max_nodes"] = 0
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
                    two_graph_data.num_nodes = 1  # supresses the warning
                    self.config["prot_max_nodes"] = max(self.config["prot_max_nodes"], two_graph_data.n_nodes("prot_"))
                    self.config["drug_max_nodes"] = max(self.config["drug_max_nodes"], two_graph_data.n_nodes("drug_"))
                    data_list.append(two_graph_data)
                if data_list:
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
        self.data, self.slices, self.config = torch.load(self.processed_paths[0])

    def index(self, id: str):
        """Find protein by id"""
        return self[self.data.id.index(id)]

    @property
    def processed_file_names(self) -> Iterable[str]:
        """Which files have to be in the dir to consider dataset processed

        Returns:
            Iterable[str]: list of files
        """
        return ["data.pt"]

    def process(self):
        """If the dataset was not seen before, process everything"""
        config = dict(max_nodes=0)
        with open(self.filename, "rb") as file:
            df = pickle.load(file)
            data_list = []
            for id, x in df["data"].to_dict().items():
                if "index_mapping" in x:
                    del x["index_mapping"]
                config["max_nodes"] = max(config["max_nodes"], x["x"].size(0))
                x["id"] = id
                data_list.append(Data(**x))

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices, config), self.processed_paths[0])


class PfamSampler(torch.utils.data.Sampler):
    def __init__(
        self, dataset: PreTrainDataset, batch_size: int = 64, prot_per_fam: int = 8, batch_per_epoch: int = 1000
    ):
        """Sampler for generating pfam-conforming batches.
        Ensure that in each batch there are positive and negative matches for each protein

        Args:
            dataset (PreTrainDataset): dataset, each data point has to contain data.fam
            batch_size (int, optional): Defaults to 64.
            prot_per_fam (int, optional): Number of proteins per family. Defaults to 8
            batch_per_epoch (int, optional): Number of batches per epoch. Defaults to 1000

        """
        assert batch_size % prot_per_fam == 0, "Batch size should be divisible by prot_per_fam!"
        self.dataset = dataset
        self.batch_size = batch_size
        self.prot_per_fam = prot_per_fam
        self.batch_per_epoch = batch_per_epoch
        self.fam_idx = defaultdict(set)
        for i, data in enumerate(self.dataset):
            self.fam_idx[data.fam].add(i)
        self.fam_idx = {k: list(v) for k, v in self.fam_idx.items()}

    def _construct_batch(self) -> List[int]:
        """Creates a single batch. takes self.prot_per_fam families and samples them

        Returns:
            List[int]: Indices of the proteins from the main dataset
        """
        batch = []
        anchor_fams = random.choices(list(self.fam_idx.keys()), k=self.batch_size // self.prot_per_fam)
        for fam in anchor_fams:
            batch += random.choices(self.fam_idx[fam], k=self.prot_per_fam)
        return batch

    def __iter__(self) -> iter:
        """Returns iterator over all the batches in the epoch

        Returns:
            iter: over list of lists, each list is one batch of indices
        """
        return iter([self._construct_batch() for _ in range(self.batch_per_epoch)])

    def __len__(self):
        return self.batch_per_epoch


def split_random(dataset: PreTrainDataset, train_frac: float = 0.8):
    """Randomly split dataset"""
    tot = len(dataset)
    train = int(tot * train_frac)
    val = int(tot * (1 - train_frac))
    return random_split(dataset, [train, val])
