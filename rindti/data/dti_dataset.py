import os
import pickle
from typing import Callable, Iterable

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from .data import TwoGraphData


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
        split: str = "train",
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
    ):
        root = self._set_filenames(filename)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.splits[split]])

    def _set_filenames(self, filename: str) -> str:
        basefilename = os.path.basename(filename)
        basefilename = os.path.splitext(basefilename)[0]
        self.filename = filename
        return os.path.join("data", basefilename)

    def process_(self, data_list: list, split: str):
        """Single loop process."""
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.splits[split]])

    def _get_datum(self, all_data: dict, id: str, which: str, **kwargs) -> dict:
        """Get either prot or drug data."""
        graph = all_data[which].loc[id, "data"]
        graph["id"] = id
        if which == "drugs" and "drugs" in kwargs and kwargs["drugs"]["node_feats"] == "IUPAC":
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
            for split in self.splits.keys():
                data_list = []
                for i in all_data["data"]:
                    if i["split"] != split:
                        continue
                    data = self._get_datum(all_data, i["prot_id"], "prots")
                    data.update(self._get_datum(all_data, i["drug_id"], "drugs"))
                    data["label"] = i["label"]
                    two_graph_data = TwoGraphData(**data)
                    data_list.append(two_graph_data)
                if data_list:
                    self.process_(data_list, split)
