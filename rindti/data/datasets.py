import json
import os
import pickle
from collections.abc import Callable, Iterable

import numpy as np
from torch_geometric.data import InMemoryDataset

from .data import TwoGraphData

CONFIG_FILENAME = "config.json"


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
        self.load(self.processed_paths[self.splits[split]])
        self.config = self._read_config()

    def _set_filenames(self, filename: str, exp_name: str) -> str:
        basefilename = os.path.basename(filename)
        basefilename = os.path.splitext(basefilename)[0]
        self.filename = filename
        return os.path.join("data", exp_name, basefilename)

    @property
    def config_path(self) -> str:
        """Sidecar file holding the snakemake config the dataset was built from."""
        return os.path.join(self.processed_dir, CONFIG_FILENAME)

    def _read_config(self) -> dict:
        with open(self.config_path) as file:
            return json.load(file)

    @staticmethod
    def _jsonable(obj):
        """The snakemake config carries numpy scalars, which json cannot encode."""
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _write_config(self) -> None:
        with open(self.config_path, "w") as file:
            json.dump(self.config, file, indent=2, default=self._jsonable)

    def _get_datum(self, all_data: dict, id: str, which: str, **kwargs) -> dict:
        """Get either prot or drug data."""
        graph = all_data[which].loc[id, "data"]
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
        self.config = {"snakemake": all_data["config"]}
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
            if not data_list:
                continue
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            self.save(data_list, self.processed_paths[self.splits[split]])
        self._write_config()
