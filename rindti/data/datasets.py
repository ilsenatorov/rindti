import json
import os
import pickle
import time
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

    def _process_lock(self) -> str:
        """Lock file guarding ``process()`` for this (exp_name, dataset) pair.

        The cache key is the directory, so two jobs sharing an ``exp_name`` and a dataset
        pickle - which is what submitting several model configs against one dataset does -
        both find the cache cold and both call ``process()`` into the same
        ``processed/`` directory. ``InMemoryDataset.save`` is not atomic, so the loser
        can read a half-written ``.pt``. On a cluster filesystem this surfaces much later
        as an unpickling error in a job that looks unrelated.
        """
        return os.path.join(self.root, "processing.lock")

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
        """Get either prot or drug data.

        There used to be an extra branch here copying an ``IUPAC`` column for
        ``drugs.node_feats: IUPAC``. The config schema has never accepted that value, so
        no dataset could reach it.
        """
        graph = all_data[which].loc[id, "data"]
        graph["id"] = id
        return {which.rstrip("s") + "_" + k: v for k, v in graph.items()}

    @property
    def processed_file_names(self) -> Iterable[str]:
        """Files that are created."""
        return [k + ".pt" for k in self.splits.keys()]

    def process(self):
        """If the dataset was not seen before, process everything.

        Serialised across processes: the first to create the lock builds, the rest wait
        and then find the cache warm. ``O_CREAT | O_EXCL`` is atomic on POSIX and on NFS
        for files, which is what the cluster runs on.
        """
        os.makedirs(self.root, exist_ok=True)
        lock = self._process_lock()
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            self._wait_for_other_process(lock)
            if all(os.path.exists(path) for path in self.processed_paths):
                return
            # The holder died without finishing. Take the lock over rather than
            # deadlocking every subsequent job on a stale file.
            print(f"Stale {lock}; reprocessing")
            os.unlink(lock)
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.close(handle)
            self._build_splits()
        finally:
            # Leave no lock behind even on failure, or the next run inherits a stale one.
            if os.path.exists(lock):
                os.unlink(lock)

    @staticmethod
    def _wait_for_other_process(lock: str, timeout: float = 1800, poll: float = 5) -> None:
        """Block while another process holds ``lock``, up to ``timeout`` seconds."""
        waited = 0.0
        while os.path.exists(lock) and waited < timeout:
            time.sleep(poll)
            waited += poll

    def _build_splits(self):
        """Build and save the three splits.

        Not named ``_process``: that is ``torch_geometric.data.Dataset``'s own method,
        the one that checks whether the cache is warm and calls ``process()``. Defining
        it here would override that check, so every construction would rebuild and
        ``process()`` - along with the lock above - would never run at all.
        """
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
