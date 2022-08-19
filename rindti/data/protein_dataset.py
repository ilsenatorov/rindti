from pathlib import Path
from typing import Callable

import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm

from .pdb_parser import PDBStructure


class LargePreTrainDataset(Dataset):
    r"""Dataset of protein graphs - for each PDB file in the input_dir a point cloud is created (`pos`, `x` and `plddt` attributes are created).
    Each processed graph is saved and loaded individually."""

    def __init__(
        self,
        root: str,
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
        threads: int = 1,
    ):
        self._len = None
        self.input_dir = Path(root)
        self.threads = threads
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        """All generated filenames."""
        return [f"data_{i}.pt" for i in range(self.len())]

    def process(self):
        """Convert pdbs into graphs."""

        def process(idx: int, raw_path: Path) -> None:
            s = PDBStructure(raw_path)
            data = Data(**s.get_graph(), uniprot_id=raw_path.stem)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, Path(self.processed_dir) / f"data_{idx}.pt")

        Parallel(n_jobs=self.threads)(
            delayed(process)(idx, i) for idx, i in tqdm(enumerate(self.input_dir.glob("*.pdb")))
        )
        self.num_structs = len([x for x in self.input_dir.glob("*.pdb")])

    def _set_len(self):
        """Calculating length each time is slow (~1s for 200k structures), setting it to an attribute."""
        self._len = len([x for x in self.input_dir.glob("*.pdb")])

    def len(self):
        """Number of graphs in the dataset."""
        if self._len is None:
            self._set_len()
        return self._len

    def get(self, idx: int):
        """Load a single graph."""
        return torch.load(self.processed_dir + "/" + f"data_{idx}.pt")


class LargePreTrainMemoryDataset(InMemoryDataset):
    r"""Dataset of protein graphs - for each PDB file in the input_dir a point cloud is created (`pos`, `x` and `plddt` attributes are created).
    All proteins are saved into a single `.pt` file."""

    def __init__(
        self,
        root: str,
        transform: Callable = None,
        pre_transform: Callable = None,
        threads: int = 1,
    ):
        self.input_dir = Path(root)
        self.threads = threads
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        """Which files have to be in the dir to consider dataset processed.

        Returns:
            Iterable[str]: list of files
        """
        return ["data.pt"]

    def process(self):
        """If the dataset was not seen before, process everything."""
        data_list = []

        def get_graph(filename: str) -> dict:
            """Get a graph using threshold as a cutoff"""
            s = PDBStructure(filename)
            entry = Data(**s.get_graph())
            if self.pre_transform is not None:
                entry = self.pre_transform(entry)
            return entry

        data_list = Parallel(n_jobs=self.threads)(delayed(get_graph)(i) for i in tqdm(self.input_dir.glob("*.pdb")))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
