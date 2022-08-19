import torch
from torch_geometric.data import Data


class SizeFilter:
    """Filters out graph that are too big/small."""

    def __init__(self, min_nnodes: int, max_nnodes: int = 0):
        self.min_nnodes = min_nnodes
        self.max_nnodes = max_nnodes

    def __call__(self, data: Data) -> bool:
        """Returns True if number of nodes in given graph is within required values else False."""
        nnodes = data.num_nodes
        return nnodes > self.min_nnodes and nnodes < self.max_nnodes


class PosNoise:
    """Add Gaussian noise to the coordinates of the nodes in a graph."""

    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma

    def __call__(self, batch) -> torch.Tensor:
        noise = torch.randn_like(batch.pos) * self.sigma
        batch.pos += noise
        batch.noise = noise
        return batch


class MaskType:
    """Masks the type of the nodes in a graph."""

    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, batch) -> torch.Tensor:
        mask = torch.rand_like(batch.x, dtype=torch.float32) < self.prob
        batch.orig_x = batch.x[mask]
        batch.x[mask] = 20
        batch.mask = mask
        return batch


class MaskTypeWeighted(MaskType):
    """Masks the type of the nodes in a graph."""

    def __call__(self, batch) -> torch.Tensor:
        num_mut = int(batch.x.size(0) * self.prob)
        num_mut_per_aa = int(num_mut / 20)
        mask = []
        for i in range(20):
            indices = torch.where(batch.x == i)[0]
            random_pick = torch.randperm(indices.size(0))[:num_mut_per_aa]
            mask.append(indices[random_pick])
        mask = torch.cat(mask)
        batch.orig_x = batch.x[mask]
        batch.x[mask] = 20
        batch.mask = mask
        return batch
