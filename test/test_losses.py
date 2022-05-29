import pytest
import torch

from rindti.losses import GeneralisedLiftedStructureLoss, NodeLoss, SoftNearestNeighborLoss


@pytest.fixture
def embeds() -> torch.Tensor:
    """Embeddings for testing. Ones and zeros, in order of 4 ones 1 zeros, 4 ones 1 zeros."""
    return torch.cat([torch.ones(4, 10), torch.zeros(1, 10), torch.ones(4, 10), torch.zeros(1, 10)])


@pytest.fixture
def fam_idx() -> torch.Tensor:
    return torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


@pytest.mark.parametrize("loss_class", [GeneralisedLiftedStructureLoss, SoftNearestNeighborLoss])
def test_graph_struct(loss_class, embeds: torch.Tensor, fam_idx: torch.Tensor) -> None:
    loss = loss_class()
    res = loss(embeds, fam_idx)["graph_loss"]
    assert res[:4].allclose(res[5:9])
    assert res[4].allclose(res[9])
    assert res[0] < res[4]
