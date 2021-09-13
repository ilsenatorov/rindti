from copy import deepcopy
from math import ceil

import pytest
import torch

from rindti.utils.transforms import DataCorruptor, corrupt_features, mask_features

LOW = 1
HIGH = 10
N_NODES = 15

fake_features = torch.randint(LOW, HIGH, (N_NODES,), dtype=torch.long)


@pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 1.0])
def test_mask(frac):
    features = deepcopy(fake_features)
    features, idx = mask_features(features, frac)
    expected_zeros = ceil(N_NODES * frac)
    assert torch.count_nonzero(features) == (N_NODES - expected_zeros)
    if frac != 0:
        assert min(idx) >= 0
        assert max(idx) <= N_NODES
        assert len(idx) == expected_zeros


@pytest.mark.parametrize("func", [mask_features, corrupt_features])
@pytest.mark.parametrize("frac", [-1, 3])
def test_assert_fail(func, frac):
    features = deepcopy(fake_features)
    with pytest.raises(AssertionError):
        func(features, frac)


@pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 1.0])
def test_corrupt(frac):
    features = deepcopy(fake_features)
    features, idx = corrupt_features(features, frac)
    expected_corrupt = ceil(N_NODES * frac)
    assert features[fake_features != features].size(0) <= expected_corrupt
    if frac != 0:
        assert min(idx) >= 0
        assert max(idx) <= N_NODES
        assert len(idx) == expected_corrupt
