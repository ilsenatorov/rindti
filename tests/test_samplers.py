import random
from collections import Counter

from torch_geometric.loader import DataLoader

from rindti.data import PfamSampler, PreTrainDataset, WeightedPfamSampler

from .conftest import BATCH_PER_EPOCH, BATCH_SIZE, PROT_PER_FAM


class BaseTestSampler:
    def test_sampler(self, pretrain_pickle):
        """Test ability to generate good batch for lifted structure loss"""
        ds = PreTrainDataset(pretrain_pickle)
        print(ds)
        sampler = self.sampler(
            ds,
            batch_size=BATCH_SIZE,
            prot_per_fam=PROT_PER_FAM,
            batch_per_epoch=BATCH_PER_EPOCH,
        )
        dl = DataLoader(ds, batch_sampler=sampler)
        total_dpoints = 0
        for batch in dl:
            total_dpoints += len(batch.fam)
            count = Counter(batch.fam)
            print(batch.fam)
            print(batch.id)
            assert count["a"] == PROT_PER_FAM
            assert count["b"] == PROT_PER_FAM
        assert total_dpoints == BATCH_SIZE * BATCH_PER_EPOCH


class TestPfamSampler(BaseTestSampler):

    sampler = PfamSampler


class TestWeightedPfamSampler(BaseTestSampler):

    sampler = WeightedPfamSampler

    def test_update(self, pretrain_pickle):
        """Test ability to update from losses"""
        ds = PreTrainDataset(pretrain_pickle)
        print(ds)
        sampler = self.sampler(
            ds,
            batch_size=BATCH_SIZE,
            prot_per_fam=PROT_PER_FAM,
            batch_per_epoch=BATCH_PER_EPOCH,
        )
        sampler.update_weights({i.id: [random.randint(0, 5)] for i in ds})
        dl = DataLoader(ds, batch_sampler=sampler)
        batch = next(iter(dl))
        count = Counter(batch.fam)
        print(batch.fam)
        print(batch.id)
        assert count["a"] == PROT_PER_FAM
        assert count["b"] == PROT_PER_FAM
