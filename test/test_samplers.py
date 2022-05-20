import random
from collections import Counter

from torch_geometric.loader import DataLoader

from rindti.data import PfamSampler, PreTrainDataset, WeightedPfamSampler

from .conftest import BATCH_PER_EPOCH, BATCH_SIZE, PROT_PER_FAM

# class BaseTestSampler:
#     def test_sampler(self, pretrain_pickle):
#         """Test ability to generate good batch for lifted structure loss"""
#         ds = PreTrainDataset(pretrain_pickle)
#         sampler = self.sampler(
#             ds,
#             batch_size=BATCH_SIZE,
#             prot_per_fam=PROT_PER_FAM,
#             batch_per_epoch=BATCH_PER_EPOCH,
#         )
#         dl = DataLoader(ds, batch_sampler=sampler)
#         total_dpoints = sum(len(batch.fam) for batch in dl)
#         assert total_dpoints == BATCH_SIZE * BATCH_PER_EPOCH


# class TestPfamSampler(BaseTestSampler):

#     sampler = PfamSampler


# class TestWeightedPfamSampler(BaseTestSampler):

#     sampler = WeightedPfamSampler

#     def test_update(self, pretrain_pickle):
#         """Test ability to update from losses"""
#         ds = PreTrainDataset(pretrain_pickle)
#         sampler = self.sampler(
#             ds,
#             batch_size=BATCH_SIZE,
#             prot_per_fam=PROT_PER_FAM,
#             batch_per_epoch=BATCH_PER_EPOCH,
#         )

#         sampler.update_weights({i.id: [random.random()] for i in ds})
#         dl = DataLoader(ds, batch_sampler=sampler)
#         batch = next(iter(dl))
