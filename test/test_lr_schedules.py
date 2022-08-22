import torch
from torch.optim import Adam

from rindti.lr_schedules.LWCA import LinearWarmupCosineAnnealingLR
from rindti.lr_schedules.LWCAWR import LinearWarmupCosineAnnealingWarmRestartsLR
from rindti.utils import read_config

default_config = read_config("config/test/default_lr.yaml")


class BaseLRTestModel:
    def test_step(self):
        model = torch.nn.Linear(10, 10)
        opt = Adam(model.parameters(), lr=0.1)
        lr = self.lr_scheduler(opt, **default_config)
        for _ in range(100):
            lr.step()
        lr.get_lr()
        sd = lr.state_dict()
        lr = self.lr_scheduler(opt, **default_config)
        lr.load_state_dict(sd)
        lr.step()
        lr.get_lr()


class TestLWCAModel(BaseLRTestModel):
    lr_scheduler = LinearWarmupCosineAnnealingLR


class TestLWCAWRModel(BaseLRTestModel):
    lr_scheduler = LinearWarmupCosineAnnealingWarmRestartsLR
