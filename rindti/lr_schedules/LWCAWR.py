import torch.nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class LinearWarmup:
    """Linearly increase the learning rate"""
    def __init__(self, optimizer, warmup_epochs, start_lr, end_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.steps = 0

    def step(self):
        """Make a step"""
        self.steps += 1

    def get_lr(self):
        """Get learning rate at the current step"""
        return [self.start_lr + self.steps * (self.end_lr - self.start_lr) / self.warmup_epochs]

    def state_dict(self):
        """Return state of the scheduler for later restoring"""
        return {
            "warmup_epochs": self.warmup_epochs,
            "start_lr": self.start_lr,
            "end_lr": self.end_lr,
            "steps": self.steps,
        }

    def load_state_dict(self, state_dict):
        """Restore scheduler from previous state"""
        self.warmup_epochs = state_dict["warmup_epochs"]
        self.start_lr = state_dict["start_lr"]
        self.end_lr = state_dict["end_lr"]
        self.steps = state_dict["steps"]


class LinearWarmupCosineAnnealingWarmRestartsLR:
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            start_lr: float,
            peak_lr: float,
            cos_restart_dist: int,
            cos_t_mult: int = 1,
            cos_eta_min: float = 0.0,
            cos_last_epoch: int = -1,
    ):
        self.steps = 0
        self.warmup_epochs = warmup_epochs
        self.optimizer = optimizer

        # init scheduler for warmup
        self.warmup_scheduler = LinearWarmup(
            optimizer,
            warmup_epochs,
            start_lr,
            peak_lr,
        )
        # init scheduler for annealing
        self.lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cos_restart_dist,
            T_mult=cos_t_mult,
            eta_min=cos_eta_min,
            last_epoch=cos_last_epoch,
        )

    def step(self):
        """Make a step and also all it#s children should do a step"""
        if self.steps < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.lr_scheduler.step()
        self.steps += 1

    def get_lr(self):
        """Get learning rate at the current step"""
        if self.steps < self.warmup_epochs:
            return self.warmup_scheduler.get_lr()
        return self.lr_scheduler.get_lr()

    def state_dict(self):
        """Return state of the scheduler for later restoring"""
        return {
            "steps": self.steps,
            "warmup_epochs": self.warmup_epochs,
            "warmup_scheduler": self.warmup_scheduler.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Restore scheduler from previous state"""
        self.warmup_scheduler.load_state_dict(state_dict["warmup_scheduler"])
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.warmup_epochs = state_dict["warmup_epochs"]
        self.steps = state_dict["steps"]


if __name__ == '__main__':
    """Check the visualization of the learning rate"""
    import matplotlib.pyplot as plt

    x = []
    model = torch.nn.Linear(10, 10)
    opt = Adam(model.parameters(), lr=0.1)
    lr = LinearWarmupCosineAnnealingWarmRestartsLR(opt, 10, 1e-7, 0.1, 30, 1, 0, -1)
    for _ in range(100):
        x.append(lr.get_lr()[0])
        lr.step()
    x.append(lr.get_lr()[0])
    plt.plot(x, color="red")

    plt.show()
