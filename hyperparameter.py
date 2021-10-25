import sys

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from torch_geometric.loader import DataLoader

from rindti.data import DTIDataset
from rindti.models import ClassificationModel

torch.manual_seed(42)


def train(kwargs, data, num_epochs=100):
    """Run one training instance"""
    train_data, val_data = data
    kwargs.update(train_data.config)
    model = ClassificationModel(**kwargs)
    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=1,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "val_loss",
                    "auroc": "val_auroc",
                    "acc": "val_acc",
                },
                on="validation_end",
            )
        ],
    )
    dataloader_kwargs = {k: v for (k, v) in kwargs.items() if k in ["batch_size", "num_workers"]}
    dataloader_kwargs.update({"follow_batch": ["prot_x", "drug_x"]})
    train_dataloader = DataLoader(train_data, **dataloader_kwargs, shuffle=True)
    val_dataloader = DataLoader(val_data, **dataloader_kwargs, shuffle=False)
    trainer.fit(model, train_dataloader, val_dataloader)


def tune_asha(num_samples=1000, num_epochs=100):
    """Tune hparams with ASHA"""
    config = {
        "feat_method": "concat",
        "drug_hidden_dim": tune.choice([8, 16, 32, 64, 128]),
        "prot_hidden_dim": tune.choice([8, 16, 32, 64, 128]),
        "prot_pretrain": None,
        "drug_pretrain": None,
        "batch_size": 128,
        "num_workers": 16,
        "prot_node_embed": tune.choice(["ginconv", "chebconv", "gatconv", "transformer"]),
        "drug_node_embed": tune.choice(["ginconv", "chebconv", "gatconv", "transformer"]),
        "prot_pool": tune.choice(["gmt", "diffpool"]),
        "drug_pool": tune.choice(["gmt", "diffpool"]),
        "optimiser": "adamw",
        "lr": 0.001,
        "weight_decay": 0.001,
        "reduce_lr_factor": 0.1,
        "reduce_lr_patience": 15,
        "weighted": False,
        "prot_dropout": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "drug_dropout": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "prot_heads": 4,
        "drug_heads": 4,
        "prot_lr": 0.001,
        "drug_lr": 0.001,
        "monitor": "val_loss",
    }
    train_data = DTIDataset(sys.argv[1], split="train")
    val_data = DTIDataset(sys.argv[1], split="val").shuffle()

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=20, reduction_factor=3)

    reporter = CLIReporter(
        parameter_columns=[
            "prot_node_embed",
            "drug_node_embed",
            "prot_pool",
            "drug_pool",
            "prot_hidden_dim",
            "drug_hidden_dim",
        ],
        metric_columns=["auroc", "acc", "training_iteration"],
    )

    analysis = tune.run(
        tune.with_parameters(train, data=(train_data, val_data), num_epochs=num_epochs),
        resources_per_trial={"cpu": 16, "gpu": 1},
        metric="auroc",
        mode="max",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="tb_logs",
        name="tune_asha",
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    with open("best_hparams.yml", "w") as file:
        yaml.dump(analysis.best_config, file)


tune_asha()
