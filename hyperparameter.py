import warnings

warnings.filterwarnings("ignore")
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from rindti.data import DTIDataModule
from rindti.models import ClassificationModel, RegressionModel
from rindti.utils import read_config, recursive_apply

models = {"class": ClassificationModel, "reg": RegressionModel}

seed_everything(42)


def train(kwargs: dict, datamodule: DTIDataModule):
    """Does a single run."""
    logger = TensorBoardLogger(
        save_dir=tune.get_trial_dir(),
        name="",
        version=".",
        default_hp_metric=False,
    )

    callbacks = [
        TuneReportCallback(
            {
                "loss": "val_loss",
                "auroc": "val_AUROC",
                "acc": "val_Accuracy",
            },
            on="validation_end",
        )
    ]
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=25,
        enable_model_summary=False,
        enable_progress_bar=False,
        **kwargs["trainer"],
    )
    model = models[kwargs["model"]["module"]](**kwargs)
    trainer.fit(model, datamodule)


def list_to_tune(l):
    """Convert object to tune object if it is a list."""
    if isinstance(l, list):
        return tune.choice(l)
    else:
        return l


def tune_asha(configfile: str, num_samples: int = 1000, num_epochs: int = 100):
    """Tune hparams with ASHA"""
    config = read_config(configfile)
    config = recursive_apply(config, list_to_tune)
    datamodule = DTIDataModule(**config["datamodule"])
    datamodule.setup()
    datamodule.update_config(config)
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=20, reduction_factor=3)
    analysis = tune.run(
        tune.with_parameters(train, datamodule=datamodule),
        resources_per_trial={"cpu": 16, "gpu": 1},
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        config=config,
        local_dir="tb_logs",
        name="tune_asha",
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    with open("config/dti/best_hparams.yml", "w") as file:
        yaml.dump(analysis.best_config, file)


if __name__ == "__main__":
    from jsonargparse import CLI

    cli = CLI(tune_asha)
