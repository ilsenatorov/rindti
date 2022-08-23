from pytorch_lightning.cli import LightningCLI

from rindti.data import DTIDataModule
from rindti.models.dti import ClassificationModel
from rindti.utils.cli import namespace_to_dict

if __name__ == "__main__":
    cli = LightningCLI(
        model_class=ClassificationModel, datamodule_class=DTIDataModule, run=False, save_config_callback=None
    )
    cli.trainer.logger.experiment.config.update(namespace_to_dict(cli.config))
    cli.trainer.fit(cli.model, cli.datamodule)
