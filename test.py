from pytorch_lightning.cli import LightningCLI

from rindti.data import ProteinDataModule
from rindti.models.pretrain.denoise import DenoiseModel
from rindti.utils.cli import namespace_to_dict

if __name__ == "__main__":
    cli = LightningCLI(
        model_class=DenoiseModel, datamodule_class=ProteinDataModule, run=False, save_config_callback=None
    )
    cli.trainer.logger.experiment.config.update(namespace_to_dict(cli.config))
    # cli.trainer.logger.watch(cli.model, log_freq=5000)
    cli.trainer.fit(cli.model, cli.datamodule)
