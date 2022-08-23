import json

from pytorch_lightning.cli import LightningCLI

from rindti.data import DTIDataModule
from rindti.layers.encoder.graph_encoder import GraphEncoder
from rindti.models.dti import ClassificationModel
from rindti.utils.cli import namespace_to_dict


class MyLightningCLI(LightningCLI):
    """Default argss"""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.drug_config", "model.drug_encoder.inputs", apply_on="instantiate", compute_fn=json.dump
        )
        parser.link_arguments(
            "data.prot_config", "model.prot_encoder.inputs", apply_on="instantiate", compute_fn=json.dump
        )
        parser.link_arguments("data.prot_config", "model.mlp.inputs", apply_on="instantiate", compute_fn=json.dump)
        parser.link_arguments("model.feat_dim", "model.mlp.input_dim", apply_on="instantiate", compute_fn=json.dump)
        parser.set_defaults({"model.prot_encoder": GraphEncoder.default_config()})
        parser.set_defaults({"model.drug_encoder": GraphEncoder.default_config()})


if __name__ == "__main__":
    cli = MyLightningCLI(
        model_class=ClassificationModel, datamodule_class=DTIDataModule, run=False, save_config_callback=None
    )
    cli.trainer.logger.experiment.config.update(namespace_to_dict(cli.config))
    cli.trainer.fit(cli.model, cli.datamodule)
