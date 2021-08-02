import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import explained_variance, mean_absolute_error
from torch.optim import SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..layers import MLP, DiffPoolNet, GINConvNet, PrecalculatedNet, SequenceEmbedding
from ..utils import remove_arg_prefix
from .base_model import BaseModel

embed_pairs = {
    "ginconvnet": GINConvNet,
    "diffpoolnet": DiffPoolNet,
    "precalculated": PrecalculatedNet,
    "sequence": SequenceEmbedding,
}


class RegressionModel(BaseModel):
    """
    Used to combine different embeddings and MLPs into a single model
    :param feat_method: Feature concatenation method
    :param drug_embed_dim: Size of drug embedding
    :param prot_embed_dim: Size of prot embedding
    :param dataset: Dataset object which handles data loading
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: from argparse
        """
        dataset = kwargs.pop("dataset")
        super().__init__(dataset)
        self.save_hyperparameters()
        self._determine_feat_method(kwargs)
        self._set_embedders(kwargs)
        self._set_mlp(kwargs)

    def _set_embedders(self, kwargs):
        """
        Create the embedding modules
        :param kwargs: Kwargs from __init__
        """
        drug_embed = kwargs.pop("drug_embed")
        drug_param = remove_arg_prefix("drug_", kwargs)
        drug_embedder = embed_pairs[drug_embed]
        self.drug_graph = drug_embedder(**drug_param)
        prot_embed = kwargs.pop("prot_embed")
        prot_param = remove_arg_prefix("prot_", kwargs)
        prot_embedder = embed_pairs[prot_embed]
        self.prot_graph = prot_embedder(**prot_param)

    def _set_mlp(self, kwargs):
        """
        Create the MLP module
        :param kwargs: Kwargs from __init__
        """
        mlp_args = remove_arg_prefix("mlp_", kwargs)
        self.mlp = MLP(**mlp_args, input_dim=self.embed_dim, out_dim=1)

    def forward(self, data):
        """
        Use the original ProteinData batched entry:
        1. pass them to respective embedders
        2. combine them using feat_method
        3. Use MLP to predict the result
        :param data: ProteinData batch object
        :returns: dict of all losses
        """
        # Pass the respective inputs to their submodules
        datavars = vars(data)
        prot_data = remove_arg_prefix("prot_", datavars)
        drug_data = remove_arg_prefix("drug_", datavars)
        drug_embed = self.drug_graph(**drug_data, batch=data.drug_x_batch)
        prot_embed = self.prot_graph(**prot_data, batch=data.prot_x_batch)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        logit = self.mlp(joint_embedding)
        return logit

    def shared_step(self, data):
        """
        This step is the same for train, val and test
        :param data: ProteinData batch object
        :returns: dict of accuracy metrics (has to contain 'loss')
        """
        output = self.forward(data)
        labels = data.label.unsqueeze(1).float()
        loss = F.mse_loss(output, labels)
        return {
            "loss": loss,
            "mae": mean_absolute_error(output, labels),
            "expvar": explained_variance(output, labels),
        }

    def configure_optimizers(self):
        """
        Configure the optimizer/s.
        Relies on initially saved hparams to contain learning rates, weight decays etc
        """
        optimiser = {"adamw": AdamW, "sgd": SGD, "rmsprop": RMSprop}[self.hparams.optimiser]
        optimiser = AdamW(
            [
                {
                    "params": self.prot_graph.parameters(),
                    "lr": self.hparams.prot_lr,
                    "weight_decay": self.hparams.prot_weight_decay,
                },
                {
                    "params": self.drug_graph.parameters(),
                    "lr": self.hparams.drug_lr,
                    "weight_decay": self.hparams.drug_weight_decay,
                },
                {
                    "params": self.mlp.parameters(),
                    "lr": self.hparams.mlp_lr,
                    "weight_decay": self.hparams.mlp_weight_decay,
                },
            ]
        )
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimiser,
                factor=self.hparams.reduce_lr_factor,
                patience=self.hparams.reduce_lr_patience,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [optimiser], [lr_scheduler]
