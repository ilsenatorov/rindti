from typing import Tuple, Union

import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import (explained_variance,
                                                  mean_absolute_error)
from torch.optim import SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..layers import (DiffPoolNet, GINConvNet, PrecalculatedNet,
                      SequenceEmbedding)
from ..utils import remove_arg_prefix
from ..utils.data import TwoGraphData
from .base_model import BaseModel

embed_pairs = {
    'ginconvnet': GINConvNet,
    'diffpoolnet': DiffPoolNet,
    'precalculated': PrecalculatedNet,
    'sequence': SequenceEmbedding,
}


class SiameseModel(BaseModel):
    '''
    Used to combine different embeddings and MLPs into a single model
    :param feat_method: Feature concatenation method
    :param drug_embed_dim: Size of drug embedding
    :param prot_embed_dim: Size of prot embedding
    :param dataset: Dataset object which handles data loading
    '''

    def __init__(self, **kwargs):
        '''
        :param kwargs: from argparse
        '''
        dataset = kwargs.pop('dataset')
        super().__init__(dataset)
        self.save_hyperparameters()
        embed = kwargs.pop('embed')
        embedder = embed_pairs[embed]
        self.prot_graph = embedder(**kwargs)

    def _set_embedders(self, kwargs):
        '''
        Create the embedding modules
        :param kwargs: Kwargs from __init__
        '''
        embed = kwargs.pop('embed')
        embedder = embed_pairs[embed]
        self.prot_graph = embedder(**kwargs)

    def embed(self, data: Union[TwoGraphData, dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed the protein and drug given by an interaction pair

        Args:
            data (Union[TwoGraphData, dict]): Data entry
            who (str): Drug or protein

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: drug and protein embeddings
        """
        if not isinstance(data, dict):
            data = vars(data)
            data = remove_arg_prefix('a_', data)
        embed = self.prot_graph(**data, batch=None, x_batch=None)
        return embed.detach().numpy().reshape(-1)

    def forward(self, data):
        # Pass the respective inputs to their submodules
        datavars = vars(data)
        a_data = remove_arg_prefix('a_', datavars)
        b_data = remove_arg_prefix('b_', datavars)
        a_embed = self.prot_graph(**a_data, batch=data.a_x_batch)
        b_embed = self.prot_graph(**b_data, batch=data.b_x_batch)
        return F.cosine_similarity(a_embed, b_embed)

    def shared_step(self, data):
        '''
        This step is the same for train, val and test
        :param data: ProteinData batch object
        :returns: dict of accuracy metrics (has to contain 'loss')
        '''
        output = self.forward(data)
        labels = data.label.float()
        loss = F.mse_loss(output, labels)
        return {
            'loss': loss,
            'mae': mean_absolute_error(output, labels),
            'expvar': explained_variance(output, labels)
        }

    def configure_optimizers(self):
        '''
        Configure the optimizer/s.
        Relies on initially saved hparams to contain learning rates, weight decays etc
        '''
        optimiser = {'adamw': AdamW, 'sgd': SGD, 'rmsprop': RMSprop}[self.hparams.optimiser]
        optimiser = AdamW(params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimiser,
                factor=self.hparams.reduce_lr_factor,
                patience=self.hparams.reduce_lr_patience,
                verbose=True
            ),
            'monitor': 'val_loss',
        }
        return [optimiser], [lr_scheduler]
