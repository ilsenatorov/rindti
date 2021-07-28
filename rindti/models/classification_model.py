from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch._C import device
from torch.nn import Embedding
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef

from ..layers import (MLP, ChebConvNet, DiffPoolNet, GatConvNet, GINConvNet,
                      GMTNet, MeanPool, NoneNet, SequenceEmbedding)
from ..utils import combine_parameters, remove_arg_prefix
from .base_model import BaseModel

node_embedders = {'ginconv': GINConvNet,
                  'chebconv': ChebConvNet,
                  'gatconv': GatConvNet,
                  'none': NoneNet}
poolers = {'gmt': GMTNet,
           'diffpool': DiffPoolNet,
           'mean': MeanPool}


class ClassificationModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._determine_feat_method(kwargs)
        drug_param = remove_arg_prefix('drug_', kwargs)
        prot_param = remove_arg_prefix('prot_', kwargs)
        # TODO fix hardcoded values
        self.prot_feat_embed = Embedding(20, kwargs['prot_node_embed_dim'])
        self.drug_feat_embed = Embedding(30, kwargs['drug_node_embed_dim'])
        self.prot_node_embed = node_embedders[prot_param['node_embed']](
            prot_param['node_embed_dim'], prot_param['hidden_dim'], **prot_param)
        self.drug_node_embed = node_embedders[drug_param['node_embed']](
            drug_param['node_embed_dim'], drug_param['hidden_dim'], **drug_param)
        self.prot_pool = poolers[prot_param['pool']](
            prot_param['hidden_dim'], prot_param['hidden_dim'], **prot_param)
        self.drug_pool = poolers[drug_param['pool']](
            drug_param['hidden_dim'], drug_param['hidden_dim'], **drug_param)
        mlp_param = remove_arg_prefix('mlp_', kwargs)
        self.mlp = MLP(**mlp_param, input_dim=self.embed_dim, out_dim=1)

    def forward(self, prot_x, drug_x, prot_edge_index, drug_edge_index, prot_batch, drug_batch):
        prot_x = self.prot_feat_embed(prot_x)
        drug_x = self.drug_feat_embed(drug_x)
        prot_x = self.prot_node_embed(prot_x, prot_edge_index, prot_batch)
        drug_x = self.drug_node_embed(drug_x, drug_edge_index, drug_batch)
        prot_embed = self.prot_pool(prot_x, prot_edge_index, prot_batch)
        drug_embed = self.drug_pool(drug_x, drug_edge_index, drug_batch)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        logit = self.mlp(joint_embedding)
        return torch.sigmoid(logit)

    def shared_step(self, data):
        '''
        This step is the same for train, val and test
        :param data: ProteinData batch object
        :returns: dict of accuracy metrics (has to contain 'loss')
        '''
        output = self.forward(
            data.prot_x,
            data.drug_x,
            data.prot_edge_index,
            data.drug_edge_index,
            data.prot_x_batch,
            data.drug_x_batch)
        labels = data.label.unsqueeze(1)
        if self.hparams.weighted:
            weight = 1/torch.sqrt(data.prot_count * data.drug_count)
            loss = F.binary_cross_entropy(output, labels.float(), weight=weight.unsqueeze(1))
        else:
            loss = F.binary_cross_entropy(output, labels.float())
        t = (output > 0.5).float()
        acc = accuracy(t, labels)
        try:
            _auroc = auroc(t, labels)
        except Exception as e:
            _auroc = torch.tensor(np.nan, device=self.device)
        _mc = matthews_corrcoef(output, labels.squeeze(1), num_classes=2)
        return {
            "loss": loss,
            'acc': acc,
            'auroc': _auroc,
            'matthews': _mc,
        }

    def training_epoch_end(self, outputs):
        '''
        What to log and save on train epoch end
        :param outputs: return of training_step in a tensor (dicts)
        '''
        entries = outputs[0].keys()
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            self.logger.experiment.add_scalar('train_epoch_' + i, val, self.current_epoch)
        self.logger.experiment.add_histogram("prot_feat_embed", combine_parameters(self.prot_feat_embed.parameters()), self.current_epoch)
        self.logger.experiment.add_histogram("drug_feat_embed", combine_parameters(self.drug_feat_embed.parameters()), self.current_epoch)
        self.logger.experiment.add_histogram("prot_node_embed", combine_parameters(self.prot_node_embed.parameters()), self.current_epoch)
        self.logger.experiment.add_histogram("drug_node_embed", combine_parameters(self.drug_node_embed.parameters()), self.current_epoch)
        self.logger.experiment.add_histogram("prot_pool", combine_parameters(self.prot_pool.parameters()), self.current_epoch)
        self.logger.experiment.add_histogram("drug_pool", combine_parameters(self.drug_pool.parameters()), self.current_epoch)

    def configure_optimizers(self):
        '''
        Configure the optimizer/s.
        Relies on initially saved hparams to contain learning rates, weight decays etc
        '''
        optimiser = {'adamw': AdamW, 'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop}[self.hparams.optimiser]
        optimiser = optimiser(params=self.parameters(),
                              lr=self.hparams.lr,
                              weight_decay=self.hparams.weight_decay)
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

    @staticmethod
    def add_arguments(parser):
        '''
        Add the arguments for the training
        '''
        # Hack to find which embedding are used and add their arguments
        tmp_parser = ArgumentParser(add_help=False)
        tmp_parser.add_argument('--drug_node_embed', type=str, default='chebconv')
        tmp_parser.add_argument('--prot_node_embed', type=str, default='chebconv')
        tmp_parser.add_argument('--prot_pool', type=str, default='gmt')
        tmp_parser.add_argument('--drug_pool', type=str, default='gmt')

        args = tmp_parser.parse_known_args()[0]
        prot_node_embed = node_embedders[args.prot_node_embed]
        drug_node_embed = node_embedders[args.drug_node_embed]
        prot_pool = poolers[args.prot_pool]
        drug_pool = poolers[args.drug_pool]
        prot = parser.add_argument_group('Prot', prefix='--prot_')
        drug = parser.add_argument_group('Drug', prefix='--drug_')
        prot.add_argument('node_embed', default='chebconv')
        prot.add_argument('node_embed_dim', default=16, type=int, help='Size of aminoacid embedding')
        drug.add_argument('node_embed', default='chebconv')
        drug.add_argument('node_embed_dim', default=16, type=int, help='Size of atom element embedding')

        prot_node_embed.add_arguments(prot)
        drug_node_embed.add_arguments(drug)
        prot_pool.add_arguments(prot)
        drug_pool.add_arguments(drug)
        return parser
