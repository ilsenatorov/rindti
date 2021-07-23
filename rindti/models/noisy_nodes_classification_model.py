from argparse import ArgumentParser
from copy import deepcopy
from math import ceil
from typing import Union
from rindti.utils.data import TwoGraphData

import numpy as np
import torch
import torch.nn.functional as F
from torch._C import device
from torch.nn import Embedding
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef

from ..layers import (MLP, ChebConvNet, DiffPoolNet, GINConvNet, GMTNet,
                      MeanPool, NoneNet, SequenceEmbedding, GatConvNet)
from ..utils import remove_arg_prefix
from .base_model import BaseModel
from .graphlog_model import GraphLogModel

node_embedders = {'ginconv': GINConvNet,
                  'chebconv': ChebConvNet,
                  'gatconv': GatConvNet,
                  'none': NoneNet}
poolers = {'gmt': GMTNet,
           'diffpool': DiffPoolNet,
           'mean': MeanPool}


class NoisyNodesModel(BaseModel):

    def corrupt_features(self, features: torch.Tensor, frac: float) -> torch.Tensor:
        num_feat = features.size(0)
        num_node_types = int(features.max())
        num_corrupt_nodes = ceil(num_feat * frac)
        corrupt_idx = np.random.choice(range(num_feat), num_corrupt_nodes, replace=False)
        corrupt_features = torch.tensor(
            np.random.choice(range(num_node_types),
                             num_corrupt_nodes, replace=True),
            dtype=torch.long, device=self.device)
        features[corrupt_idx] = corrupt_features
        return features, corrupt_idx

    def corrupt_data(self, orig_data: Union[TwoGraphData, dict],
                     prot_frac: float = 0.05,
                     drug_frac: float = 0.05) -> TwoGraphData:
        # sourcery skip: extract-duplicate-method
        data = deepcopy(orig_data)
        if prot_frac > 0:
            prot_feat, prot_idx = self.corrupt_features(data['prot_x'], prot_frac)
            data['prot_x'] = prot_feat
            data['prot_cor_idx'] = prot_idx
        if drug_frac > 0:
            drug_feat, drug_idx = self.corrupt_features(data['drug_x'], drug_frac)
            data['drug_x'] = drug_feat
            data['drug_cor_idx'] = drug_idx
        return data

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

        self.prot_pred = node_embedders[prot_param['node_embed']](prot_param['hidden_dim'], 20, num_layers=3, hidden_dim=32)
        self.drug_pred = node_embedders[drug_param['node_embed']](drug_param['hidden_dim'], 30, num_layers=3, hidden_dim=32)

    def forward(self, data):
        datavars = vars(data)
        prot_data = remove_arg_prefix('prot_', datavars)
        drug_data = remove_arg_prefix('drug_', datavars)
        prot_data['x'] = self.prot_feat_embed(prot_data['x'])
        drug_data['x'] = self.drug_feat_embed(drug_data['x'])
        prot_data['x'] = self.prot_node_embed(**prot_data)
        drug_data['x'] = self.drug_node_embed(**drug_data)
        prot_embed = self.prot_pool(**prot_data)
        drug_embed = self.drug_pool(**drug_data)

        joint_embedding = self.merge_features(drug_embed, prot_embed)
        logit = self.mlp(joint_embedding)
        prot_pred = self.prot_pred(**prot_data)
        drug_pred = self.drug_pred(**drug_data)
        return torch.sigmoid(logit), prot_pred, drug_pred

    def shared_step(self, data):
        '''
        This step is the same for train, val and test
        :param data: ProteinData batch object
        :returns: dict of accuracy metrics (has to contain 'loss')
        '''
        cor_data = self.corrupt_data(data, self.hparams.prot_frac, self.hparams.drug_frac)
        output, prot_pred, drug_pred = self.forward(cor_data)
        labels = data.label.unsqueeze(1)
        if self.hparams.weighted:
            weight = 1/torch.sqrt(data.prot_count * data.drug_count)
            loss = F.binary_cross_entropy(output, labels.float(), weight=weight.unsqueeze(1))
        else:
            loss = F.binary_cross_entropy(output, labels.float())
        prot_idx = cor_data.prot_cor_idx
        drug_idx = cor_data.drug_cor_idx
        prot_loss = F.cross_entropy(prot_pred[prot_idx], data['prot_x'][prot_idx])
        drug_loss = F.cross_entropy(drug_pred[drug_idx], data['drug_x'][drug_idx])
        t = (output > 0.5).float()
        acc = accuracy(t, labels)
        try:
            _auroc = auroc(t, labels)
        except Exception as e:
            _auroc = torch.tensor(np.nan, device=self.device)
        _mc = matthews_corrcoef(output, labels.squeeze(1), num_classes=2)
        return {
            "loss": loss + self.hparams.prot_alpha * prot_loss + self.hparams.drug_alpha * drug_loss,
            "prot_loss": prot_loss,
            "drug_loss": drug_loss,
            "pred_loss": loss,
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
        tmp_parser.add_argument('--drug_node_embed', type=str, default='ginconv')
        tmp_parser.add_argument('--prot_node_embed', type=str, default='ginconv')
        tmp_parser.add_argument('--prot_pool', type=str, default='gmt')
        tmp_parser.add_argument('--drug_pool', type=str, default='gmt')

        args = tmp_parser.parse_known_args()[0]
        prot_node_embed = node_embedders[args.prot_node_embed]
        drug_node_embed = node_embedders[args.drug_node_embed]
        prot_pool = poolers[args.prot_pool]
        drug_pool = poolers[args.drug_pool]
        prot = parser.add_argument_group('Prot', prefix='--prot_')
        drug = parser.add_argument_group('Drug', prefix='--drug_')
        drug.add_argument('alpha', default=0.1, type=float, help='Drug node loss factor')
        drug.add_argument('frac', default=0.05, type=float, help='Proportion of drug nodes to corrupt')
        drug.add_argument('node_embed', default='chebconv')
        drug.add_argument('node_embed_dim', default=16, type=int, help='Size of atom element embedding')
        prot.add_argument('alpha', default=0.1, type=float, help='Prot node loss factor')
        prot.add_argument('frac', default=0.05, type=float, help='Proportion of prot nodes to corrupt')
        prot.add_argument('node_embed', default='chebconv')
        prot.add_argument('node_embed_dim', default=16, type=int, help='Size of aminoacid embedding')

        prot_node_embed.add_arguments(prot)
        drug_node_embed.add_arguments(drug)
        prot_pool.add_arguments(prot)
        drug_pool.add_arguments(drug)
        return parser
