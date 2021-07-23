from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.data.dataloader import DataLoader

from rindti.models import ClassificationModel, RegressionModel, NoisyNodesModel
from rindti.utils.data import Dataset
from rindti.utils.transforms import GnomadTransformer, RandomTransformer


def train(**kwargs):
    if kwargs['transformer'] != 'none':
        transform = {"gnomad": GnomadTransformer, "random": RandomTransformer}[kwargs['transformer']].from_pickle(
            kwargs['transformer_pickle'], max_num_mut=kwargs['max_num_mut'])
    else:
        transform = None
    train = Dataset(kwargs['data'], split='train', transform=transform)
    val = Dataset(kwargs['data'], split='val')
    test = Dataset(kwargs['data'], split='test')

    kwargs.update(dict(
        prot_max_nodes=train.info['prot_max_nodes'],
        drug_max_nodes=train.info['drug_max_nodes'],
        prot_feat_dim=train.info['prot_feat_dim'],
        drug_feat_dim=train.info['drug_feat_dim']
    ))
    model_name = kwargs['model_name']
    if not model_name:
        model_name = kwargs['model']
    logger = TensorBoardLogger('tb_logs',
                               name=model_name,
                               default_hp_metric=False)
    callbacks = [
        ModelCheckpoint(monitor='val_loss',
                        save_top_k=3,
                        mode='min'),
        EarlyStopping(monitor='val_loss',
                      patience=kwargs['early_stop_patience'],
                      mode='min')
    ]
    trainer = Trainer(gpus=kwargs['gpus'],
                      callbacks=callbacks,
                      logger=logger,
                      gradient_clip_val=kwargs['gradient_clip_val'],
                      stochastic_weight_avg=True,
                      )
    if kwargs['model'] == 'regression':
        model = RegressionModel(**kwargs)
    elif kwargs['model'] == 'classification':
        model = ClassificationModel(**kwargs)
    elif kwargs['model'] == 'noisynodes':
        model = NoisyNodesModel(**kwargs)
    else:
        raise ValueError('Unknown model type')
    print(model)
    dataloader_kwargs = {k: v for (k, v) in kwargs.items() if k in ['batch_size', 'num_workers']}
    dataloader_kwargs.update({'follow_batch': ['prot_x', 'drug_x']})
    train_dataloader = DataLoader(train, **dataloader_kwargs, shuffle=True)
    val_dataloader = DataLoader(val, **dataloader_kwargs, shuffle=False)
    test_dataloader = DataLoader(test, **dataloader_kwargs, shuffle=False)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == '__main__':

    import argparse

    from rindti.utils import MyArgParser

    parser = MyArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', type=str)
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers for data loading')
    parser.add_argument('--early_stop_patience', type=int, default=60, help='epochs with no improvement before stop')
    parser.add_argument('--feat_method', type=str, default='concatenate',
                        help='How to combine drug and protein embeddings')
    parser.add_argument('--prot_pretrained', type=str, default=None, help='Model of pretrained prot embedder')
    parser.add_argument('--model_name', type=str, default=None, help='Name under which to save the model')

    trainer = parser.add_argument_group('Trainer')
    model = parser.add_argument_group('Model')
    optim = parser.add_argument_group('Optimiser')
    transformer = parser.add_argument_group('Transformer')

    trainer.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    trainer.add_argument('--max_epochs', type=int, default=1000, help='Max number of epochs')
    trainer.add_argument('--model', type=str, default='classification', help='Type of model')
    trainer.add_argument('--weighted', type=bool, default=1, help='Whether to weight the data points')
    trainer.add_argument('--gradient_clip_val', type=float, default=10, help='Gradient clipping')

    model.add_argument('--mlp_hidden_dim', default=64, type=int, help='MLP hidden dims')
    model.add_argument('--mlp_dropout', default=0.2, type=float, help='MLP dropout')

    optim.add_argument('--optimiser', type=str, default='adamw', help='Optimisation algorithm')
    optim.add_argument('--momentum', type=float, default=0.3, help='Optimisation momentum (where applicable)')
    optim.add_argument('--lr', type=float, default=0.001, help='mlp learning rate')
    optim.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    optim.add_argument('--reduce_lr_patience', type=int, default=20,
                       help='epoch with no improvement before lr is reduced')
    optim.add_argument('--reduce_lr_factor', type=float, default=0.1,
                       help='factor for lr reduction (new_lr = factor*lr)')

    transformer.add_argument('--transformer', type=str, default='none', help='Type of transformer to apply')
    transformer.add_argument(
        '--transformer_pickle', type=str,
        default='../rins/results/prepare_transformer/onehot_simple_transformer.pkl',
        help='Location of transform pickle')
    transformer.add_argument("--max_num_mut", type=int, default=100, help='Max number of mutations per protein')

    parser = NoisyNodesModel.add_arguments(parser)

    args = parser.parse_args()
    argvars = vars(args)
    train(**argvars)
