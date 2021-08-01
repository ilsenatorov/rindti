import torch
from pytorch_lightning import Trainer
from rindti.utils.data import PreTrainDataset
from torch_geometric.data import DataLoader, Data
from rindti.models import GraphLogModel
from torch.utils.data import random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import sys

dataset = PreTrainDataset(sys.argv[1])
logger = TensorBoardLogger('tb_logs',
                           name='graphlog',
                           default_hp_metric=False)
callbacks = [
    ModelCheckpoint(monitor='train_loss',
                    save_top_k=3,
                    mode='min'),
    EarlyStopping(monitor='train_loss',
                  patience=20,
                  mode='min')
]
trainer = Trainer(gpus=1,
                  callbacks=callbacks,
                  logger=logger,
                  gradient_clip_val=30,
                  max_epochs=11,
                  stochastic_weight_avg=True,
                  )
model = GraphLogModel(feat_dim=dataset.info['feat_dim'],
                      max_nodes=dataset.info['max_nodes'],
                      embed_dim=32,
                      optimiser='adam',
                      lr=0.001,
                      weight_decay=0.01,
                      reduce_lr_patience=10,
                      reduce_lr_factor=0.1,
                      hidden_dim=32)
dl = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
trainer.fit(model, train_dataloader=dl)
