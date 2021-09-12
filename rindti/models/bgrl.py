from argparse import ArgumentParser
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from torch_geometric.data import Data

from ..utils import MyArgParser
from ..utils.data import TwoGraphData, mask_data
from .base_model import BaseModel, node_embedders, poolers


class EMA:
    """Exponential moving average"""

    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        """Update"""
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


def loss_fn(x, y):
    """Cosine similarity"""
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def update_moving_average(ema_updater, ma_model, current_model):
    """User ema updater on all the weights"""
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    """Sets grad as true or False"""
    for p in model.parameters():
        p.requires_grad = val


class Encoder(BaseModel):
    """Encodes nodes in a graph"""

    def __init__(self, **kwargs):
        super().__init__()
        self.feat_embed = self._get_feat_embed(kwargs)
        self.node_embed = self._get_node_embed(kwargs)
        self.pool = self._get_pooler(kwargs)

    def forward(self, x, edge_index, batch, **kwargs):
        """Forward pass"""
        x = self.feat_embed(x)
        x = self.node_embed(x, edge_index)
        embed = self.pool(x, edge_index, batch)
        return embed, x


def init_weights(m):
    """"""
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BGRLModel(BaseModel):
    """Bootrstrape Graph Representational learning"""

    def __init__(self, dropout=0.0, moving_average_decay=0.99, epochs=1000, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.student_encoder = Encoder(dropout=dropout, **kwargs)
        self.teacher_encoder = deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        self.student_node_predictor = nn.Sequential(
            nn.Linear(kwargs["hidden_dim"], kwargs["hidden_dim"]),
            nn.BatchNorm1d(kwargs["hidden_dim"], momentum=0.01),
            nn.PReLU(),
            nn.Linear(kwargs["hidden_dim"], kwargs["hidden_dim"]),
        )
        self.student_graph_predictor = nn.Sequential(
            nn.Linear(kwargs["hidden_dim"], kwargs["hidden_dim"]),
            nn.BatchNorm1d(kwargs["hidden_dim"], momentum=0.01),
            nn.PReLU(),
            nn.Linear(kwargs["hidden_dim"], kwargs["hidden_dim"]),
        )
        self.student_node_predictor.apply(init_weights)
        self.student_graph_predictor.apply(init_weights)

    def reset_moving_average(self):
        """"""
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        """"""
        assert self.teacher_encoder is not None, "teacher encoder has not been created yet"
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """Forward pass"""

        graph_student, node_student = self.student_encoder(**data)
        node_pred = self.student_node_predictor(node_student)
        graph_pred = self.student_graph_predictor(graph_student)
        with torch.no_grad():
            graph_teacher, node_teacher = self.teacher_encoder(**data)

        return graph_teacher, graph_pred, node_teacher, node_pred

    def shared_step(self, data: Data):
        """Shared step"""
        a = mask_data(data, self.hparams.frac).__dict__
        b = mask_data(data, self.hparams.frac).__dict__
        a_graph_teacher, a_graph_pred, a_node_teacher, a_node_pred = self.forward(a)
        b_graph_teacher, b_graph_pred, b_node_teacher, b_node_pred = self.forward(b)

        node_loss1 = loss_fn(a_node_pred, a_node_teacher.detach())
        node_loss2 = loss_fn(b_node_pred, b_node_teacher.detach())
        graph_loss1 = loss_fn(a_graph_pred, a_graph_teacher.detach())
        graph_loss2 = loss_fn(b_graph_pred, b_graph_teacher.detach())

        node_loss = (node_loss1 + node_loss2).mean()
        graph_loss = (graph_loss1 + graph_loss2).mean()
        self.update_moving_average()
        return {
            "loss": node_loss + self.hparams.alpha * graph_loss,
            "node_loss": node_loss.detach(),
            "graph_loss": graph_loss.detach(),
        }

    @staticmethod
    def add_arguments(parser: MyArgParser) -> MyArgParser:
        """Generate arguments for this module"""
        # Hack to find which embedding are used and add their arguments
        tmp_parser = ArgumentParser(add_help=False)
        tmp_parser.add_argument("--node_embed", type=str, default="ginconv")
        tmp_parser.add_argument("--pool", type=str, default="gmt")
        args = tmp_parser.parse_known_args()[0]

        node_embed = node_embedders[args.node_embed]
        pool = poolers[args.pool]
        parser.add_argument("--frac", default=0.1, type=float, help="Corruption percentage")
        parser.add_argument("--alpha", default=1.0, type=float)
        parser.add_argument("--feat_embed_dim", default=32, type=int)

        pooler_args = parser.add_argument_group("Pool", prefix="--")
        node_embed_args = parser.add_argument_group("Node embedding", prefix="--")
        node_embed.add_arguments(node_embed_args)
        pool.add_arguments(pooler_args)
        return parser
