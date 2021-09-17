import random
from argparse import ArgumentParser
from copy import deepcopy
from math import ceil
from typing import Tuple

import torch
from torch_geometric.data import Data

from ..utils import MyArgParser
from .base_model import BaseModel, node_embedders, poolers
from .encoder import Encoder

# NCE loss between graphs and prototypes


class GraphLogModel(BaseModel):
    """Work in progress
    https://github.com/DeepGraphLearning/GraphLoG
    https://arxiv.org/pdf/2106.04113.pdf
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(return_nodes=True, **kwargs)
        self.proto = []
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(kwargs["hidden_dim"], 128), torch.nn.ReLU(), torch.nn.Linear(128, kwargs["hidden_dim"])
        )

    def mask_nodes(self, batch: Data) -> Data:
        """Mask the nodes according to self.mask_rate

        Args:
            batch (Data): Graph batch

        Returns:
            Data: Masked graph batch
        """
        masked_node_indices = []
        # select indices of masked nodes
        for i in range(batch["batch"][-1] + 1):
            idx = torch.nonzero((batch["batch"] == i).float()).squeeze(-1)
            num_node = idx.shape[0]
            sample_size = ceil(num_node * self.hparams.mask_rate)
            masked_node_idx = random.sample(idx.tolist(), sample_size)
            masked_node_idx.sort()
            masked_node_indices += masked_node_idx

        batch["masked_node_indices"] = torch.tensor(masked_node_indices, device=self.device)

        # mask nodes' features
        for node_idx in masked_node_indices:
            batch["x"][node_idx] = torch.zeros_like(batch["x"][node_idx])

        return batch

    def intra_NCE_loss(self, node_reps, node_modify_reps, batch, tau=0.04, epsilon=1e-6):
        """Loss between nodes"""
        node_reps_norm = torch.norm(node_reps, dim=1).unsqueeze(-1)
        node_modify_reps_norm = torch.norm(node_modify_reps, dim=1).unsqueeze(-1)
        sim = torch.mm(node_reps, node_modify_reps.t()) / (
            torch.mm(node_reps_norm, node_modify_reps_norm.t()) + epsilon
        )
        exp_sim = torch.exp(sim / tau)

        mask = torch.stack([(batch.batch == i).float() for i in batch.batch.tolist()], dim=1).to(self.device)
        exp_sim_mask = exp_sim * mask
        exp_sim_all = torch.index_select(exp_sim_mask, 1, batch.masked_node_indices)
        exp_sim_positive = torch.index_select(exp_sim_all, 0, batch.masked_node_indices)
        positive_ratio = exp_sim_positive.sum(0) / (exp_sim_all.sum(0) + epsilon)

        return -torch.log(positive_ratio).sum() / batch.masked_node_indices.shape[0]

    def inter_NCE_loss(self, graph_reps, graph_modify_reps, tau=0.04, epsilon=1e-6):
        """Loss between graphs"""
        graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
        graph_modify_reps_norm = torch.norm(graph_modify_reps, dim=1).unsqueeze(-1)
        sim = torch.mm(graph_reps, graph_modify_reps.t()) / (
            torch.mm(graph_reps_norm, graph_modify_reps_norm.t()) + epsilon
        )
        exp_sim = torch.exp(sim / tau)

        mask = torch.eye(graph_reps.shape[0], device=self.device)
        positive = (exp_sim * mask).sum(0)
        negative = (exp_sim * (1 - mask)).sum(0)
        positive_ratio = positive / (positive + negative + epsilon)

        return -torch.log(positive_ratio).sum() / graph_reps.shape[0]

    # NCE loss for global-local mutual information maximization

    def gl_NCE_loss(self, node_reps, graph_reps, batch, tau=0.04, epsilon=1e-6):
        """Don't even know what it's for"""
        node_reps_norm = torch.norm(node_reps, dim=1).unsqueeze(-1)
        graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
        sim = torch.mm(node_reps, graph_reps.t()) / (torch.mm(node_reps_norm, graph_reps_norm.t()) + epsilon)
        exp_sim = torch.exp(sim / tau)

        mask = torch.stack([(batch == i).float() for i in range(graph_reps.shape[0])], dim=1)
        positive = exp_sim * mask
        negative = exp_sim * (1 - mask)
        positive_ratio = positive / (positive + negative.sum(0).unsqueeze(0) + epsilon)

        return -torch.log(positive_ratio + (1 - mask)).sum() / node_reps.shape[0]

    def proto_NCE_loss(self, graph_reps, tau=0.04, epsilon=1e-6):
        """Prototype loss"""
        # similarity for original and modified graphs
        graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
        exp_sim_list = []
        mask_list = []
        NCE_loss = 0

        for i in range(len(self.proto) - 1, -1, -1):
            tmp_proto = self.proto[i]
            proto_norm = torch.norm(tmp_proto, dim=1).unsqueeze(-1)

            sim = torch.mm(graph_reps, tmp_proto.t()) / (torch.mm(graph_reps_norm, proto_norm.t()) + epsilon)
            exp_sim = torch.exp(sim / tau)

            if i != (len(self.proto) - 1):
                # apply the connection mask
                exp_sim_last = exp_sim_list[-1]
                idx_last = torch.argmax(exp_sim_last, dim=1).unsqueeze(-1)
                connection = self.proto_connection[i]
                connection_mask = (connection.unsqueeze(0) == idx_last.float()).float()
                exp_sim = exp_sim * connection_mask

                # define NCE loss between prototypes from consecutive layers
                upper_proto = self.proto[i + 1]
                upper_proto_norm = torch.norm(upper_proto, dim=1).unsqueeze(-1)
                proto_sim = torch.mm(tmp_proto, upper_proto.t()) / (
                    torch.mm(proto_norm, upper_proto_norm.t()) + epsilon
                )
                proto_exp_sim = torch.exp(proto_sim / tau)

                proto_positive_list = [proto_exp_sim[j, connection[j].long()] for j in range(proto_exp_sim.shape[0])]
                proto_positive = torch.stack(proto_positive_list, dim=0)
                proto_positive_ratio = proto_positive / (proto_exp_sim.sum(1) + epsilon)
                NCE_loss += -torch.log(proto_positive_ratio).mean()

            mask = (exp_sim == exp_sim.max(1)[0].unsqueeze(-1)).float()

            exp_sim_list.append(exp_sim)
            mask_list.append(mask)

        # define NCE loss between graph embedding and prototypes
        for i in range(len(self.proto)):
            exp_sim = exp_sim_list[i]
            mask = mask_list[i]

            positive = exp_sim * mask
            negative = exp_sim * (1 - mask)
            positive_ratio = positive.sum(1) / (positive.sum(1) + negative.sum(1) + epsilon)
            NCE_loss += -torch.log(positive_ratio).mean()

        return torch.tensor(NCE_loss, device=self.device, dtype=torch.float32)

    def update_proto_lowest(self, graph_reps, decay_ratio=0.7, epsilon=1e-6):
        """Update lowest prototypes"""
        graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
        proto_norm = torch.norm(self.proto[0], dim=1).unsqueeze(-1)
        sim = torch.mm(graph_reps, self.proto[0].t()) / (torch.mm(graph_reps_norm, proto_norm.t()) + epsilon)

        # update states of prototypes
        mask = (sim == sim.max(1)[0].unsqueeze(-1)).float()
        cnt = mask.sum(0)
        self.proto_state[0] = (self.proto_state[0] + cnt).detach()

        # update prototypes
        batch_cnt = mask.t() / (cnt.unsqueeze(-1) + epsilon)
        batch_mean = torch.mm(batch_cnt, graph_reps)
        self.proto[0] = (
            self.proto[0] * (cnt == 0).float().unsqueeze(-1)
            + (self.proto[0] * decay_ratio + batch_mean * (1 - decay_ratio)) * (cnt != 0).float().unsqueeze(-1)
        ).detach()

    def init_proto_lowest(self, num_iter=5):
        """Intitalise lowest prototypes"""
        self.eval()
        for _ in range(num_iter):
            for step, batch in enumerate(self.train_dataloader()):
                # get node and graph representations
                batch = batch.to(self.device)
                graph_reps, node_reps = self.encoder(batch)

                # feature projection
                graph_reps_proj = self.proj(graph_reps)

                # update prototypes
                self.update_proto_lowest(graph_reps_proj)

        idx = torch.nonzero((self.proto_state[0] >= 2).float()).squeeze(-1)
        return torch.index_select(self.proto[0], 0, idx)

    def init_proto(self, index, num_iter=20):
        """Initialise prototypes"""
        proto_connection = torch.zeros(self.proto[index - 1].shape[0], device=self.device)

        for iter in range(num_iter):
            for i in range(self.proto[index - 1].shape[0]):
                # update the closest prototype
                sim = torch.mm(self.proto[index], self.proto[index - 1][i, :].unsqueeze(-1)).squeeze(-1)
                idx = torch.argmax(sim)
                if iter == (num_iter - 1):
                    self.proto_state[index][idx] = 1
                proto_connection[i] = idx
                self.proto[index][idx, :] = self.proto[index][idx, :] * self.hparams.decay_ratio + self.proto[
                    index - 1
                ][i, :] * (1 - self.hparams.decay_ratio)

                # penalize rivalndarray
                sim[idx] = 0
                rival_idx = torch.argmax(sim)
                self.proto[index][rival_idx, :] = self.proto[index][rival_idx, :] * (
                    2 - self.hparams.decay_ratio
                ) - self.proto[index - 1][i, :] * (1 - self.hparams.decay_ratio)

        indices = torch.nonzero(self.proto_state[index]).squeeze(-1)
        proto_selected = torch.index_select(self.proto[index], 0, indices)
        for i in range(indices.shape[0]):
            idx = indices[i]
            idx_connection = torch.nonzero((proto_connection == idx.float()).float()).squeeze(-1)
            proto_connection[idx_connection] = i

        return proto_selected, proto_connection

    def embed_batch(self, orig_batch: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed a single batch (normal or masked doesn't matter)"""
        batch = deepcopy(orig_batch)
        graph_reps, node_reps = self.encoder(batch)
        node_reps = self.proj(node_reps)
        graph_reps = self.proj(graph_reps)
        return node_reps, graph_reps

    def forward(self, batch: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate node and graph reps for normal and masked batch"""
        batch_modify = deepcopy(batch)
        batch_modify = self.mask_nodes(batch)
        node_reps, graph_reps = self.embed_batch(batch)
        node_modify_reps, graph_modify_reps = self.embed_batch(batch_modify)
        return node_reps, node_modify_reps, graph_reps, graph_modify_reps

    def shared_step(self, batch: Data, proto: bool = True) -> dict:
        """Calculate all the losses"""
        (
            node_reps_proj,
            node_modify_reps_proj,
            graph_reps_proj,
            graph_modify_reps_proj,
        ) = self.forward(batch)
        NCE_intra_loss = self.intra_NCE_loss(node_reps_proj, node_modify_reps_proj, batch)
        NCE_inter_loss = self.inter_NCE_loss(graph_reps_proj, graph_modify_reps_proj)
        if proto:
            NCE_proto_loss = self.proto_NCE_loss(graph_reps_proj)
        else:
            NCE_proto_loss = torch.tensor(10, dtype=float)
        NCE_loss = (
            self.hparams.alpha * NCE_intra_loss
            + self.hparams.beta * NCE_inter_loss
            + self.hparams.gamma * NCE_proto_loss
        )
        return {
            "loss": NCE_loss,
            "NCE_intra_loss": NCE_intra_loss.detach(),
            "NCE_inter_loss": NCE_inter_loss.detach(),
            "NCE_proto_loss": NCE_proto_loss.detach(),
        }

    def training_step(self, data: Data, data_idx: int):
        """Training step"""
        if self.trainer.current_epoch == 0:
            ss = self.shared_step(data, proto=False)
        else:
            ss = self.shared_step(data, proto=True)
        for key, value in ss.items():
            self.log("train_" + key, value)
        return ss

    def training_epoch_end(self, outputs: dict):
        """Update prototypes and then do normal stuff"""
        if self.trainer.current_epoch == 0:
            self.proto = [
                torch.rand((self.hparams.num_proto, self.hparams.hidden_dim), device=self.device)
                for i in range(self.hparams.hierarchy)
            ]
            self.proto_state = [
                torch.zeros(self.hparams.num_proto, device=self.device) for i in range(self.hparams.hierarchy)
            ]
            self.proto_connection = []
            tmp_proto = self.init_proto_lowest()
            self.proto[0] = tmp_proto
            for i in range(1, self.hparams.hierarchy):
                print("Initialize prototypes: layer ", i + 1)
                tmp_proto, tmp_proto_connection = self.init_proto(i)
                self.proto[i] = tmp_proto
                self.proto_connection.append(tmp_proto_connection)

        super().training_epoch_end(outputs)

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
        parser.add_argument("--feat_embed_dim", default=32, type=int)
        parser.add_argument("--decay_ratio", default=0.5, type=float)
        parser.add_argument("--mask_rate", default=0.3, type=float)
        parser.add_argument("--alpha", default=1.0, type=float)
        parser.add_argument("--beta", default=1.0, type=float)
        parser.add_argument("--gamma", default=0.1, type=float)
        parser.add_argument("--num_proto", default=8, type=int)
        parser.add_argument("--hierarchy", default=3, type=int)
        pooler_args = parser.add_argument_group("Pool", prefix="--")
        node_embed_args = parser.add_argument_group("Node embedding", prefix="--")
        node_embed.add_arguments(node_embed_args)
        pool.add_arguments(pooler_args)
        return parser
