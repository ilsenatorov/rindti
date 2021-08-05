import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_batch

from ..base_layer import BaseLayer


class SequenceEmbedding(BaseLayer):
    """
    Embed a protein sequence with nn.Embedding and Conv1d
    :param embed_dim: size of output embedding
    :param feat_dim: number of features
    """

    def __init__(self, **kwargs):
        super().__init__()
        seq_embed_dim = 25
        maxlen = 600
        out_dim = kwargs["embed_dim"]

        self.embedding = nn.Embedding(25, seq_embed_dim)
        self.conv = nn.Conv1d(in_channels=maxlen, out_channels=16, kernel_size=8)
        self.fc = nn.Linear(16 * 18, out_dim)

    def forward(self, **kwargs):
        x = kwargs["x"]  # (batch_size, 600)
        batch_size = x.size(0)
        x = self.embedding(x)  # (batch_size, 600, 21)
        x = self.conv(x)  # (batch_size, 32, 21-kernel_size-1)
        x = x.view(batch_size, -1)  # (batch_size, 448)
        x = F.relu(self.fc(x))  # (batch_size, 64)
        return x
