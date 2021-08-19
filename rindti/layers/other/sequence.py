import torch.nn.functional as F
from torch import nn
from torch.functional import Tensor

from ..base_layer import BaseLayer


class SequenceEmbedding(BaseLayer):
    """Embed the sequence data

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32, **kwargs):
        super().__init__()
        # FIXME
        maxlen = 600
        out_dim = kwargs["embed_dim"]
        self.conv = nn.Conv1d(in_channels=maxlen, out_channels=16, kernel_size=8)
        self.fc = nn.Linear(16 * 18, out_dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward pass of the module

        Args:
            x (Tensor): Input features

        Returns:
            Tensor: Updated features
        """
        x = kwargs["x"]  # (batch_size, 600)
        batch_size = x.size(0)
        x = self.embedding(x)  # (batch_size, 600, 21)
        x = self.conv(x)  # (batch_size, 32, 21-kernel_size-1)
        x = x.view(batch_size, -1)  # (batch_size, 448)
        x = F.relu(self.fc(x))  # (batch_size, 64)
        return x
