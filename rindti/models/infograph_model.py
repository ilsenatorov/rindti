import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torch.nn.modules.sparse import Embedding

from rindti.models.base_model import BaseModel
from rindti.utils.data import TwoGraphData
from rindti.utils.utils import remove_arg_prefix

from ..layers import MutualInformation
from .noisy_nodes_model import NoisyNodesModel


class InfoGraph(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.feat_embed = Embedding()
        self.mi = MutualInformation(kwargs["hidden_dim"], kwargs["hidden_dim"])
        raise NotImplementedError()
