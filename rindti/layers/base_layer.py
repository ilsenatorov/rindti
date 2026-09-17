from torch import nn


class BaseLayer(nn.Module):
    """Base class for all layers."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError()


def interlayer_activations(count: int, dropout: float = 0.0) -> nn.ModuleList:
    """One PReLU (+ optional dropout) to apply after each non-final convolution.

    Without this, a stack of graph convolutions is a composition of linear maps.
    ``ChebConvNet`` was measurably *exactly* affine - three layers collapsed to a single
    linear operator, so it had no more capacity than one ``ChebConv`` - and ``GatConvNet``
    and ``TransformerNet`` were non-linear only through their attention softmax.
    ``GINConvNet`` was unaffected because its non-linearity lives inside the ``GINConv``
    MLP, which made a ``node.module`` ablation a comparison between a real GNN and a
    linear model.

    A separate PReLU per position rather than one shared module, matching the per-layer
    PReLU inside ``GINConvNet``: the slope is learnable, so sharing it would couple the
    layers.

    Args:
        count: number of non-final convolutions to follow.
        dropout: feature dropout applied after the activation. ``model.<tower>.node.dropout``
            used to be accepted and silently discarded by every module except
            ``TransformerNet``, where it meant *attention* dropout instead.
    """
    return nn.ModuleList(
        [nn.Sequential(nn.PReLU(), nn.Dropout(dropout)) if dropout else nn.PReLU() for _ in range(count)]
    )
