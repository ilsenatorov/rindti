from ..base_layer import BaseLayer
# from .. import BaseLayer


class PrecalculatedNet(BaseLayer):

    '''
    Fake embedding, assumes data is precalculated
    '''

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
