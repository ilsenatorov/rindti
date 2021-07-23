from argparse import ArgumentParser, _ArgumentGroup
from typing import Union

from pytorch_lightning import LightningModule


class BaseLayer(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def add_arguments(group: Union[ArgumentParser, _ArgumentGroup]):
        return group
