from pytorch_lightning.callbacks import EarlyStopping


class DeepDoubleDescentEarlyStopping(EarlyStopping):
    """Delayed early stopping with start counting epochs until ES later. This should prevent DeepDoubleDescent"""
    def __init__(self, kick_in=50, **kwargs):
        super().__init__(**kwargs)
        self.kick_in = kick_in

    def on_validation_end(self, trainer, pl_module):
        """Delay counting of epochs until ES"""
        if trainer.current_epoch < self.kick_in:
            return

        super().on_validation_end(trainer, pl_module)
