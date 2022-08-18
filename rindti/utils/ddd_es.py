from pytorch_lightning.callbacks import EarlyStopping


class DeepDoubleDescentEarlyStopping(EarlyStopping):
    def __init__(self, kick_in=50, **kwargs):
        super().__init__(**kwargs)
        self.kick_in = kick_in

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.kick_in:
            return

        super().on_validation_end(trainer, pl_module)
        # logs = trainer.callback_metrics
        #
        # if trainer.fast_dev_run or not self._validate_condition_metric(logs):
        #     return
        #
        # current = logs.get(self.monitor)
