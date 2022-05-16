import pytest
from pytorch_lightning import Trainer

from rindti.models import ClassificationModel
from rindti.utils import IterDict, read_config

CONFIG_FILE = "tests/configs/default_dti.yaml"


default_config = read_config(CONFIG_FILE)
all_configs = IterDict()(default_config)


class BaseTestModel:
    @pytest.mark.parametrize("config", all_configs)
    @pytest.mark.slow
    def test_full(self, config, dti_datamodule):
        dti_datamodule.setup()
        dti_datamodule.update_config(config)
        model = self.model_class(**config)
        trainer = Trainer(
            gpus=0,
            fast_dev_run=True,
            enable_checkpointing=False,
            logger=None,
        )
        trainer.fit(model, dti_datamodule)

    @pytest.mark.parametrize("config", all_configs)
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_full_gpu(self, config, dti_datamodule):
        dti_datamodule.setup()
        dti_datamodule.update_config(config)
        model = self.model_class(**config)
        trainer = Trainer(
            gpus=1,
            fast_dev_run=True,
            enable_checkpointing=False,
            logger=None,
        )
        trainer.fit(model, dti_datamodule)

    @pytest.mark.parametrize("config", all_configs)
    def test_shared(self, config, dti_datamodule):
        dti_datamodule.setup()
        dti_datamodule.update_config(config)
        model = self.model_class(**config)
        batch = next(iter(dti_datamodule.train_dataloader()))
        output = model.shared_step(batch)
        assert "loss" in output.keys()
        assert "preds" in output.keys()
        assert "labels" in output.keys()


class TestClassificationModel(BaseTestModel):
    model_class = ClassificationModel
