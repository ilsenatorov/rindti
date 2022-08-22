import pytest
from pytorch_lightning import Trainer

from rindti.models import ClassificationModel, RegressionModel
from rindti.utils import IterDict, read_config

CONFIG_FILE = "config/test/default_glylec.yaml"


default_config = read_config(CONFIG_FILE)
all_configs = IterDict()(default_config)


class BaseGlylecTestModel:
    @pytest.mark.parametrize("config", all_configs)
    @pytest.mark.slow
    def test_full(self, config, glylec_datamodule):
        glylec_datamodule.setup()
        glylec_datamodule.update_config(config)
        model = self.model_class(**config).cpu()
        trainer = Trainer(
            gpus=0,
            fast_dev_run=True,
            enable_checkpointing=False,
            logger=None,
            **config["trainer"],
        )
        trainer.fit(model, glylec_datamodule)

    @pytest.mark.parametrize("config", all_configs)
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_full_gpu(self, config, glylec_datamodule):
        glylec_datamodule.setup()
        glylec_datamodule.update_config(config)
        model = self.model_class(**config)
        trainer = Trainer(
            gpus=1,
            fast_dev_run=True,
            enable_checkpointing=False,
            logger=None,
            **config["trainer"],
        )
        trainer.fit(model, glylec_datamodule)

    @pytest.mark.parametrize("config", all_configs)
    def test_shared(self, config, glylec_datamodule):
        glylec_datamodule.setup()
        glylec_datamodule.update_config(config)
        model = self.model_class(**config)
        batch = next(iter(glylec_datamodule.train_dataloader()))
        output = model.shared_step(batch)
        assert "loss" in output.keys()
        assert "preds" in output.keys()
        assert "labels" in output.keys()


class TestGlylecClassificationModel(BaseGlylecTestModel):
    model_class = ClassificationModel
