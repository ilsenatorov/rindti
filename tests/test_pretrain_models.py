import pytest
from pytorch_lightning import Trainer

from rindti.models import ProtClassModel
from rindti.utils import IterDict, read_config

CONFIG_FILE = "tests/configs/default_pfam.yaml"


default_config = read_config(CONFIG_FILE)
all_configs = IterDict()(default_config)


class BaseTestModel:
    @pytest.mark.parametrize("config", all_configs)
    @pytest.mark.slow
    def test_full(self, config, pretrain_datamodule):
        pretrain_datamodule.setup()
        pretrain_datamodule.update_config(config)
        config["model"]["label_list"] = [0, 1, 2]
        model = self.model_class(**config)
        trainer = Trainer(
            gpus=0,
            fast_dev_run=True,
            enable_checkpointing=False,
            logger=None,
        )
        trainer.fit(model, pretrain_datamodule)

    @pytest.mark.parametrize("config", all_configs)
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_full_gpu(self, config, pretrain_datamodule):
        pretrain_datamodule.setup()
        pretrain_datamodule.update_config(config)
        config["model"]["label_list"] = [0, 1, 2]
        model = self.model_class(**config)
        trainer = Trainer(
            gpus=1,
            fast_dev_run=True,
            enable_checkpointing=False,
            logger=None,
        )
        trainer.fit(model, pretrain_datamodule)

    @pytest.mark.parametrize("config", all_configs)
    def test_shared(self, config, pretrain_datamodule):
        pretrain_datamodule.setup()
        pretrain_datamodule.update_config(config)
        config["model"]["label_list"] = [0, 1, 2]
        model = self.model_class(**config)
        batch = next(iter(pretrain_datamodule.train_dataloader()))
        output = model.shared_step(batch)
        assert "loss" in output.keys()
        assert "preds" in output.keys()
        assert "labels" in output.keys()


class TestPfamClassModel(BaseTestModel):
    model_class = ProtClassModel
