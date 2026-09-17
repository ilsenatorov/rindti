import pytest
from lightning.pytorch import Trainer

from rindti.models import ClassificationModel, RegressionModel
from rindti.utils import IterDict, read_config

CONFIG_FILE = "config/test/default_dti.yaml"


default_config = read_config(CONFIG_FILE)
all_configs = IterDict()(default_config)


class BaseTestModel:
    # `dti_datamodule` is produced by the full snakemake pipeline.
    pytestmark = pytest.mark.snakemake

    @pytest.mark.parametrize("config", all_configs)
    @pytest.mark.slow
    def test_full(self, config, dti_datamodule):
        dti_datamodule.setup()
        dti_datamodule.update_config(config)
        model = self.model_class(**config)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            enable_checkpointing=False,
            logger=None,
            **config["trainer"],
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
            accelerator="gpu",
            devices=1,
            fast_dev_run=True,
            enable_checkpointing=False,
            logger=None,
            **config["trainer"],
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


class TestRegressionModel(BaseTestModel):
    model_class = RegressionModel


class TestMergeFeatures:
    """The three `feat_method` arms must be three different functions.

    There used to be a fourth, `element_l2`, computing sqrt((d - p)**2 + 1e-6) - which is
    `element_l1` to six decimal places, so the ablation ran one arm twice.
    """

    @staticmethod
    def _model():
        return ClassificationModel.__new__(ClassificationModel)

    def test_merges_are_distinct(self):
        import torch

        model = self._model()
        drug, prot = torch.randn(8, 16), torch.randn(8, 16)
        merges = {
            "element_l1": ClassificationModel._element_l1(model, drug, prot),
            "mult": ClassificationModel._mult(model, drug, prot),
        }
        names = list(merges)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                assert not torch.allclose(merges[left], merges[right], atol=1e-3)

    def test_element_l2_is_gone(self):
        assert not hasattr(ClassificationModel, "_element_l2")
