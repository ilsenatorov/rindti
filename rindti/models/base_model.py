import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    ExplainedVariance,
    MatthewsCorrCoef,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)

from ..data import TwoGraphData
from .metrics import RM2, ConcordanceIndex, monitor_mode


class BaseModel(LightningModule):
    """Base model, defines a lot of helper functions."""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = kwargs["datamodule"]["batch_size"]

    def _set_class_metrics(self, num_classes: int = 2):
        task = "binary" if num_classes == 2 else "multiclass"
        kwargs = {} if num_classes == 2 else {"num_classes": num_classes}
        metrics = MetricCollection(
            [
                Accuracy(task=task, **kwargs),
                AUROC(task=task, **kwargs),
                # DTI datasets are heavily imbalanced (Davis is ~7% positive), where
                # AUROC flatters and average precision is the informative metric.
                AveragePrecision(task=task, **kwargs),
                MatthewsCorrCoef(task=task, **kwargs),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _set_reg_metrics(self):
        # MSE, CI and rm2 are the three numbers the DeepDTA -> GraphDTA -> DGraphDTA ->
        # GEFA lineage reports, so they are what any comparison table needs. Pearson
        # and Spearman come along cheaply; Spearman is the rank-based analogue of CI.
        metrics = MetricCollection(
            [
                MeanAbsoluteError(),
                MeanSquaredError(),
                ExplainedVariance(),
                PearsonCorrCoef(),
                SpearmanCorrCoef(),
                ConcordanceIndex(),
                RM2(),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _determine_feat_method(
        self,
        feat_method: str,
        drug_hidden_dim: int = None,
        prot_hidden_dim: int = None,
        **kwargs,
    ):
        """Which method to use for concatenating drug and protein representations."""
        if feat_method == "concat":
            self.merge_features = self._concat
            self.embed_dim = drug_hidden_dim + prot_hidden_dim
        elif feat_method == "element_l2":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._element_l2
            self.embed_dim = drug_hidden_dim
        elif feat_method == "element_l1":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._element_l1
            self.embed_dim = drug_hidden_dim
        elif feat_method == "mult":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._mult
            self.embed_dim = drug_hidden_dim
        else:
            raise ValueError("unsupported feature method")

    def _concat(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Concatenation."""
        return torch.cat((drug_embed, prot_embed), dim=1)

    def _element_l2(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L2 distance."""
        return torch.sqrt(((drug_embed - prot_embed) ** 2) + 1e-6).float()

    def _element_l1(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L1 distance."""
        return (drug_embed - prot_embed).abs()

    def _mult(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Multiplication."""
        return drug_embed * prot_embed

    def collect_aux_loss(self) -> Tensor:
        """Sum the auxiliary losses stashed by submodules during the forward pass.

        DiffPool's link-prediction and entropy regularizers are produced inside the
        pooling layer, which has no way to return them through the encoder, so it
        stores them on ``aux_loss`` and they are gathered here.
        """
        total = None
        for module in self.modules():
            aux = getattr(module, "aux_loss", None)
            if aux is not None:
                total = aux if total is None else total + aux
        return total

    def training_step(self, data: TwoGraphData, data_idx: int) -> Tensor:
        """What to do during training step."""
        ss = self.shared_step(data)
        self.train_metrics.update(ss["preds"], ss["labels"])
        self.log("train_loss", ss["loss"], batch_size=self.batch_size, on_epoch=True)
        loss = ss["loss"]
        aux = self.collect_aux_loss()
        if aux is not None:
            self.log("train_aux_loss", aux, batch_size=self.batch_size, on_epoch=True)
            loss = loss + aux
        return loss

    def validation_step(self, data: TwoGraphData, data_idx: int) -> Tensor:
        """What to do during validation step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        self.val_metrics.update(ss["preds"], ss["labels"])
        self.log("val_loss", ss["loss"], batch_size=self.batch_size, on_epoch=True)
        return ss["loss"]

    def test_step(self, data: TwoGraphData, data_idx: int) -> Tensor:
        """What to do during test step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        self.test_metrics.update(ss["preds"], ss["labels"])
        self.log("test_loss", ss["loss"], batch_size=self.batch_size, on_epoch=True)
        return ss["loss"]

    def _log_metrics(self, metrics: MetricCollection) -> None:
        """Compute, log and reset an accumulated metric collection.

        Values are cast to float: MatthewsCorrCoef can come back as an integer
        tensor, which Lightning cannot reduce.
        """
        self.log_dict({k: v.float() for k, v in metrics.compute().items()})
        metrics.reset()

    def on_train_epoch_end(self):
        """Compute, log and reset the accumulated training metrics."""
        self._log_metrics(self.train_metrics)

    def on_validation_epoch_end(self):
        """Compute, log and reset the accumulated validation metrics."""
        self._log_metrics(self.val_metrics)

    def on_test_epoch_end(self):
        """Compute, log and reset the accumulated test metrics."""
        self._log_metrics(self.test_metrics)

    def configure_optimizers(self) -> dict:
        """Configure the optimizer and the lr scheduler."""
        opt_params = self.hparams.model["optimizer"]
        optimizer_class = {"adamw": AdamW, "adam": Adam, "sgd": SGD, "rmsprop": RMSprop}[opt_params["module"]]
        kwargs = {"lr": opt_params["lr"]}
        if "weight_decay" in opt_params:
            kwargs["weight_decay"] = opt_params["weight_decay"]
        if optimizer_class in (SGD, RMSprop) and "momentum" in opt_params:
            kwargs["momentum"] = opt_params["momentum"]

        # Per-encoder learning rates: each encoder forms its own param group, and the
        # default group holds everything that is not already covered by one of them.
        groups, claimed = [], set()
        for name in ("prot", "drug"):
            encoder = getattr(self, f"{name}_encoder", None)
            lr = opt_params.get(f"{name}_lr")
            if encoder is None or lr is None:
                continue
            params = list(encoder.parameters())
            groups.append({"params": params, "lr": lr})
            claimed.update(id(p) for p in params)
        rest = [p for p in self.parameters() if id(p) not in claimed]
        groups.insert(0, {"params": rest})

        optimizer = optimizer_class(groups, **kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "monitor": self.hparams["model"]["monitor"],
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    # ReduceLROnPlateau defaults to "min" too, so a higher-is-better
                    # monitor would have had its LR cut whenever the model improved.
                    mode=monitor_mode(self.hparams["model"]["monitor"]),
                    factor=opt_params["reduce_lr"]["factor"],
                    patience=opt_params["reduce_lr"]["patience"],
                ),
            },
        }
