import torch
import torch.nn.functional as F
import numpy as np

from ...data import TwoGraphData
from ...layers.encoder import GraphEncoder, PretrainedEncoder, SweetNetEncoder
from ...layers.other import MLP
from ...utils import remove_arg_prefix
from ..base_model import BaseModel

encoders = {"graph": GraphEncoder, "sweetnet": SweetNetEncoder, "pretrained": PretrainedEncoder}


class ClassificationModel(BaseModel):
    """Model for DTI prediction as a classification problem."""

    def __init__(self, pos_weight=1, neg_weight=1, **kwargs):
        super().__init__(**kwargs)
        self._determine_feat_method(
            kwargs["model"]["feat_method"],
            kwargs["model"]["prot"]["hidden_dim"],
            kwargs["model"]["drug"]["hidden_dim"],
        )
        self.prot_encoder = encoders[kwargs["model"]["prot"]["method"]](**kwargs["model"]["prot"])
        self.drug_encoder = encoders[kwargs["model"]["drug"]["method"]](**kwargs["model"]["drug"])
        self.mlp = MLP(input_dim=self.embed_dim, out_dim=1, **kwargs["model"]["mlp"])
        
        self.pos_weight = torch.tensor(pos_weight)
        self.neg_weight = torch.tensor(neg_weight)
        
        self.train_metrics, self.val_metrics, self.test_metrics = self._set_class_metrics()

    def forward(self, prot: dict, drug: dict) -> dict:
        """Forward the data though the classification model"""
        prot_embed, _ = self.prot_encoder(prot)
        drug_embed, _ = self.drug_encoder(drug)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        return dict(
            pred=self.mlp(joint_embedding),
            prot_embed=prot_embed,
            drug_embed=drug_embed,
            joint_embed=joint_embedding,
        )

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test.

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        fwd_dict = self.forward(prot, drug)
        labels = data["label"].unsqueeze(1)
        
        # weights = torch.where(labels == 0, self.pos_weight.to(labels.device),
        #                       self.neg_weight.to(labels.device)).float()
        bce_loss = F.binary_cross_entropy_with_logits(fwd_dict["pred"], labels.float())
        
        return dict(loss=bce_loss, preds=torch.sigmoid(fwd_dict["pred"].detach()), labels=labels.detach())

    def validation_epoch_end(self, outputs: dict):
        """WHat to do at the end of a validation epoch. Logs everthing."""
        metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        self.log_all(metrics, hparams=True)
        self.log("val_acc", metrics["val_Accuracy"])


class MultitaskClassification(ClassificationModel):
    def __init__(self, **kwargs):
        kwargs["model"]["mlp"]["hidden_dims"] = [512, 128, 64]
        super(MultitaskClassification, self).__init__(**kwargs)

        self.prot_node_classifier = MLP(input_dim=kwargs["model"]["drug"]["hidden_dim"], hidden_dims=[64, 32], out_dim=21)
        self.drug_node_classifier = MLP(input_dim=kwargs["model"]["prot"]["hidden_dim"], hidden_dims=[64, 16], out_dim=3)

        self.prot_class = "prot" in kwargs["transform"]["graphs"]
        self.drug_class = "drug" in kwargs["transform"]["graphs"]

        self.main_weight = kwargs["transform"]["graphs"]["main"]["weight"]

        if self.prot_class:
            self.pp_train_metrics, self.pp_val_metrics, self.pp_test_metrics = self._set_class_metrics(num_classes=21,
                                                                                                       prefix="pp_")
            self.prot_weight = kwargs["transform"]["graphs"]["prot"]["weight"]
        if self.drug_class:
            self.dp_train_metrics, self.dp_val_metrics, self.dp_test_metrics = self._set_class_metrics(num_classes=3,
                                                                                                       prefix="dp_")
            self.drug_weight = kwargs["transform"]["graphs"]["drug"]["weight"]

    def forward(self, prot: dict, drug: dict) -> dict:
        """Forward teh data through the model and also compute the predictions for side tasks"""
        prot_graph_embed, prot_node_embed = self.prot_encoder(prot)
        drug_graph_embed, drug_node_embed = self.drug_encoder(drug)
        joint_embedding = self.merge_features(drug_graph_embed, prot_graph_embed)

        prot_node_class = self.prot_node_classifier(prot_node_embed) if self.prot_class else None
        drug_node_class = self.drug_node_classifier(drug_node_embed) if self.drug_class else None

        return dict(
            pred=self.mlp(joint_embedding),
            prot_embed=prot_graph_embed,
            drug_embed=drug_graph_embed,
            joint_embed=joint_embedding,
            prot_node_class=prot_node_class,
            drug_node_class=drug_node_class,
        )

    def shared_step(self, data: TwoGraphData) -> dict:
        """"Compute the shared step of the model"""
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        fwd_dict = self.forward(prot, drug)
        labels = data["label"].unsqueeze(1)
        bce_loss = F.binary_cross_entropy_with_logits(fwd_dict["pred"], labels.float())
        loss = self.main_weight * bce_loss

        # if protein multitasking is enabled, compute it's loss
        if self.prot_class:
            idx = torch.tensor(np.argwhere(data["prot_x_orig"].detach().cpu() - data["prot_x"].detach().cpu())).squeeze()
            prot_loss = F.cross_entropy(torch.softmax(fwd_dict["prot_node_class"], dim=1)[idx, :], data["prot_x_orig"][idx])
            loss += self.prot_weight * prot_loss
        else:
            prot_loss = None

        # if drug multitasking is enabled, compute it's loss
        if self.drug_class:
            drug_loss = F.cross_entropy(torch.softmax(fwd_dict["drug_node_class"], dim=1), data["drug_x_orig"])
            loss += self.drug_weight * drug_loss
        else:
            drug_loss = None

        # fill the output directory accordingly
        output = dict(
            loss=loss,
            pred_loss=bce_loss.detach(),
            preds=torch.sigmoid(fwd_dict["pred"].detach()),
            labels=labels.detach(),
            prot_embed=fwd_dict["prot_embed"],
            drug_embed=fwd_dict["drug_embed"],
        )
        if self.prot_class:
            output.update(
                dict(
                    prot_loss=prot_loss.detach(),
                    prot_preds=torch.softmax(fwd_dict["prot_node_class"], dim=1),
                    prot_labels=data["prot_x_orig"],
                )
            )
        if self.drug_class:
            output.update(
                dict(
                    drug_loss=drug_loss.detach(),
                    drug_preds=torch.softmax(fwd_dict["drug_node_class"], dim=1),
                    drug_labels=data["drug_x_orig"],
                )
            )
        return output

    def training_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """Perform training and keep track of all its metrics"""
        ss = super(MultitaskClassification, self).training_step(data, data_idx)
        self.log("train_dti_loss", ss["pred_loss"], batch_size=self.batch_size)

        if self.prot_class:
            self.pp_train_metrics.update(ss["prot_preds"], ss["prot_labels"])
            self.log("pp_train_loss", ss["prot_loss"], batch_size=self.batch_size)
        if self.drug_class:
            self.dp_train_metrics.update(ss["drug_preds"], ss["drug_labels"])
            self.log("dp_train_loss", ss["drug_loss"], batch_size=self.batch_size)
        return ss

    def validation_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """Perform validation and keep track of all its metrics"""
        ss = super(MultitaskClassification, self).validation_step(data, data_idx)
        self.log("val_dti_loss", ss["pred_loss"], batch_size=self.batch_size)

        if self.prot_class:
            self.pp_val_metrics.update(ss["prot_preds"], ss["prot_labels"])
            self.log("pp_val_loss", ss["prot_loss"], batch_size=self.batch_size)
        if self.drug_class:
            self.dp_val_metrics.update(ss["drug_preds"], ss["drug_labels"])
            self.log("dp_val_loss", ss["drug_loss"], batch_size=self.batch_size)
        return ss

    def test_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """Perform testing and keep track of all its metrics"""
        ss = super(MultitaskClassification, self).test_step(data, data_idx)
        self.log("test_dti_loss", ss["pred_loss"], batch_size=self.batch_size)

        if self.prot_class:
            self.pp_test_metrics.update(ss["prot_preds"], ss["prot_labels"])
            self.log("pp_test_loss", ss["prot_loss"], batch_size=self.batch_size)
        if self.drug_class:
            self.dp_test_metrics.update(ss["drug_preds"], ss["drug_labels"])
            self.log("dp_test_loss", ss["drug_loss"], batch_size=self.batch_size)
        return ss

    def training_epoch_end(self, outputs: dict):
        """Calculate training metrics"""
        super(MultitaskClassification, self).training_epoch_end(outputs)
        if self.prot_class:
            metrics = self.pp_train_metrics.compute()
            self.pp_train_metrics.reset()
            self.log_all(metrics)
        if self.drug_class:
            metrics = self.dp_train_metrics.compute()
            self.dp_train_metrics.reset()
            self.log_all(metrics)

    def validation_epoch_end(self, outputs: dict):
        """Calculate validation metrics"""
        metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        self.log_all(metrics, hparams=True)
        self.log("val_acc", metrics["val_Accuracy"])

        if self.prot_class:
            metrics = self.pp_val_metrics.compute()
            self.pp_val_metrics.reset()
            self.log_all(metrics, hparams=True)
        if self.drug_class:
            metrics = self.dp_val_metrics.compute()
            self.dp_val_metrics.reset()
            self.log_all(metrics, hparams=True)

    def test_epoch_end(self, outputs: dict):
        """Calculate test metrics"""
        super(MultitaskClassification, self).test_epoch_end(outputs)
        if self.prot_class:
            metrics = self.pp_test_metrics.compute()
            self.pp_test_metrics.reset()
            self.log_all(metrics)
        if self.drug_class:
            metrics = self.dp_test_metrics.compute()
            self.dp_test_metrics.reset()
            self.log_all(metrics)
