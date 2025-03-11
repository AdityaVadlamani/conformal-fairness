from abc import ABC
from typing import Dict, List

import lightning as L
import torch
import torch.optim as optim
from torchmetrics.classification import F1Score

from ..config import BaseModelConfig
from ..constants import LABELS_KEY, NODE_IDS_KEY, PROBS_KEY


class BaseModel(ABC, L.LightningModule):
    def __init__(self, config: BaseModelConfig, num_features, num_classes):
        super(BaseModel, self).__init__()
        self.save_hyperparameters()

        self.lr = config.lr
        self.num_layers = config.layers
        self.hidden_layer_size = config.hidden_layer_size
        self.num_classes = num_classes
        self.num_features = num_features

        # F1Score metrics
        self.train_acc = F1Score(
            num_classes=num_classes, task="multiclass", average="micro"
        )
        self.val_acc = F1Score(
            num_classes=num_classes, task="multiclass", average="micro"
        )
        self.test_acc = F1Score(
            num_classes=num_classes, task="multiclass", average="micro"
        )

        # Outputs
        self.training_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.validation_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.stepwise_test_results: List[Dict[str, torch.Tensor]] = []
        self.latest_test_results: Dict[str, torch.Tensor] = {}

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
        )

        return optimizer

    def on_train_epoch_end(self) -> None:
        smx_probs = torch.cat([x[PROBS_KEY] for x in self.training_step_outputs])
        labels = torch.cat([x[LABELS_KEY] for x in self.training_step_outputs])

        self.train_acc(smx_probs, labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        smx_probs = torch.cat([x[PROBS_KEY] for x in self.validation_step_outputs])
        labels = torch.cat([x[LABELS_KEY] for x in self.validation_step_outputs])

        self.val_acc(smx_probs, labels)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.validation_step_outputs.clear()

    def on_test_start(self):
        self.stepwise_test_results = []
        self.latest_test_results = {}

    def on_test_end(self):
        self.latest_test_results = {
            NODE_IDS_KEY: torch.concat(
                [x[NODE_IDS_KEY] for x in self.stepwise_test_results], dim=0
            ).cpu(),
            PROBS_KEY: torch.concat(
                [x[PROBS_KEY] for x in self.stepwise_test_results], dim=0
            ).cpu(),
            LABELS_KEY: torch.concat(
                [x[LABELS_KEY] for x in self.stepwise_test_results], dim=0
            ).cpu(),
        }

        sorted_idx = self.latest_test_results[NODE_IDS_KEY].argsort()

        self.latest_test_results = {
            k: v[sorted_idx] for k, v in self.latest_test_results.items()
        }
