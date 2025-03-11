import torch.nn as nn
import torch.nn.functional as F

from ..config import BaseMLPConfig
from ..constants import FEATURE_FIELD, LABEL_FIELD, LABELS_KEY, NODE_IDS_KEY, PROBS_KEY
from .base_models import BaseModel


class MLPLayer(nn.Module):
    def __init__(self, num_in, num_out, dropout):
        super().__init__()
        self.linear = nn.Linear(num_in, num_out)
        self.batchnorm = nn.BatchNorm1d(num_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Linear -> Batchnorm -> Activation -> Dropout
        x = self.linear(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        return self.dropout(x)


class MLP(BaseModel):
    def __init__(self, config: BaseMLPConfig, num_features, num_classes):
        super(MLP, self).__init__(config, num_features, num_classes)
        self.layers = nn.ModuleList()

        nodes = [num_features] + ([self.hidden_layer_size] * self.num_layers)
        for i in range(len(nodes) - 1):
            self.layers.append(MLPLayer(nodes[i], nodes[i + 1], config.dropout))
        self.layers.append(nn.Linear(nodes[-1], num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x)

    def training_step(self, batch, batch_idx):
        _, _, mfgs = batch
        batch_inputs = mfgs[0].srcdata[FEATURE_FIELD]
        batch_labels = mfgs[-1].dstdata[LABEL_FIELD]
        batch_pred = self.forward(batch_inputs)

        loss = F.cross_entropy(batch_pred, batch_labels)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_inputs.shape[0],
            sync_dist=True,
        )

        self.training_step_outputs.append(
            {PROBS_KEY: batch_pred.softmax(-1), LABELS_KEY: batch_labels}
        )

        return loss

    def validation_step(self, batch, batch_idx):
        _, _, mfgs = batch
        batch_inputs = mfgs[0].srcdata[FEATURE_FIELD]
        batch_labels = mfgs[-1].dstdata[LABEL_FIELD]
        batch_pred = self.forward(batch_inputs)

        val_loss = F.cross_entropy(batch_pred, batch_labels)
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_inputs.shape[0],
            sync_dist=True,
        )

        self.validation_step_outputs.append(
            {PROBS_KEY: batch_pred.softmax(-1), LABELS_KEY: batch_labels}
        )

        return val_loss

    def test_step(self, batch, batch_idx):
        _, node_ids, mfgs = batch
        batch_inputs = mfgs[0].srcdata[FEATURE_FIELD]
        batch_labels = mfgs[-1].dstdata[LABEL_FIELD]
        batch_pred = self.forward(batch_inputs)

        self.test_acc(batch_pred.softmax(-1), batch_labels)
        # TODO: Do we need to output test_acc?
        self.log(
            "test_acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_inputs.shape[0],
        )

        self.stepwise_test_results.append(
            {
                NODE_IDS_KEY: node_ids,
                PROBS_KEY: batch_pred.softmax(-1),
                LABELS_KEY: batch_labels,
            }
        )
