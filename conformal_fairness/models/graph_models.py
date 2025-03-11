import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv, GraphConv, SAGEConv

from ..config import BaseGNNConfig
from ..constants import (
    FEATURE_FIELD,
    LABEL_FIELD,
    LABELS_KEY,
    NODE_IDS_KEY,
    PROBS_KEY,
    LayerType,
)
from .base_models import BaseModel


class SimpleGATConv(torch.nn.Module):
    def __init__(self, *args, last_layer: bool = False, **kwargs):
        super(SimpleGATConv, self).__init__()
        self.layer = GATConv(*args, **kwargs)
        self.is_last_layer = last_layer

    def forward(self, mfg, x):
        x = self.layer(mfg, x)
        if self.is_last_layer:
            return x.mean(dim=1)
        return x.flatten(start_dim=1)


class BackboneGNN(torch.nn.Module):
    def __init__(
        self,
        config: BaseGNNConfig,
        num_features,
        num_classes,
    ):
        super(BackboneGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(config.dropout)
        base_layer = {
            LayerType.GCN.value: GraphConv,
            LayerType.GAT.value: SimpleGATConv,
            LayerType.GRAPHSAGE.value: SAGEConv,
        }[config.model]

        if config.layers == 1:
            l_args = (num_features, num_classes)
            l_kwargs = dict()
            if config.model == LayerType.GAT.value:
                l_args = (num_features, num_classes, config.heads)
                l_kwargs = dict(last_layer=True)
            elif config.model == LayerType.GRAPHSAGE.value:
                l_args = (num_features, num_classes, config.aggr)

            self.layers.append(base_layer(*l_args, **l_kwargs))
        else:
            in_l_args = (num_features, config.hidden_layer_size)
            in_l_kwargs = dict()

            mid_l_args = (config.hidden_layer_size, config.hidden_layer_size)
            mid_l_kwargs = dict()

            out_l_args = (config.hidden_layer_size, num_classes)
            out_l_kwargs = dict()
            if config.model == LayerType.GAT.value:
                in_l_args = (num_features, config.hidden_layer_size, config.heads)
                in_l_kwargs = dict(last_layer=False)
                mid_l_args = (
                    config.heads * config.hidden_layer_size,
                    config.hidden_layer_size,
                    config.heads,
                )
                mid_l_kwargs = dict(last_layer=False)

                out_l_args = (
                    config.heads * config.hidden_layer_size,
                    num_classes,
                    config.heads,
                )
                out_l_kwargs = dict(last_layer=True)
            elif config.model == LayerType.GRAPHSAGE.value:
                in_l_args = (num_features, config.hidden_layer_size, config.aggr)
                mid_l_args = (
                    config.hidden_layer_size,
                    config.hidden_layer_size,
                    config.aggr,
                )
                out_l_args = (config.hidden_layer_size, num_classes, config.aggr)

            self.layers.append(base_layer(*in_l_args, **in_l_kwargs))
            for _ in range(config.layers - 2):
                self.layers.append(base_layer(*mid_l_args, **mid_l_kwargs))
            self.layers.append(base_layer(*out_l_args, **out_l_kwargs))

        self.num_layers = config.layers

    def forward(self, mfgs, x):
        for idx, layer in enumerate(self.layers):
            x = layer(mfgs[idx], self.dropout(x))
            if idx != len(self.layers) - 1:
                x = F.relu(x)

        return x


class GNN(BaseModel):
    def __init__(self, config: BaseGNNConfig, num_features, num_classes):
        super(GNN, self).__init__(config, num_features, num_classes)
        self.base_model = BackboneGNN(
            config,
            num_features,
            num_classes,
        )

    def forward(self, mfgs, x):
        return self.base_model(mfgs, x)

    def training_step(self, batch, batch_idx):
        _, _, mfgs = batch
        batch_inputs = mfgs[0].srcdata[FEATURE_FIELD]
        batch_labels = mfgs[-1].dstdata[LABEL_FIELD]
        batch_pred = self.forward(mfgs, batch_inputs)

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
        batch_pred = self.forward(mfgs, batch_inputs)

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
        batch_pred = self.forward(mfgs, batch_inputs)

        self.test_acc(batch_pred.softmax(-1), batch_labels)
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
