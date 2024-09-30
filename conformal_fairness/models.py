from typing import Dict, List

import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn import GATConv, GraphConv, SAGEConv, SGConv
from torchmetrics.classification import F1Score

from .config import BaseGNNConfig, BaseMLPConfig, ConfGNNConfig, PrimitiveScoreConfig
from .constants import (
    FEATURE_FIELD,
    LABEL_FIELD,
    LABELS_KEY,
    NODE_IDS_KEY,
    PROBS_KEY,
    SCORES_KEY,
    layer_types,
)
from .scores import ELEM_SCORE_MAP


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
            layer_types.GCN.name: GraphConv,
            layer_types.GAT.name: SimpleGATConv,
            layer_types.GraphSAGE.name: SAGEConv,
            layer_types.SGC.name: SGConv,
        }[config.model]

        if config.layers == 1:
            l_args = (num_features, num_classes)
            l_kwargs = dict()
            if config.model == layer_types.GAT.name:
                l_args = (num_features, num_classes, config.heads)
                l_kwargs = dict(last_layer=True)
            elif config.model == layer_types.GraphSAGE.name:
                l_args = (num_features, num_classes, config.aggr)

            self.layers.append(base_layer(*l_args, **l_kwargs))
        else:
            in_l_args = (num_features, config.hidden_channels)
            in_l_kwargs = dict()

            mid_l_args = (config.hidden_channels, config.hidden_channels)
            mid_l_kwargs = dict()

            out_l_args = (config.hidden_channels, num_classes)
            out_l_kwargs = dict()
            if config.model == layer_types.GAT.name:
                in_l_args = (num_features, config.hidden_channels, config.heads)
                in_l_kwargs = dict(last_layer=False)
                mid_l_args = (
                    config.heads * config.hidden_channels,
                    config.hidden_channels,
                    config.heads,
                )
                mid_l_kwargs = dict(last_layer=False)

                out_l_args = (
                    config.heads * config.hidden_channels,
                    num_classes,
                    config.heads,
                )
                out_l_kwargs = dict(last_layer=True)
            elif config.model == layer_types.GraphSAGE.name:
                in_l_args = (num_features, config.hidden_channels, config.aggr)
                mid_l_args = (
                    config.hidden_channels,
                    config.hidden_channels,
                    config.aggr,
                )
                out_l_args = (config.hidden_channels, num_classes, config.aggr)

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


class GNN(L.LightningModule):
    def __init__(self, config: BaseGNNConfig, num_features, num_classes):
        super(GNN, self).__init__()
        self.save_hyperparameters()
        self.base_model = BackboneGNN(
            config,
            num_features,
            num_classes,
        )

        self.lr = config.lr
        self.num_layers = config.layers

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

    def forward(self, mfgs, x):
        return self.base_model(mfgs, x)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
        )

        return optimizer

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

    def on_train_epoch_end(self) -> None:
        smx_probs = torch.cat([x[PROBS_KEY] for x in self.training_step_outputs])
        labels = torch.cat([x[LABELS_KEY] for x in self.training_step_outputs])

        self.train_acc(smx_probs, labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )

        self.training_step_outputs.clear()

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

    def on_validation_epoch_end(self) -> None:
        smx_probs = torch.cat([x[PROBS_KEY] for x in self.validation_step_outputs])
        labels = torch.cat([x[LABELS_KEY] for x in self.validation_step_outputs])

        self.val_acc(smx_probs, labels)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.validation_step_outputs.clear()

    def on_test_start(self):
        self.stepwise_test_results = []
        self.latest_test_results = {}

    def test_step(self, batch, batch_idx):
        _, node_ids, mfgs = batch
        batch_inputs = mfgs[0].srcdata[FEATURE_FIELD]
        batch_labels = mfgs[-1].dstdata[LABEL_FIELD]
        batch_pred = self.forward(mfgs, batch_inputs)

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


# TODO for DDP:
# 1. Use `sync_dist = True` for logging losses
# 2. More efficient and more accurate to save outputs/preds to an array and then on_validation_epoch_end compute acc


class CFGNN(L.LightningModule):
    def __init__(
        self,
        config: ConfGNNConfig,
        alpha: float,
        num_epochs,
        num_classes,
    ):
        super(CFGNN, self).__init__()
        self.save_hyperparameters()
        self.config = config

        self.alpha = alpha
        train_score_fn = ELEM_SCORE_MAP.get(config.train_fn)
        eval_score_fn = ELEM_SCORE_MAP.get(config.eval_fn)

        # TODO: Support for args (beyond use_aps_epsilon) for trian/eval score fn within the config
        fn_args = dict(
            config=PrimitiveScoreConfig(use_aps_epsilon=config.use_aps_epsilon),
            alpha=alpha,
        )

        self.confgnn = BackboneGNN(
            config,
            num_classes,
            num_classes,
        )
        self.temperature = config.temperature
        self.lr = config.lr

        # TODO: ensure that this doesn't construct paths differently from utils.load_basegnn
        if not self.config.load_probs:
            self.base_model = GNN.load_from_checkpoint(config.base_model_path)

        self.train_score_fn = train_score_fn(**fn_args)
        self.eval_score_fn = eval_score_fn(**fn_args)

        # Used to determine fraction of training epochs for pure loss training vs loss+efficiency training
        self.confgnn_f_label_train = config.label_train_fraction

        # Weighting for cross_entropy in loss+efficiency training
        self.confgnn_wt_ce = config.ce_weight
        self.num_epochs = num_epochs

        if not self.config.load_probs:
            self.base_model_num_layers = self.base_model.num_layers

        # Outputs
        self.training_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.validation_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.stepwise_test_results: List[Dict[str, torch.Tensor]] = []
        self.latest_test_results: Dict[str, torch.Tensor] = {}

        # train loss
        self.label_loss = nn.CrossEntropyLoss()

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

    def forward(self, mfgs, in_feat):
        if not self.config.load_probs:
            n_base_layers = self.base_model_num_layers
            with torch.no_grad():
                base_probs = F.softmax(
                    self.base_model(mfgs[: self.base_model_num_layers], in_feat), dim=1
                )
        else:
            n_base_layers = 0
            base_probs = in_feat

        adjusted_probs = self.confgnn(mfgs[n_base_layers:], base_probs)
        return base_probs, adjusted_probs

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
        )

        return optimizer

    def training_step(self, batch, batch_idx):
        # assume we get the full calibration dataset
        _, _, mfgs = batch
        batch_inputs = mfgs[0].srcdata[FEATURE_FIELD]
        batch_labels = mfgs[-1].dstdata[LABEL_FIELD]
        _, batch_logits = self.forward(mfgs, batch_inputs)
        batch_probs = F.softmax(batch_logits, dim=1)
        batch_size = batch_labels.shape[0]

        if (
            self.num_epochs is not None
            and self.current_epoch <= self.confgnn_f_label_train * self.num_epochs
        ):
            loss = self.label_loss(batch_logits, batch_labels)
            # self.log("confgnn_train_phase", 0.0, prog_bar=True, on_step=False, on_epoch=True) # training with crossentropy
        else:
            batch_scores = self.train_score_fn.compute(batch_probs)
            label_scores = torch.gather(
                batch_scores, 1, batch_labels.unsqueeze(1)
            ).squeeze()
            n_calib = len(label_scores)
            # compute quantile for label scores from second half
            corr_calib_quantile = self.train_score_fn.compute_quantile(
                label_scores[n_calib // 2 :]
            )

            # use quantile to compute coformal scores for first half based on ConfTr
            # corr_test_scores = torch.sigmoid((-batch_scores[:n_calib//2] + corr_calib_quantile)/self.temperature)
            # TODO: Try ReLU/alternative losses
            corr_test_scores = torch.relu(
                (-batch_scores[: n_calib // 2] + corr_calib_quantile) / self.temperature
            )

            conformal_eff_loss = torch.mean(corr_test_scores)
            loss = (
                1 - self.confgnn_wt_ce
            ) * conformal_eff_loss + self.confgnn_wt_ce * self.label_loss(
                batch_logits, batch_labels
            )

            under_quantile = (
                batch_scores[: n_calib // 2] <= corr_calib_quantile
            ).float()
            eff = torch.mean(torch.sum(under_quantile, dim=1))
            self.log(
                "confgnn_train_eff",
                eff,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            # self.log("confgnn_train_phase", 1.0, prog_bar=True, on_step=False, on_epoch=True) # training with conformal loss

        # DEBUG
        # self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)

        # self.train_acc(batch_probs, batch_labels)
        # self.log("train_acc", self.train_acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        _, _, mfgs = batch
        batch_inputs = mfgs[0].srcdata[FEATURE_FIELD]
        batch_labels = mfgs[-1].dstdata[LABEL_FIELD]

        _, batch_logits = self.forward(mfgs, batch_inputs)
        batch_probs = F.softmax(batch_logits, dim=1)
        batch_size = batch_labels.shape[0]
        batch_scores = self.eval_score_fn.compute(batch_probs)

        label_scores = torch.gather(
            batch_scores, 1, batch_labels.unsqueeze(1)
        ).squeeze()
        n_calib = len(label_scores)
        corr_calib_quantile = self.eval_score_fn.compute_quantile(
            label_scores[n_calib // 2 :]
        )
        under_quantile = (batch_scores[: n_calib // 2] <= corr_calib_quantile).float()
        eff = torch.mean(torch.sum(under_quantile, dim=1))
        self.log(
            "confgnn_val_eff",
            eff,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.val_acc(batch_probs, batch_labels)
        self.log(
            "confgnn_val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def on_test_start(self):
        self.stepwise_test_results = []
        self.latest_test_results = {}

    def test_step(self, batch, batch_idx):
        _, node_ids, mfgs = batch
        batch_inputs = mfgs[0].srcdata[FEATURE_FIELD]
        batch_labels = mfgs[-1].dstdata[LABEL_FIELD]

        _, batch_logits = self.forward(mfgs, batch_inputs)
        batch_probs = F.softmax(batch_logits, dim=1)
        batch_scores = self.eval_score_fn.compute(batch_probs)

        self.stepwise_test_results.append(
            {NODE_IDS_KEY: node_ids, SCORES_KEY: batch_scores, LABELS_KEY: batch_labels}
        )
        return batch_scores, batch_labels

    def on_test_end(self):
        self.latest_test_results = {
            NODE_IDS_KEY: torch.concat(
                [x[NODE_IDS_KEY] for x in self.stepwise_test_results], dim=0
            ).cpu(),
            SCORES_KEY: torch.concat(
                [x[SCORES_KEY] for x in self.stepwise_test_results], dim=0
            ).cpu(),
            LABELS_KEY: torch.concat(
                [x[LABELS_KEY] for x in self.stepwise_test_results], dim=0
            ).cpu(),
        }

        sorted_idx = self.latest_test_results[NODE_IDS_KEY].argsort()

        self.latest_test_results = {
            k: v[sorted_idx] for k, v in self.latest_test_results.items()
        }


# Standard MLP Models


class MLP(L.LightningModule):
    def __init__(self, config: BaseMLPConfig, num_features, num_classes):
        super(MLP, self).__init__()
        self.save_hyperparameters()
        self.layers = nn.ModuleList()

        self.lr = config.lr
        self.num_layers = config.layers
        self.hidden_nodes = config.d_hidden
        self.num_classes = num_classes

        nodes = [num_features] + ([self.hidden_nodes] * self.num_layers)
        for i in range(len(nodes) - 1):
            self.layers.append(BaseLinearLayer(nodes[i], nodes[i + 1], config.dropout))
        self.layers.append(nn.Linear(nodes[-1], num_classes))

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

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
        )

        return optimizer

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

    def on_train_epoch_end(self) -> None:
        smx_probs = torch.cat([x[PROBS_KEY] for x in self.training_step_outputs])
        labels = torch.cat([x[LABELS_KEY] for x in self.training_step_outputs])

        self.train_acc(smx_probs, labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )

        self.training_step_outputs.clear()

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

    def on_validation_epoch_end(self) -> None:
        smx_probs = torch.cat([x[PROBS_KEY] for x in self.validation_step_outputs])
        labels = torch.cat([x[LABELS_KEY] for x in self.validation_step_outputs])

        self.val_acc(smx_probs, labels)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.validation_step_outputs.clear()

    def on_test_start(self):
        self.stepwise_test_results = []
        self.latest_test_results = {}

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


class BaseLinearLayer(nn.Module):
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
