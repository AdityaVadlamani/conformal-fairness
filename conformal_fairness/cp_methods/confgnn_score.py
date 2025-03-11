import glob
import os
from typing import Dict, List, Optional

import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from torchmetrics.classification import F1Score

from ..config import ConfExptConfig, ConfGNNConfig, PrimitiveScoreConfig
from ..constants import (
    FEATURE_FIELD,
    LABEL_FIELD,
    LABELS_KEY,
    NODE_IDS_KEY,
    SCORES_KEY,
    ConformalMethod,
)
from ..custom_logger import CustomLogger
from ..data import BaseDataModule
from ..models.graph_models import GNN, BackboneGNN
from ..utils import data_utils as utils
from .scores import APSScore, CPScore, TPSScore

# elementary scores map for pointwise scores
ELEM_SCORE_MAP = {
    ConformalMethod.TPS: TPSScore,
    ConformalMethod.APS: APSScore,
}


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


class CFGNNScore(CPScore):
    def __init__(
        self,
        conf_config: ConfExptConfig,
        datamodule: BaseDataModule,
        confgnn_config: ConfGNNConfig,
        logger: Optional[CustomLogger] = None,
    ):
        super().__init__(confgnn_config=confgnn_config)
        self.alpha = conf_config.alpha

        self.confgnn_config = confgnn_config

        self.trainable_model = CFGNN(
            config=confgnn_config,
            alpha=conf_config.alpha,
            num_epochs=conf_config.epochs,
            num_classes=datamodule.num_classes,
        )

        if not confgnn_config.load_probs:
            self.total_layers = (
                self.trainable_model.base_model.num_layers
                + self.trainable_model.confgnn.num_layers
            )
        else:
            self.total_layers = self.trainable_model.confgnn.num_layers

        callbacks: List[Callback] = [TQDMProgressBar(refresh_rate=100)]

        if confgnn_config.ckpt_dir is not None:
            best_callback = ModelCheckpoint(
                monitor="confgnn_val_eff",
                dirpath=confgnn_config.ckpt_dir,
                filename=f"confgnn-{confgnn_config.model}-{{epoch:02d}}-{{confgnn_val_eff:.2f}}",
                save_top_k=1,
                mode="min",
            )
            callbacks.append(best_callback)

        self.pt = utils.setup_trainer(conf_config, logger, callbacks=callbacks)

    def compute(self, dl, **kwargs):
        with utils.dl_affinity_setup(dl)():
            with torch.no_grad():
                self.pt.test(self.trainable_model, dataloaders=dl)
                scores, labels = (
                    self.trainable_model.latest_test_results[SCORES_KEY],
                    self.trainable_model.latest_test_results[LABELS_KEY],
                )

        return scores, labels

    def compute_quantile(self, scores, alpha=None, **kwargs):
        return self.trainable_model.eval_score_fn.compute_quantile(scores, self.alpha)

    def learn_params(self, calib_tune_dl, eval_dl):
        if not self.confgnn_config.trained_model_dir or not os.path.exists(
            self.confgnn_config.trained_model_dir
        ):
            with utils.dl_affinity_setup(calib_tune_dl)():
                # first fit the model
                self.pt.fit(
                    self.trainable_model,
                    train_dataloaders=calib_tune_dl,
                    val_dataloaders=calib_tune_dl,
                    ckpt_path=None,
                )
        else:
            paths = glob.glob(
                os.path.join(self.confgnn_config.trained_model_dir, "*.ckpt")
            )
            assert (
                paths and len(paths) == 1
            ), f"Expected exactly 1 checkpoint in the {self.confgnn_config.trained_model_dir}"

            self.trainable_model = CFGNN.load_from_checkpoint(paths[0], strict=False)

        # determine qhat
        scores, _ = self.compute(eval_dl)

        return scores
