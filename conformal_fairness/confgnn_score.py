import glob
import os
from typing import List, Optional

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar

from . import utils
from .config import ConfExptConfig, ConfGNNConfig
from .constants import LABELS_KEY, SCORES_KEY
from .custom_logger import CustomLogger
from .data_module import DataModule
from .models import CFGNN
from .scores import CPScore


class CFGNNScore(CPScore):
    def __init__(
        self,
        conf_config: ConfExptConfig,
        datamodule: DataModule,
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
