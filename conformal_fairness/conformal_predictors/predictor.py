from typing import Dict, Optional

import torch
from dgl.dataloading import MultiLayerFullNeighborSampler

from ..config import (
    ConfExptConfig,
    ConfGNNConfig,
    DiffusionConfig,
    NeighborhoodConfig,
    PrimitiveScoreConfig,
    RegularizedConfig,
    SplitConfInput,
)
from ..constants import ConformalMethod, Stage
from ..cp_methods.confgnn_score import CFGNNScore
from ..cp_methods.scores import APSScore, NAPSScore, TPSScore
from ..cp_methods.transformations import (
    DiffusionTransformation,
    PredSetTransformation,
    RegularizationTransformation,
)
from ..custom_logger import CustomLogger
from ..data import BaseDataModule
from ..utils.data_utils import get_label_scores


class ConformalPredictor:
    def __init__(self, config: ConfExptConfig, **kwargs):
        self.config = config  # coverage req

    def C(self, x):
        """Generate a set/interval of values such that $P(y \in C(x)) \geq 1 - \alpha$"""
        raise NotImplementedError


class ConformalClassifier(ConformalPredictor):
    def __init__(self, config: ConfExptConfig, datamodule: BaseDataModule, **kwargs):
        super().__init__(config, **kwargs)
        self.datamodule = datamodule


class SplitConformalClassifier(ConformalClassifier):
    def __init__(self, config: ConfExptConfig, datamodule: BaseDataModule, **kwargs):
        super().__init__(config, datamodule, **kwargs)

    def calibrate(self, **calib_data):
        """Calibrate the conformal Predictor"""
        raise NotImplementedError


class ScoreSplitConformalClassifer(SplitConformalClassifier):
    """A score based split conformal classifier"""

    def __init__(self, config: ConfExptConfig, datamodule: BaseDataModule, **kwargs):
        super().__init__(config, datamodule, **kwargs)
        self.conformal_method = ConformalMethod(config.conformal_method)
        self.split_dict: Dict[Stage, torch.LongTensor] = datamodule.split_dict
        self._qhat = None
        self._score_module = None
        self._transform_module = None
        self._cached_scores = None

    def _get_scores(
        self,
        probs: torch.Tensor,
        split_conf_input: SplitConfInput,
    ):
        # calibration using score quantile
        # assuming that score is exchangeable, this should work
        if self.conformal_method in [ConformalMethod.TPS, ConformalMethod.APS]:
            assert isinstance(split_conf_input, PrimitiveScoreConfig)
        elif self.conformal_method == ConformalMethod.NAPS:
            assert isinstance(split_conf_input, NeighborhoodConfig)
        else:
            raise NotImplementedError

        if self.conformal_method == ConformalMethod.TPS:
            self._score_module = TPSScore(split_conf_input, alpha=self.config.alpha)
        elif self.conformal_method == ConformalMethod.APS:
            self._score_module = APSScore(split_conf_input, alpha=self.config.alpha)
        elif self.conformal_method == ConformalMethod.NAPS:
            self._score_module = NAPSScore(
                split_conf_input, datamodule=self.datamodule, alpha=self.config.alpha
            )  # Implementation in Paper uses APS Score
        else:
            raise NotImplementedError

        scores = self._score_module.pipe_compute(probs)

        if self._transform_module is not None:
            scores = self._transform_module.pipe_transform(scores)
        return scores

    def calibrate(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        split_conf_input: SplitConfInput,
    ):
        scores = self._cached_scores
        if scores is None:
            scores = self._get_scores(probs, split_conf_input)
            self._cached_scores = scores
        assert self._score_module is not None

        label_scores = get_label_scores(
            labels, scores, self.split_dict[Stage.CALIBRATION], self.config.dataset.name
        )

        # additional kwargs for tps
        if isinstance(self._score_module, TPSScore):
            kwargs = {
                "labels": labels[self.split_dict[Stage.CALIBRATION]],
                "num_classes": self.datamodule.num_classes,
            }
        else:
            kwargs = {}

        self._qhat = self._score_module.compute_quantile(label_scores, **kwargs)
        return self._qhat

    def run(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        split_conf_input: SplitConfInput,
    ):
        qhat = self.calibrate(probs, labels, split_conf_input)
        assert self._cached_scores is not None

        test_labels = labels[self.split_dict[Stage.TEST]]
        test_scores = self._cached_scores[self.split_dict[Stage.TEST]]

        # Scores could have been implemented as a pipeline of transformations
        prediction_sets = PredSetTransformation(threshold=qhat).pipe_transform(
            test_scores
        )

        return prediction_sets, test_labels


class ScoreMultiSplitConformalClassifier(ScoreSplitConformalClassifer):
    def __init__(self, config: ConfExptConfig, datamodule: BaseDataModule, **kwargs):
        super().__init__(config, datamodule, **kwargs)
        self.best_params = None

    def get_dataloader(self, points, batch_size=-1):
        if self.conformal_method == ConformalMethod.CFGNN:
            assert isinstance(self._score_module, CFGNNScore)
            total_num_layers = self._score_module.total_layers
            sampler = MultiLayerFullNeighborSampler(total_num_layers)
            # if batch_size < 0:
            #    raise ValueError("Unexpected batch size")
            if batch_size < 0:
                batch_size = len(points)
            return self.datamodule.custom_dataloader(points, batch_size, sampler)
        else:
            raise NotImplementedError

    def calibrate_with_model(
        self,
        labels: torch.Tensor,
        split_conf_input: SplitConfInput,
        logger: CustomLogger,
    ):
        if self.conformal_method == ConformalMethod.CFGNN:
            assert isinstance(split_conf_input, ConfGNNConfig)
            self._score_module = CFGNNScore(
                conf_config=self.config,
                datamodule=self.datamodule,
                confgnn_config=split_conf_input,
                logger=logger,
            )
            calib_tune_points = self.split_dict[Stage.CALIBRATION_TUNE]
            calib_qscore_points = self.split_dict[Stage.CALIBRATION_QSCORE]
            calib_tune_dl = self.get_dataloader(
                calib_tune_points, self.config.batch_size
            )
            calib_qscore_dl = self.get_dataloader(
                calib_qscore_points, self.config.batch_size
            )
            scores = self._score_module.learn_params(calib_tune_dl, calib_qscore_dl)

            calib_qscore_points = self.split_dict[Stage.CALIBRATION_QSCORE]
            label_scores = get_label_scores(
                labels, scores, calib_qscore_points, self.config.dataset.name
            )

            self._qhat = self._score_module.compute_quantile(
                label_scores, self.config.alpha
            )
        else:
            raise NotImplementedError

        return self._qhat

    def calibrate_with_probs(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        split_conf_input: SplitConfInput,
    ):
        match self.conformal_method:
            case ConformalMethod.DAPS | ConformalMethod.DTPS:
                primitive_config = PrimitiveScoreConfig(
                    use_aps_epsilon=split_conf_input.use_aps_epsilon,
                    use_tps_classwise=split_conf_input.use_tps_classwise,
                )
                if self.conformal_method == ConformalMethod.DAPS:
                    self._score_module = APSScore(
                        primitive_config, alpha=self.config.alpha
                    )
                elif self.conformal_method == ConformalMethod.DTPS:
                    self._score_module = TPSScore(
                        primitive_config, alpha=self.config.alpha
                    )

                kwargs = {"dataset": self.config.dataset.name}
                self._transform_module = DiffusionTransformation(
                    split_conf_input, **kwargs
                )

                self.best_params = self._transform_module.find_params(
                    probs, labels, self._score_module, self.datamodule
                )

                scores = self._cached_scores
                if scores is None:
                    scores = self._score_module.pipe_compute(probs)
                    scores = self._transform_module.transform(
                        scores,
                        datamodule=self.datamodule,
                        diffusion_param=self.best_params["diffusion_param"],
                    )
            case ConformalMethod.RAPS:
                primitive_config = PrimitiveScoreConfig(
                    use_aps_epsilon=False,
                    use_tps_classwise=split_conf_input.use_tps_classwise,
                )
                self._score_module = APSScore(primitive_config, alpha=self.config.alpha)

                self._transform_module = RegularizationTransformation(
                    config=split_conf_input
                )

                self.best_params = self._transform_module.find_params(
                    probs, labels, self._score_module, self.datamodule
                )

                scores = None
                if split_conf_input.raps_mod and self._cached_scores is not None:
                    # we can use cached score in this case
                    scores = self._cached_scores

                if scores is None:
                    # don't apply \lambda * \1[r_y > K] adjustment to scores if not raps_mod
                    # for computing threshold
                    scores = self._score_module.pipe_compute(probs)
                    scores = self._transform_module.transform(
                        scores,
                        probs,
                        raps_modified=split_conf_input.raps_mod,
                        **self.best_params,
                    )
            case _:
                raise NotImplementedError

        self._cached_scores = scores

        # compute quantile on separate subset
        calib_qscore_points = self.split_dict[Stage.CALIBRATION_QSCORE]
        label_scores = get_label_scores(
            labels, scores, calib_qscore_points, self.config.dataset.name
        )

        if isinstance(self._score_module, TPSScore):
            kwargs = {
                "labels": labels[calib_qscore_points],
                "num_classes": self.datamodule.num_classes,
            }
        else:
            kwargs = {}
        self._qhat = self._score_module.compute_quantile(
            label_scores, self.config.alpha, **kwargs
        )
        return self._qhat

    def run(
        self,
        *,
        split_conf_input: SplitConfInput,
        probs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        logger: Optional[CustomLogger] = None,
    ):
        # resplit calibration into tuning and test splits
        self.datamodule.split_calib_tune_qscore(
            tune_frac=split_conf_input.tuning_fraction
        )
        # update splits
        self.split_dict = self.datamodule.split_dict

        test_points = self.split_dict[Stage.TEST]

        if isinstance(split_conf_input, ConfGNNConfig):
            if split_conf_input.load_probs:
                # use probabilities directly as features
                self.datamodule.update_features(probs)
            qhat = self.calibrate_with_model(
                split_conf_input=split_conf_input,
                logger=logger,
            )
            test_dl = self.get_dataloader(test_points, self.config.batch_size)
            test_scores, test_labels = self._score_module.pipe_compute(test_dl)
        elif isinstance(split_conf_input, DiffusionConfig):
            qhat = self.calibrate_with_probs(
                probs=probs,
                labels=labels,
                split_conf_input=split_conf_input,
            )
            test_labels = labels[test_points]
            test_scores = self._cached_scores[test_points]

        elif isinstance(split_conf_input, RegularizedConfig):
            qhat = self.calibrate_with_probs(
                probs=probs,
                labels=labels,
                split_conf_input=split_conf_input,
            )
            test_labels = labels[self.split_dict[Stage.TEST]]
            if split_conf_input.raps_mod:
                test_scores = self._cached_scores[self.split_dict[Stage.TEST]]
            else:
                # need to recompute scores for sets in original RAPS to adjust random sets
                test_probs = probs[test_points]
                test_scores = self._score_module.pipe_compute(test_probs)
                test_scores = self._transform_module.transform(
                    test_scores, test_probs, raps_modified=True, **self.best_params
                )

        else:
            raise NotImplementedError

        prediction_sets = PredSetTransformation(threshold=qhat).pipe_transform(
            test_scores
        )

        return prediction_sets, test_labels
