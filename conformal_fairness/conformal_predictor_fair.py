import math
from typing import Dict

import torch
from dgl.dataloading import MultiLayerFullNeighborSampler

from .confgnn_score import CFGNNScore
from .config import (
    ConfFairExptConfig,
    ConfGNNConfig,
    DiffusionConfig,
    PrimitiveScoreConfig,
    SplitConfInput,
)
from .constants import SENS_FIELD, ConformalMethod, Stage, fairness_metric
from .data_module import DataModule
from .data_utils import get_label_scores
from .fair_utils import inverse_quantile
from .scores import APSScore, TPSScore
from .transformations import DiffusionTransformation, PredSetTransformation


class FairConformalPredictor:
    def __init__(self, config: ConfFairExptConfig, **kwargs):
        self.config = config  # coverage req
        self.lambda_hat = None
        self.step_size: int = kwargs.get("step_size", 0.001)

    def C(self, x):
        """Generate a set/interval of values such that $P(y \in C(x)) \geq 1 - \alpha$"""
        raise NotImplementedError


class FairConformalClassifier(FairConformalPredictor):
    def __init__(self, config: ConfFairExptConfig, datamodule: DataModule, **kwargs):
        super().__init__(config, **kwargs)
        self.datamodule = datamodule


class SplitFairConformalClassifier(FairConformalClassifier):
    def __init__(self, config: ConfFairExptConfig, datamodule: DataModule, **kwargs):
        super().__init__(config, datamodule, **kwargs)

    def calibrate(self, **calib_data):
        """Calibrate the conformal Predictor"""
        raise NotImplementedError


class ScoreSplitFairConformalClassifer(SplitFairConformalClassifier):
    """A score based split conformal classifier"""

    def __init__(self, config: ConfFairExptConfig, datamodule: DataModule, **kwargs):
        super().__init__(config, datamodule, **kwargs)
        self.conformal_method = ConformalMethod(config.conformal_method)
        self.split_dict: Dict[Stage, torch.LongTensor] = datamodule.split_dict
        self._qhat = None
        self._score_module = None
        self._transform_module = None
        self._cached_scores = None

    def get_dataloader(self, nodes, batch_size=-1):
        if self.conformal_method == ConformalMethod.CFGNN:
            assert isinstance(self._score_module, CFGNNScore)
            total_num_layers = self._score_module.total_layers
            sampler = MultiLayerFullNeighborSampler(total_num_layers)
            # if batch_size < 0:
            #    raise ValueError("Unexpected batch size")
            if batch_size < 0:
                batch_size = len(nodes)
            return self.datamodule.custom_nodes_dataloader(
                nodes, batch_size=batch_size, sampler=sampler
            )
        else:
            raise NotImplementedError

    def _get_scores(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        split_conf_input: SplitConfInput,
    ):
        # calibration using score quantile
        # assuming that score is exchangeable, this should work
        # TODO: optimize the code here for the calls
        if self.conformal_method in [ConformalMethod.TPS, ConformalMethod.APS]:
            assert isinstance(split_conf_input, PrimitiveScoreConfig)
        elif self.conformal_method in [ConformalMethod.DAPS, ConformalMethod.DTPS]:
            assert isinstance(split_conf_input, DiffusionConfig)
            if self.conformal_method == ConformalMethod.DTPS:
                assert (
                    split_conf_input.use_tps_classwise
                ), f"Expected TPS classwise for DTPS"
        elif self.conformal_method == ConformalMethod.CFGNN:
            assert isinstance(split_conf_input, ConfGNNConfig)
        else:
            raise NotImplementedError

        if self.conformal_method in [ConformalMethod.TPS, ConformalMethod.DTPS]:
            self._score_module = TPSScore(split_conf_input, alpha=self.config.alpha)
        elif self.conformal_method in [ConformalMethod.APS, ConformalMethod.DAPS]:
            self._score_module = APSScore(split_conf_input, alpha=self.config.alpha)
        elif self.conformal_method == ConformalMethod.CFGNN:
            self._score_module = CFGNNScore(
                conf_config=self.config,
                confgnn_config=split_conf_input,
                datamodule=self.datamodule,
            )
        else:
            raise NotImplementedError

        best_params = {}
        if self.conformal_method in [ConformalMethod.DAPS, ConformalMethod.DTPS]:
            kwargs = {
                "dataset": self.config.dataset.name,
            }
            self._transform_module = DiffusionTransformation(split_conf_input, **kwargs)
            best_params = self._transform_module.find_params(
                probs, labels, self._score_module, self.datamodule
            )
            scores = self._score_module.pipe_compute(probs)
        elif self.conformal_method == ConformalMethod.CFGNN:
            if split_conf_input.load_probs:
                # use probabilities directly as features
                self.datamodule.update_features(probs)

            calib_tune_nodes = self.split_dict[Stage.CALIBRATION_TUNE]
            all_nodes = torch.arange(self.datamodule.num_nodes)
            calib_tune_dl = self.get_dataloader(
                calib_tune_nodes, self.config.batch_size
            )
            all_dl = self.get_dataloader(all_nodes, self.config.batch_size)

            scores = self._score_module.learn_params(calib_tune_dl, all_dl)
        else:
            scores = self._score_module.pipe_compute(probs)

        if self._transform_module is not None:
            kwargs = {"datamodule": self.datamodule, **best_params}
            scores = self._transform_module.transform(scores, **kwargs)
        return scores

    def _get_filter_mask(self, labels, groups, pos_label, group_id):
        assert (
            labels.shape[0] == groups.shape[0]
        ), f"Got {labels.shape[0]} labels, but {groups.shape[0]} groups"

        match self.config.fairness_metric:
            case fairness_metric.Equal_Opportunity.name:
                label_satisfied = labels == pos_label
                group_satisfied = groups == group_id
                return (label_satisfied & group_satisfied).reshape(-1)

            case fairness_metric.Predictive_Equality.name:
                label_not_satisfied = labels != pos_label
                group_satisfied = groups == group_id
                return (label_not_satisfied & group_satisfied).reshape(-1)

            case fairness_metric.Equalized_Odds.name:
                label_satisfied = labels == pos_label
                label_not_satisfied = labels != pos_label
                group_satisfied = groups == group_id

                return (
                    (label_satisfied & group_satisfied).reshape(-1),
                    (label_not_satisfied & group_satisfied).reshape(-1),
                )
            case (
                fairness_metric.Demographic_Parity.name
                | fairness_metric.Disparate_Impact.name
                | fairness_metric.Overall_Acc_Equality.name
            ):
                return (groups == group_id).reshape(-1)
            case _:
                raise NotImplementedError(
                    f"Filtering function not implemented for {self.config.fairness_metric}"
                )

    def _compute_qhat(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        split_conf_input: SplitConfInput,
    ):
        scores = self._cached_scores
        if scores is None:
            scores = self._get_scores(probs, labels, split_conf_input)
            self._cached_scores = scores
        assert self._score_module is not None

        label_scores = get_label_scores(
            labels,
            scores,
            self.split_dict[Stage.CALIBRATION_QSCORE],
            self.config.dataset.name,
        )
        # additional kwargs for tps
        if isinstance(self._score_module, TPSScore):
            kwargs = {
                "labels": labels[self.split_dict[Stage.CALIBRATION_QSCORE]],
                "num_classes": self.datamodule.num_classes,
            }
        else:
            kwargs = {}

        self._qhat = self._score_module.compute_quantile(label_scores, **kwargs)

    def _satisfies_lambda(
        self,
        labels: torch.Tensor,
        lmbda: torch.Tensor,
    ):
        num_groups = self.datamodule.num_sensitive_groups
        num_classes = self.datamodule.num_classes

        groups = self.datamodule.graph.ndata[SENS_FIELD]

        if self.config.fairness_metric not in [
            fairness_metric.Overall_Acc_Equality.name
        ]:
            classes = torch.arange(1, num_classes)
        else:
            # To make the looping more efficient force this to be a singleton,
            # since Overall Accuracy Equality doesn't fix a label
            classes = torch.arange(1)

        calib_mask = self.split_dict[Stage.CALIBRATION_QSCORE]  # n_nodes
        calib_groups = groups[calib_mask]
        calib_labels = labels[calib_mask]

        coverage_map = torch.zeros((num_groups, num_classes))
        interval_widths = torch.zeros((num_groups, num_classes))
        for group_id in torch.arange(num_groups):
            for label in classes:
                filtered_calib_mask = self._get_filter_mask(
                    calib_labels, calib_groups, label, group_id
                )

                if self.config.fairness_metric not in [
                    fairness_metric.Overall_Acc_Equality.name
                ]:
                    filtered_scores = self._cached_scores[calib_mask][
                        filtered_calib_mask
                    ][:, label]

                    coverage_map[group_id][label] = inverse_quantile(
                        filtered_scores, lmbda
                    )
                    interval_widths[group_id][label] = 1 / (
                        filtered_calib_mask.sum() + 1
                    )
                else:
                    filtered_scores = self._cached_scores[calib_mask][
                        filtered_calib_mask
                    ][
                        torch.arange(filtered_calib_mask.sum()),
                        calib_labels[filtered_calib_mask],
                    ]
                    coverage_map[group_id] = inverse_quantile(filtered_scores, lmbda)
                    interval_widths[group_id] = 1 / (filtered_calib_mask.sum() + 1)

        satisfying_arr = torch.ones(num_classes)
        for label in classes:
            alpha_min = max(0, min(coverage_map[:, label] - interval_widths[:, label]))
            alpha_max = max(coverage_map[:, label])
            if self.config.fairness_metric == fairness_metric.Disparate_Impact.name:
                if (1 - alpha_max) / (
                    1 - alpha_min
                ) < self.config.closeness_measure:  # Choose c = 0.8 for 80% rule
                    satisfying_arr[label] = 0
            else:
                if abs(alpha_max - alpha_min) > self.config.closeness_measure:
                    satisfying_arr[label] = 0

        if self.config.use_classwise_lambdas:
            return satisfying_arr * lmbda

        return torch.all(satisfying_arr) * lmbda

        # AVERAGE METHOD
        avg_coverage_map = torch.mean(coverage_map, dim=-1)
        avg_coverage_map_shifted = torch.mean(coverage_map - interval_widths, dim=-1)
        alpha_min = min(avg_coverage_map_shifted)
        alpha_max = max(avg_coverage_map)
        return abs(alpha_max - alpha_min) <= self.config.closeness_measure

    def _satisfies_lambda_pp(
        self, labels: torch.Tensor, lmbda: torch.Tensor, balance_npv: bool = False
    ):
        num_groups = self.datamodule.num_sensitive_groups
        num_classes = self.datamodule.num_classes

        groups = self.datamodule.graph.ndata[SENS_FIELD]
        classes = torch.arange(1, num_classes)

        calib_mask = self.split_dict[Stage.CALIBRATION_QSCORE]
        calib_groups = groups[calib_mask]
        calib_labels = labels[calib_mask]

        coverage_map = {
            "prior": torch.zeros((num_groups, num_classes)),
            fairness_metric.Equal_Opportunity.name: torch.zeros(
                (num_groups, num_classes)
            ),
            fairness_metric.Demographic_Parity.name: torch.zeros(
                (num_groups, num_classes)
            ),
        }

        interval_widths = {
            "prior": torch.zeros((num_groups, num_classes)),
            fairness_metric.Equal_Opportunity.name: torch.zeros(
                (num_groups, num_classes)
            ),
            fairness_metric.Demographic_Parity.name: torch.zeros(
                (num_groups, num_classes)
            ),
        }
        for group_id in torch.arange(num_groups):
            for label in classes:
                try:
                    for key in coverage_map:
                        if key == "prior":
                            self.config.fairness_metric = (
                                fairness_metric.Demographic_Parity.name
                            )
                            filtered_calib_mask = self._get_filter_mask(
                                calib_labels, calib_groups, label, group_id
                            )

                            coverage_map[key][group_id][label] = (
                                calib_labels[filtered_calib_mask] == label
                            ).sum() / (filtered_calib_mask.sum() + 1)

                            interval_widths[key][group_id][label] = 1 / (
                                filtered_calib_mask.sum() + 1
                            )
                        else:
                            self.config.fairness_metric = key
                            filtered_calib_mask = self._get_filter_mask(
                                calib_labels, calib_groups, label, group_id
                            )

                            filtered_scores = self._cached_scores[calib_mask][
                                filtered_calib_mask
                            ][:, label]

                            coverage_map[key][group_id][label] = inverse_quantile(
                                filtered_scores, lmbda
                            )
                            interval_widths[key][group_id][label] = 1 / (
                                filtered_calib_mask.sum() + 1
                            )
                finally:
                    self.config.fairness_metric = fairness_metric.Predictive_Parity.name

        satisfying_arr = torch.ones(num_classes)
        for label in classes:
            coverage_map[fairness_metric.Equal_Opportunity.name][:, label]

            eo_min = (
                coverage_map[fairness_metric.Equal_Opportunity.name][:, label]
                - interval_widths[fairness_metric.Equal_Opportunity.name][:, label]
            )

            eo_max = coverage_map[fairness_metric.Equal_Opportunity.name][:, label]

            dp_min = (
                coverage_map[fairness_metric.Demographic_Parity.name][:, label]
                - interval_widths[fairness_metric.Demographic_Parity.name][:, label]
            )

            dp_max = coverage_map[fairness_metric.Demographic_Parity.name][:, label]

            pr_min = coverage_map["prior"][:, label]
            pr_max = (
                coverage_map["prior"][:, label] + interval_widths["prior"][:, label]
            )

            # if (
            #     balance_npv
            #     and abs((eo_max) * pr_max / (dp_min) - (eo_min) * pr_min / (dp_max))
            #     > self.config.closeness_measure
            # ):
            #     satisfying_arr[label] = 0

            ppr_max = (1 - eo_min) * pr_max / (1 - dp_max)

            ppr_min = (1 - eo_max) * pr_min / (1 - dp_min)

            if (
                abs(max((ppr_max - pr_min)) - min((ppr_min - pr_max)))
                > self.config.closeness_measure
            ):
                satisfying_arr[label] = 0

        if self.config.use_classwise_lambdas:
            return satisfying_arr * lmbda

        return torch.all(satisfying_arr) * lmbda

        # AVERAGE METHOD
        # TODO: Classwise version
        avg_coverage_map = torch.mean(coverage_map, dim=-1)
        avg_coverage_map_shifted = torch.mean(coverage_map - interval_widths, dim=-1)
        alpha_min = min(avg_coverage_map_shifted)
        alpha_max = max(avg_coverage_map)
        return abs(alpha_max - alpha_min) <= self.config.closeness_measure

    def run(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        split_conf_input: SplitConfInput,
    ):
        self._compute_qhat(probs, labels, split_conf_input)
        upper_bound_lambda = max(
            1, math.ceil(1000 * torch.max(self._cached_scores)) / 1000
        )
        lower_bound_lambda = math.ceil(1000 * torch.min(self._qhat)) / 1000

        sort_res = (
            self._cached_scores[self._cached_scores >= torch.min(self._qhat)]
            .reshape(-1)
            .unique()
            .sort()
        )

        lambdas = torch.cat(
            (
                sort_res.values[:: max(1, len(sort_res.values) // 500)],
                torch.max(self._cached_scores).reshape(-1),
            )
        )

        print(f"Num Lambdas: {len(lambdas)}")

        # lambdas = torch.arange(
        #     lower_bound_lambda,
        #     upper_bound_lambda + self.step_size,
        #     step=self.step_size,
        # )

        match self.config.fairness_metric:
            case fairness_metric.Equalized_Odds.name:
                try:
                    self.config.fairness_metric = fairness_metric.Equal_Opportunity.name
                    eo_satisfying_lambdas = torch.stack(
                        [self._satisfies_lambda(labels, lmbda) for lmbda in lambdas]
                    )
                    self.config.fairness_metric = (
                        fairness_metric.Predictive_Equality.name
                    )
                    pe_satisfying_lambdas = torch.stack(
                        [self._satisfies_lambda(labels, lmbda) for lmbda in lambdas]
                    )
                finally:
                    self.config.fairness_metric = fairness_metric.Equalized_Odds.name

                satisfying_lambdas = torch.sqrt(
                    eo_satisfying_lambdas * pe_satisfying_lambdas
                )
            case (
                fairness_metric.Equal_Opportunity.name
                | fairness_metric.Predictive_Equality.name
                | fairness_metric.Demographic_Parity.name
                | fairness_metric.Disparate_Impact.name
                | fairness_metric.Overall_Acc_Equality.name
            ):
                satisfying_lambdas = torch.stack(
                    [self._satisfies_lambda(labels, lmbda) for lmbda in lambdas],
                )

            case fairness_metric.Conditional_Use_Acc_Equality.name:
                try:
                    self.config.fairness_metric = fairness_metric.Predictive_Parity.name
                    pp_satisfying_lambdas = torch.stack(
                        [self._satisfies_lambda_pp(labels, lmbda) for lmbda in lambdas]
                    )

                    npp_satisfying_lambdas = torch.stack(
                        [
                            self._satisfies_lambda_pp(labels, lmbda, balance_npv=True)
                            for lmbda in lambdas
                        ]
                    )
                finally:
                    self.config.fairness_metric = fairness_metric.Equalized_Odds.name

                satisfying_lambdas = torch.sqrt(
                    pp_satisfying_lambdas * npp_satisfying_lambdas
                )
            case fairness_metric.Predictive_Parity.name:
                satisfying_lambdas = torch.stack(
                    [self._satisfies_lambda_pp(labels, lmbda) for lmbda in lambdas]
                )
            case _:
                raise NotImplementedError(
                    f"CP Algorithm not implemented for {self.config.fairness_metric}"
                )

        if len(satisfying_lambdas.shape) < 2:
            satisfying_lambdas = satisfying_lambdas.reshape(-1, 1)
        # Choose the smallest lambda s.t. normal CP coverage is satisfied
        mask = (satisfying_lambdas >= self._qhat) * 1

        self.lambda_hat = []
        for col in range(mask.shape[-1]):
            l_idx = torch.argmax(mask[:, col])

            # Edge case of no lambda satisfying.
            if l_idx == 0 and mask[l_idx, col] == 0:
                l_idx = -1

            self.lambda_hat.append(lambdas[l_idx])

        self.lambda_hat = torch.stack(self.lambda_hat)

        print(f"q_hat: {self._qhat}")
        print(f"lambda_hat: {self.lambda_hat.tolist()}")

        assert self._cached_scores is not None

        test_labels = labels[self.split_dict[Stage.TEST]]
        test_scores = self._cached_scores[self.split_dict[Stage.TEST]]

        # Scores could have been implemented as a pipeline of transformations
        prediction_sets = PredSetTransformation(
            threshold=self.lambda_hat
        ).pipe_transform(test_scores)

        baseline_prediction_sets = PredSetTransformation(
            threshold=self._qhat
        ).pipe_transform(test_scores)

        return prediction_sets, test_labels, baseline_prediction_sets
