from typing import Dict

import torch

from .config import (
    ConfFairExptConfig,
    DiffusionConfig,
    NeighborhoodConfig,
    PrimitiveScoreConfig,
    RegularizedConfig,
    SplitConfInput,
)
from .constants import SENS_FIELD, ConformalMethod, Stage, fairness_metric
from .crc_loss import (
    DemographicParityLoss,
    EqualizedOddsLoss,
    EqualOpportunityLoss,
    PredictiveEqualityLoss,
    PredictiveParityLoss,
)
from .data_module import DataModule
from .scores import APSScore, NAPSScore, TPSScore
from .transformations import (
    DiffusionTransformation,
    PredSetTransformation,
    RegularizationTransformation,
)


class ConformalRiskController:
    def __init__(self, config: ConfFairExptConfig, **kwargs):
        self.config = config  # coverage req
        self.lambda_hat = None  # tunable parameter
        self.B = kwargs.get("B", 1)
        self.num_lams: int = kwargs.get("num_lams", 1001)
        self.lambdas = torch.linspace(0, 1, steps=int(self.num_lams))

    def get_lambda_hat(self, losses, prior=1, alpha=None):
        "From Risk Control Paper: https://github.com/aangelopoulos/conformal-risk"
        assert torch.all(
            torch.argsort(self.lambdas) == torch.arange(self.num_lams)
        ), f"Expected self.lambdas to be in sorted ascending order"
        n = losses.shape[0]
        rhat = losses.mean(axis=0).reshape(-1)

        if alpha is None:
            alpha = self.config.alpha

        # Recall that rhat (as a vector from left to right) is non-increasing (since lambdas are increasing)
        # Argmax will return the first True index (since the input is a 0-1 vector).
        # We then subtract by 1 to get the largest guaranteed lower bound (infimum) considering new test points.
        satisfying_rhats = (
            rhat * (n / (n + 1)) + self.B / (n + 1) <= (alpha * prior)
        ) * 1

        lhat_idx = torch.argmax(satisfying_rhats)

        # If lhat_idx == 0, then either all or none of lambdas satisfy Equation 4 (from the paper).
        # If non satsify, we set lhat_idx = -1 (which corresponds to lambda_hat = 1), otherwise, we leave as 0
        if lhat_idx == 0 and not satisfying_rhats[0]:
            lhat_idx = -1

        return self.lambdas[lhat_idx]


class ConformalRiskClassifier(ConformalRiskController):
    def __init__(self, config: ConfFairExptConfig, datamodule: DataModule, **kwargs):
        super().__init__(config, **kwargs)
        self.datamodule = datamodule


class SplitConformalRiskClassifier(ConformalRiskClassifier):
    def __init__(self, config: ConfFairExptConfig, datamodule: DataModule, **kwargs):
        super().__init__(config, datamodule, **kwargs)

    def calibrate(self, **calib_data):
        """Calibrate the conformal risk classifier"""
        raise NotImplementedError


class ScoreSplitConformalRiskClassifier(SplitConformalRiskClassifier):
    """A score based split conformal risk classifier"""

    def __init__(self, config: ConfFairExptConfig, datamodule: DataModule, **kwargs):
        super().__init__(config, datamodule, **kwargs)
        self.conformal_method = ConformalMethod(config.conformal_method)
        self.split_dict: Dict[Stage, torch.LongTensor] = datamodule.split_dict
        self._lambda_hat = None
        self._score_module = None

        match config.fairness_metric:
            case fairness_metric.Equal_Opportunity.name:
                self._loss_module = EqualOpportunityLoss()
            case fairness_metric.Equalized_Odds.name:
                self._loss_module = EqualizedOddsLoss()
            case fairness_metric.Predictive_Parity.name:
                self._loss_module = PredictiveParityLoss()
            case fairness_metric.Predictive_Equality.name:
                self._loss_module = PredictiveEqualityLoss()
            case fairness_metric.Demographic_Parity.name:
                self._loss_module = DemographicParityLoss()
            case _:
                raise NotImplementedError()

        self._transform_module = None
        self._cached_scores = None

    def _get_scores(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        split_conf_input: SplitConfInput,
    ):
        # calibration using score quantile
        # assuming that score is exchangeable, this should work
        if self.conformal_method in [ConformalMethod.TPS, ConformalMethod.APS]:
            assert isinstance(split_conf_input, PrimitiveScoreConfig)
        elif self.conformal_method == ConformalMethod.NAPS:
            assert isinstance(split_conf_input, NeighborhoodConfig)
        elif self.conformal_method in [ConformalMethod.DAPS, ConformalMethod.DTPS]:
            assert isinstance(split_conf_input, DiffusionConfig)
        elif self.conformal_method == ConformalMethod.RAPS:
            assert isinstance(split_conf_input, RegularizedConfig)
        else:
            raise NotImplementedError()

        if self.conformal_method == ConformalMethod.TPS:
            self._score_module = TPSScore(split_conf_input, alpha=self.config.alpha)
        elif self.conformal_method == ConformalMethod.APS:
            self._score_module = APSScore(split_conf_input, alpha=self.config.alpha)
        elif self.conformal_method == ConformalMethod.NAPS:
            self._score_module = NAPSScore(
                split_conf_input, datamodule=self.datamodule, alpha=self.config.alpha
            )  # Implementation in Paper uses APS Score
        elif self.conformal_method in [ConformalMethod.DAPS, ConformalMethod.DTPS]:
            primitive_config = PrimitiveScoreConfig(
                use_aps_epsilon=split_conf_input.use_aps_epsilon,
                use_tps_classwise=split_conf_input.use_tps_classwise,
            )
            if self.conformal_method == ConformalMethod.DAPS:
                self._score_module = APSScore(primitive_config, alpha=self.config.alpha)
            elif self.conformal_method == ConformalMethod.DTPS:
                self._score_module = TPSScore(primitive_config, alpha=self.config.alpha)

            kwargs = {"dataset": self.config.dataset.name}
            self._transform_module = DiffusionTransformation(split_conf_input, **kwargs)

            best_params = self._transform_module.find_params(
                probs,
                labels,
                self._score_module,
                self.datamodule,
            )

            scores = self._score_module.pipe_compute(probs)
            scores = self._transform_module.transform(
                scores,
                datamodule=self.datamodule,
                diffusion_param=best_params["diffusion_param"],
            )

            return scores
        elif self.conformal_method == ConformalMethod.RAPS:
            primitive_config = PrimitiveScoreConfig(
                use_aps_epsilon=False,
                use_tps_classwise=split_conf_input.use_tps_classwise,
            )
            self._score_module = APSScore(primitive_config, alpha=self.config.alpha)

            self._transform_module = RegularizationTransformation(
                config=split_conf_input
            )

            best_params = self._transform_module.find_params(
                probs,
                labels,
                self._score_module,
                self.datamodule,
            )
            scores = self._score_module.pipe_compute(probs)
            scores = self._transform_module.transform(
                scores,
                probs,
                raps_modified=split_conf_input.raps_mod,
                **best_params,
            )
            return scores
        else:
            raise NotImplementedError()

        scores = self._score_module.pipe_compute(probs)

        if self._transform_module is not None:
            scores = self._transform_module.pipe_transform(scores)
        return scores

    def _compute_prior(self, labels, groups, pos_label, group_id, scores=None):
        assert (
            labels.shape[0] == groups.shape[0]
        ), f"Got {labels.shape[0]} labels, but {groups.shape[0]} groups"
        n = groups.shape[0]

        # Using (n + 1) in all returns because it gives lower bound on calibration_tune set which also satisfies exchangeability with new points
        match self.config.fairness_metric:
            case fairness_metric.Equal_Opportunity.name:
                label_satisfied = torch.where(labels == pos_label, True, False)
                group_satisfied = torch.where(groups == group_id, True, False)

                return sum(torch.bitwise_and(label_satisfied, group_satisfied)) / (
                    n + 1
                )
            case fairness_metric.Predictive_Equality.name:
                label_not_satisfied = torch.where(labels != pos_label, True, False)
                group_satisfied = torch.where(groups == group_id, True, False)

                return sum(torch.bitwise_and(label_not_satisfied, group_satisfied)) / (
                    n + 1
                )
            case fairness_metric.Equalized_Odds.name:
                label_satisfied = torch.where(labels == pos_label, True, False)
                label_not_satisfied = torch.where(labels != pos_label, True, False)
                group_satisfied = torch.where(groups == group_id, True, False)

                return (
                    sum(torch.bitwise_and(label_satisfied, group_satisfied)) / (n + 1),
                    sum(torch.bitwise_and(label_not_satisfied, group_satisfied))
                    / (n + 1),
                )
            case fairness_metric.Demographic_Parity.name:
                return sum(torch.where(groups == group_id, True, False)) / (n + 1)
            case fairness_metric.Predictive_Parity.name:
                assert (
                    scores is not None
                ), f"Missing scores for {fairness_metric.Predictive_Parity.name}"

                assert (
                    scores.shape[0] == groups.shape[0]
                ), f"Got {scores.shape[0].shape[0]} scores, but {groups.shape[0]} groups"

                group_satisfied = torch.where(groups == group_id, True, False).reshape(
                    -1, 1
                )
                condition_satisfied = torch.zeros((n, self.num_lams), dtype=torch.bool)
                for i, lmbda in enumerate(self.lambdas):
                    pred_set = PredSetTransformation(threshold=lmbda).pipe_transform(
                        scores
                    )
                    condition_satisfied[:, i] = pred_set[:, pos_label]

                prior = torch.sum(
                    torch.bitwise_and(condition_satisfied, group_satisfied), dim=0
                ) / (n + 1)

                if self._lambda_hat is not None:
                    return prior[
                        torch.where(self.lambdas == self._lambda_hat, True, False)
                    ][0].item()
                return prior
            case _:
                raise NotImplementedError(
                    f"Prior computation not implemented for {self.config.fairness_metric}"
                )

    def calibrate(
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

        pos_labels = range(1, self.datamodule.num_classes)
        groups = self.datamodule.graph.ndata[SENS_FIELD]
        lambda_hats = [1]
        for pos_label in pos_labels:
            lambda_hats_per_label = []
            for g_i in range(self.datamodule.num_sensitive_groups):
                prior = self._compute_prior(
                    labels=labels[self.split_dict[Stage.CALIBRATION_TUNE]],
                    groups=groups[self.split_dict[Stage.CALIBRATION_TUNE]],
                    pos_label=pos_label,
                    group_id=g_i,
                    scores=scores[self.split_dict[Stage.CALIBRATION_TUNE]],
                )

                losses = self._loss_module.compute(
                    scores=scores[self.split_dict[Stage.CALIBRATION_QSCORE]],
                    labels=labels[self.split_dict[Stage.CALIBRATION_QSCORE]],
                    groups=groups[self.split_dict[Stage.CALIBRATION_QSCORE]],
                    lambdas=self.lambdas,
                    pos_label=pos_label,
                    group_id=g_i,
                    num_classes=self.datamodule.num_classes,
                )

                if self.config.fairness_metric in [
                    fairness_metric.Equalized_Odds.name,
                    fairness_metric.Conditional_Use_Acc_Equality.name,
                ]:
                    for l, p in zip(losses, prior):
                        l_hat = self.get_lambda_hat(losses=l, prior=p)
                        lambda_hats_per_label.append(l_hat)
                else:
                    l_hat = self.get_lambda_hat(losses=losses, prior=prior)
                    lambda_hats_per_label.append(l_hat)
                    # print(
                    #     f"y_k={pos_label}, g_i={g_i} with prior={prior} has lambda_hat={l_hat}"
                    # )

            lambda_hats.append(max(lambda_hats_per_label))
        print()
        if self.config.use_classwise_lambdas:
            return torch.as_tensor(lambda_hats)
        return max(lambda_hats)

    def run(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        split_conf_input: SplitConfInput,
    ):
        self._lambda_hat = self.calibrate(probs, labels, split_conf_input)
        assert self._cached_scores is not None and self._lambda_hat is not None

        test_labels = labels[self.split_dict[Stage.TEST]]
        test_scores = self._cached_scores[self.split_dict[Stage.TEST]]

        prediction_sets = PredSetTransformation(
            threshold=self._lambda_hat
        ).pipe_transform(test_scores)

        return prediction_sets, test_labels
