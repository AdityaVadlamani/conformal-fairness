import logging
from typing import Optional, Tuple

import torch

from ..config import ConfExptConfig, ConfFairExptConfig
from ..conformal_predictors.fair_predictor import ScoreSplitFairConformalClassifer
from ..conformal_predictors.predictor import (
    ConformalMethod,
    ScoreMultiSplitConformalClassifier,
    ScoreSplitConformalClassifer,
)
from ..constants import *
from ..custom_logger import CustomLogger
from ..data import BaseDataModule
from .ml_utils import (
    load_basegnn_outputs,
    set_conf_ckpt_dir_fname,
    set_trained_basegnn_path,
)

logging.basicConfig(level=logging.INFO)


# region CONFORMAL PREDICTION METRICS
def set_sizes(prediction_sets):
    return prediction_sets.sum(dim=1)


def calc_coverage(prediction_sets, labels):
    includes_true_label = prediction_sets.gather(1, labels.unsqueeze(1)).squeeze()
    empirical_coverage = includes_true_label.sum() / len(prediction_sets)
    return empirical_coverage


def calc_efficiency(prediction_sets):
    empirical_efficiency = set_sizes(prediction_sets).sum() / len(prediction_sets)
    return empirical_efficiency


def _conditional_coverages(prediction_sets, conditions, labels):
    # conditions: a tensor of shape (n) ints with each int representing one conditions
    # eg. could represent sensitive node properties
    covered_pts = prediction_sets.gather(1, labels.unsqueeze(1)).squeeze()
    # collect the number of covered pts in each group
    # TODO: make more efficient, differentiable
    # group_sizes = torch.bincount(conditions)
    # TODO: Could be used as a loss function
    # groupwise_covered_pts = torch.scatter_add(covered_pts, conditions, group_sizes)
    groupwise_coverages = []
    for i in range(conditions.max() + 1):
        group_covered_pts = torch.sum(covered_pts[conditions == i])
        group_size = torch.sum(conditions == i)
        groupwise_coverages.append(group_covered_pts / max(group_size.item(), 1))
    groupwise_coverages = torch.tensor(groupwise_coverages)
    return groupwise_coverages


def calc_feature_stratified_coverage(prediction_sets, features, labels):
    if features is None:
        return None
    # TODO Assumes that the feature is an integer value - bin it prior
    groupwise_coverages = _conditional_coverages(prediction_sets, features, labels)
    return torch.mean(groupwise_coverages)


def calc_size_stratified_coverage(prediction_sets, labels):
    sizes = prediction_sets.sum(dim=1)
    groupwise_coverages = _conditional_coverages(prediction_sets, sizes, labels)
    return torch.mean(groupwise_coverages)


def calc_label_stratified_coverage(prediction_sets, labels):
    groupwise_coverages = _conditional_coverages(prediction_sets, labels, labels)
    return torch.mean(groupwise_coverages)


def singleton_hit_ratio(prediction_sets, labels):
    set_size_vals = set_sizes(prediction_sets)
    singleton_labels = labels[set_size_vals == 1]
    singleton_preds = prediction_sets[set_size_vals == 1].nonzero(as_tuple=True)[1]
    return (singleton_labels == singleton_preds).sum() / max(len(singleton_labels), 1)


def calc_size_stratified_coverage_violation(prediction_sets, labels, alpha):
    sizes = set_sizes(prediction_sets)
    groupwise_coverages = _conditional_coverages(prediction_sets, sizes, labels)
    return torch.max(torch.abs(groupwise_coverages - (1 - alpha)))


def compute_metric(metric, prediction_sets, labels, alpha=None, features=None):
    match metric:
        case ConformalMetric.SET_SIZES.value:
            return set_sizes(prediction_sets)
        case ConformalMetric.COVERAGE.value:
            return calc_coverage(prediction_sets, labels)
        case ConformalMetric.EFFICIENCY.value:
            return calc_efficiency(prediction_sets)
        case ConformalMetric.FEATURE_STRATIFIED_COVERAGE.value:
            return calc_feature_stratified_coverage(prediction_sets, features, labels)
        case ConformalMetric.SIZE_STRATIFIED_COVERAGE.value:
            return calc_size_stratified_coverage(prediction_sets, labels)
        case ConformalMetric.LABEL_STRATIFIED_COVERAGE.value:
            return calc_label_stratified_coverage(prediction_sets, labels)
        case ConformalMetric.SINGLETON_HIT_RATIO.value:
            return singleton_hit_ratio(prediction_sets, labels)
        case ConformalMetric.SIZE_STRATIFIED_COVERAGE_VIOLATION.value:
            if alpha is None:
                logging.warning(
                    "Size stratified coverage violation requires alpha to be set"
                )
            return calc_size_stratified_coverage_violation(
                prediction_sets, labels, alpha
            )
        case _:
            logging.warning(f"Metric not implemented: {metric}")


# endregion


# region CONFORMAL PREDICTION EXECUTORS


def run_conformal(
    args: ConfExptConfig,
    datamodule: BaseDataModule,
    expt_logger: CustomLogger,
    base_ckpt_dir: Optional[str] = None,
):
    # Load probs/labels from base
    probs, labels = load_basegnn_outputs(args, base_ckpt_dir)
    assert (
        probs.shape[1] == datamodule.num_classes
    ), f"Loaded probs has {probs.shape[1]} classes, but the dataset has {datamodule.num_classes} classes"

    # note that splits are setup but not sampler
    conformal_method = ConformalMethod(args.conformal_method)
    split_conf_input = get_split_conf(args)
    match conformal_method:
        case ConformalMethod.TPS | ConformalMethod.APS | ConformalMethod.NAPS:
            cp = ScoreSplitConformalClassifer(config=args, datamodule=datamodule)

            pred_sets, test_labels = cp.run(
                probs=probs,
                labels=labels,
                split_conf_input=split_conf_input,
            )

        case ConformalMethod.DAPS | ConformalMethod.DTPS | ConformalMethod.RAPS:
            cp = ScoreMultiSplitConformalClassifier(config=args, datamodule=datamodule)

            pred_sets, test_labels = cp.run(
                probs=probs, labels=labels, split_conf_input=split_conf_input
            )

            if cp.best_params is not None:
                expt_logger.log_hyperparams(cp.best_params)

        case ConformalMethod.CFGNN:
            assert (
                args.confgnn_config is not None
            ), f"confgnn_config cannot be None for CFGNN"
            _ = set_trained_basegnn_path(args, base_ckpt_dir)
            _, _ = set_conf_ckpt_dir_fname(args, conformal_method.value)
            cp = ScoreMultiSplitConformalClassifier(config=args, datamodule=datamodule)

            pred_sets, test_labels = cp.run(
                split_conf_input=split_conf_input, logger=expt_logger, probs=probs
            )

        case _:
            raise NotImplementedError
    return pred_sets, test_labels


def run_conformal_fairness(
    args: ConfFairExptConfig,
    datamodule: BaseDataModule,
    base_ckpt_dir: Optional[str] = None,
):
    # Load probs/labels from base for all (not necessarily labeled) points
    probs, labels = load_basegnn_outputs(args, base_ckpt_dir)
    assert (
        probs.shape[1] == datamodule.num_classes
    ), f"Loaded probs has {probs.shape[1]} classes, but the dataset has {datamodule.num_classes} classes"

    test_mask = datamodule.split_dict[Stage.TEST]  # Test mask of shape (n_points,)

    # Perform Conformal Fairness
    split_conf_input = get_split_conf(args)

    cpf = ScoreSplitFairConformalClassifer(args, datamodule)

    groups = datamodule.sens
    N = len(labels[test_mask])
    test_labels = labels[test_mask]
    test_groups = groups[test_mask]
    pred_sets = torch.zeros((N, datamodule.num_classes), dtype=torch.bool)
    pred_sets[torch.arange(N), probs[test_mask].argmax(dim=-1)] = True
    print(
        f"Model Test Accuracy with conformal seed {args.conformal_seed}: {calc_coverage(pred_sets, test_labels)}"
    )

    pred_sets, test_labels, baseline_pred_sets = cpf.run(
        probs=probs,
        labels=labels,
        split_conf_input=split_conf_input,
    )

    print(f"Mean Efficiency: {calc_efficiency(pred_sets)}")
    print(f"Coverage: {calc_coverage(pred_sets, test_labels)}\n")

    print(f"Mean Baseline Efficiency: {calc_efficiency(baseline_pred_sets)}")
    print(f"Baseline Coverage: {calc_coverage(baseline_pred_sets, test_labels)}\n")

    labels = torch.arange(1, datamodule.num_classes)

    print(f"Expected Coverage Delta/Ratio={args.closeness_measure}\n")
    res = {
        "c": args.closeness_measure,
        "base_eff": calc_efficiency(baseline_pred_sets),
        "base_coverage": calc_coverage(baseline_pred_sets, test_labels),
        "base_violation": (
            float("inf")
            if args.fairness_metric == FairnessMetric.DISPARATE_IMPACT.value
            else float("-inf")
        ),
        "eff": calc_efficiency(pred_sets),
        "coverage": calc_coverage(pred_sets, test_labels),
        "violation": (
            float("inf")
            if args.fairness_metric == FairnessMetric.DISPARATE_IMPACT.value
            else float("-inf")
        ),
    }
    for label in labels:
        coverages = []
        baseline_coverages = []
        for g_i in range(datamodule.num_sensitive_groups):
            match args.fairness_metric:
                case (
                    FairnessMetric.EQUAL_OPPORTUNITY.value
                    | FairnessMetric.PREDICTIVE_EQUALITY.value
                    | FairnessMetric.EQUALIZED_ODDS.value
                    | FairnessMetric.DEMOGRAPHIC_PARITY.value
                    | FairnessMetric.DISPARATE_IMPACT.value
                    | FairnessMetric.OVERALL_ACC_EQUALITY.value
                ):
                    filtered_test_mask = cpf._get_filter_mask(
                        test_labels, test_groups, label, g_i
                    )

                    if isinstance(filtered_test_mask, Tuple):
                        temp_cov = []
                        temp_base_cov = []
                        for mask in filtered_test_mask:
                            cov_labels = torch.full_like(test_labels[mask], label)
                            cov = calc_coverage(pred_sets[mask, :], cov_labels)
                            temp_cov.append(cov)

                            base_cov = calc_coverage(
                                baseline_pred_sets[mask, :], cov_labels
                            )
                            temp_base_cov.append(base_cov)

                            print(
                                f"Positive Label Coverage for y_k = {label} and g_i = {g_i} = {cov}"
                            )
                        coverages.append(temp_cov)
                        baseline_coverages.append(temp_base_cov)
                    else:
                        if (
                            args.fairness_metric
                            != FairnessMetric.OVERALL_ACC_EQUALITY.value
                        ):
                            cov_labels = torch.full_like(
                                test_labels[filtered_test_mask], label
                            )
                        else:
                            cov_labels = test_labels[filtered_test_mask]

                        cov = calc_coverage(
                            pred_sets[filtered_test_mask, :], cov_labels
                        )
                        coverages.append(cov)

                        base_cov = calc_coverage(
                            baseline_pred_sets[filtered_test_mask, :], cov_labels
                        )
                        baseline_coverages.append(base_cov)
                        print(
                            f"Positive Label Coverage for y_k = {label} and g_i = {g_i} = {cov}"
                        )
                case FairnessMetric.PREDICTIVE_PARITY.value:
                    try:
                        cpf.config.fairness_metric = (
                            FairnessMetric.EQUAL_OPPORTUNITY.value
                        )
                        eo_filtered_test_mask = cpf._get_filter_mask(
                            test_labels, test_groups, label, g_i
                        )

                        cpf.config.fairness_metric = (
                            FairnessMetric.DEMOGRAPHIC_PARITY.value
                        )
                        dp_filtered_test_mask = cpf._get_filter_mask(
                            test_labels, test_groups, label, g_i
                        )
                    finally:
                        cpf.config.fairness_metric = (
                            FairnessMetric.PREDICTIVE_PARITY.value
                        )

                    pos_labels = torch.full_like(
                        test_labels[eo_filtered_test_mask], label
                    )
                    eo_coverage = calc_coverage(
                        pred_sets[eo_filtered_test_mask, :], pos_labels
                    )
                    eo_base_coverage = calc_coverage(
                        baseline_pred_sets[eo_filtered_test_mask, :], pos_labels
                    )
                    pos_labels = torch.full_like(
                        test_labels[dp_filtered_test_mask], label
                    )
                    dp_coverage = calc_coverage(
                        pred_sets[dp_filtered_test_mask, :], pos_labels
                    )
                    dp_base_coverage = calc_coverage(
                        baseline_pred_sets[dp_filtered_test_mask, :], pos_labels
                    )

                    prior = (test_labels[dp_filtered_test_mask] == pos_labels).sum() / (
                        dp_filtered_test_mask.sum()
                    )

                    cov = abs(eo_coverage * prior / dp_coverage - prior)
                    coverages.append(cov)

                    base_cov = abs(eo_base_coverage * prior / dp_base_coverage - prior)
                    baseline_coverages.append(base_cov)

                    print(
                        f"Positive Label Coverage for y_k = {label} and g_i = {g_i} = {cov}"
                    )

        if args.fairness_metric == FairnessMetric.DISPARATE_IMPACT.value:
            print(f"Actual Coverage Ratio={min(coverages) / max(coverages)}\n")
            res["violation"] = min(min(coverages) / max(coverages), res["violation"])
            res["base_violation"] = min(
                min(baseline_coverages) / max(baseline_coverages), res["base_violation"]
            )
        elif args.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
            print(
                f"Actual Coverage Delta = {max(max([x[0] for x in coverages]) - min([x[0] for x in coverages]), max([x[1] for x in coverages]) - min([x[1] for x in coverages]))}\n"
            )

            res["violation"] = max(
                max(
                    (
                        max([x[0] for x in coverages]) - min([x[0] for x in coverages]),
                        max([x[1] for x in coverages]) - min([x[1] for x in coverages]),
                    )
                ),
                res["violation"],
            )

            res["base_violation"] = max(
                max(
                    (
                        max([x[0] for x in baseline_coverages])
                        - min([x[0] for x in baseline_coverages]),
                        max([x[1] for x in baseline_coverages])
                        - min([x[1] for x in baseline_coverages]),
                    )
                ),
                res["base_violation"],
            )

        else:
            print(f"Actual Coverage Delta={max(coverages) - min(coverages)}\n")
            res["violation"] = max(max(coverages) - min(coverages), res["violation"])
            res["base_violation"] = max(
                max(baseline_coverages) - min(baseline_coverages), res["base_violation"]
            )

    return pred_sets, test_labels, res


# endregion


def get_split_conf(args: ConfExptConfig):
    conformal_method = ConformalMethod(args.conformal_method)
    match conformal_method:
        case ConformalMethod.TPS | ConformalMethod.APS | ConformalMethod.NAPS:
            split_conf_input = args.primitive_config
            if conformal_method == ConformalMethod.NAPS:
                split_conf_input = args.neighborhood_config
        case ConformalMethod.DAPS | ConformalMethod.DTPS | ConformalMethod.RAPS:
            split_conf_input = (
                args.raps_config
                if conformal_method == ConformalMethod.RAPS
                else args.diffusion_config
            )
        case ConformalMethod.CFGNN:
            split_conf_input = args.confgnn_config
        case _:
            raise NotImplementedError
    return split_conf_input


def inverse_quantile(x, target):
    n = x.shape[0]
    sorted_x = torch.sort(x)

    # Get the index of the largest value less than or equal to the target.
    # Add 1 to make it 1-indexed, instead of 0-indexed
    satisfied = torch.where(sorted_x.values <= target)[0].reshape(-1)
    if len(satisfied) > 0:
        return 1 - (satisfied[-1] + 1) / (n + 1)
    return 1
