import logging

import torch

from .constants import conf_metric_names


# verification
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
        case conf_metric_names.set_sizes.name:
            return set_sizes(prediction_sets)
        case conf_metric_names.coverage.name:
            return calc_coverage(prediction_sets, labels)
        case conf_metric_names.efficiency.name:
            return calc_efficiency(prediction_sets)
        case conf_metric_names.feature_stratified_coverage.name:
            return calc_feature_stratified_coverage(prediction_sets, features, labels)
        case conf_metric_names.size_stratified_coverage.name:
            return calc_size_stratified_coverage(prediction_sets, labels)
        case conf_metric_names.label_stratified_coverage.name:
            return calc_label_stratified_coverage(prediction_sets, labels)
        case conf_metric_names.singleton_hit_ratio.name:
            return singleton_hit_ratio(prediction_sets, labels)
        case conf_metric_names.size_stratified_coverage_violation.name:
            if alpha is None:
                logging.warn(
                    "Size stratified coverage violation requires alpha to be set"
                )
            return calc_size_stratified_coverage_violation(
                prediction_sets, labels, alpha
            )
        case _:
            logging.warn(f"Metric not implemented: {metric}")
