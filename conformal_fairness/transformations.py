import math
from abc import ABC
from typing import List

import pandas as pd
import torch

from .conf_metrics import compute_metric
from .config import DiffusionConfig, RegularizedConfig
from .constants import DEFAULT_DEVICE, Stage, conf_metric_names
from .data_module import DataModule
from .data_utils import get_label_scores
from .scores import APSScore, CPScore, TPSScore


class Transformation(ABC):
    def __init__(self, **kwargs):
        self.defined_args = kwargs

    def pipe_transform(self, x):
        return self.transform(x, **self.defined_args)

    def transform(self, x, **kwargs):
        return x


class PredSetTransformation(Transformation):
    def transform(self, x, **kwargs):
        threshold = kwargs.get("threshold")
        return x <= threshold

    def group_transform(self, x, groups, **kwargs):
        threshold = kwargs.get("threshold")
        unique_groups = torch.unique(groups)
        pred_sets = torch.zeros_like(x, dtype=torch.bool)
        for i, g in enumerate(unique_groups):
            pred_sets[groups == g] = x[groups == g] <= threshold[i]

        return pred_sets


# TODO Create a Quantile Transform and Remove Compute Quantile From all Score Modules
RAPS_K = "k_reg"
RAPS_LAMBDA = "lambda"


class RegularizationTransformation(Transformation):
    def __init__(self, config: RegularizedConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def find_params(
        self, probs, labels, score_module: APSScore, datamodule: DataModule
    ):
        gen = torch.Generator()
        gen.manual_seed(datamodule.seed)

        calib_tune_nodes = datamodule.split_dict[Stage.CALIBRATION_TUNE]
        N = len(calib_tune_nodes)
        # use only the first half of the calibration set for tuning
        tune_calib_nodes, test_calib_nodes = calib_tune_nodes.split(math.ceil(N / 2))

        eff_str = conf_metric_names.efficiency.name

        overall_results = []

        base_scores = score_module.compute(probs)
        for _ in range(self.config.n_iterations):
            iteration_results = []

            # k_reg is computted just using the ranks
            tune_calib_ranks = torch.argsort(
                torch.argsort(probs[tune_calib_nodes], dim=1, descending=True), dim=1
            )
            label_ranks = (
                tune_calib_ranks.gather(1, labels[tune_calib_nodes].unsqueeze(1))
                .squeeze()
                .float()
            )

            k_reg = score_module.compute_quantile(label_ranks)

            for lambda_fac in [0.001, 0.01, 0.1, 0.2, 0.5]:
                # for quantile computation, we use aps epsilon when raps set adjustment is used
                tune_raps_adjust = self.config.raps_mod

                tune_scores = base_scores[tune_calib_nodes]
                params_kws = {RAPS_K: k_reg, RAPS_LAMBDA: lambda_fac}
                tune_scores = self.transform(
                    tune_scores,
                    probs[tune_calib_nodes],
                    raps_modified=tune_raps_adjust,
                    **params_kws
                )

                qhat = score_module.compute_quantile(
                    tune_scores.gather(
                        1, labels[tune_calib_nodes].unsqueeze(1)
                    ).squeeze()
                )

                # compute tune scores - always uses set adjustment
                test_raps_adjust = True
                test_scores = base_scores[test_calib_nodes]
                test_scores = self.transform(
                    test_scores,
                    probs[test_calib_nodes],
                    raps_modified=test_raps_adjust,
                    **params_kws
                )

                # get result metrics
                prediction_sets = PredSetTransformation(threshold=qhat).pipe_transform(
                    test_scores
                )
                eff = compute_metric(eff_str, prediction_sets, labels[test_calib_nodes])
                iteration_results.append(
                    {RAPS_K: k_reg.item(), RAPS_LAMBDA: lambda_fac, eff_str: eff.item()}
                )

            overall_results.extend(iteration_results)

            if self.config.resplit_every_iteration:
                shuffle_idx = torch.randperm(N, generator=gen)
                tune_calib_nodes, test_calib_nodes = calib_tune_nodes[
                    shuffle_idx
                ].split(math.ceil(N / 2))

        overall_results = pd.DataFrame(overall_results)
        best_k, best_lambda = (
            overall_results.groupby([RAPS_K, RAPS_LAMBDA]).mean()[eff_str].idxmin()
        )
        return {RAPS_K: best_k, RAPS_LAMBDA: best_lambda}

    def transform(self, x, probs, raps_modified=False, **kwargs):
        k_reg = kwargs.get(RAPS_K, 3.0)
        lambda_fac = kwargs.get(RAPS_LAMBDA, 0.1)

        # x are APSScores
        ranks = torch.argsort(torch.argsort(probs, dim=1, descending=True), dim=1)
        penalties = ranks - k_reg
        penalties[penalties < 0] = 0.0
        u = torch.rand(x.shape[0], 1, device=x.device)
        if raps_modified:
            # in this case, the base aps conformal score should not have used aps epsilon
            penalty_ind = (penalties > 0).float()
            return x + lambda_fac * penalties - u * (probs + lambda_fac * penalty_ind)
        else:
            # with the original RAPS implementation, the base aps conformal score should have used aps epsilon
            return x - u * probs + lambda_fac * penalties


class DiffusionTransformation(Transformation):
    def __init__(self, config: DiffusionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def find_params(self, probs, labels, score_module: CPScore, datamodule: DataModule):
        gen = torch.Generator()
        gen.manual_seed(datamodule.seed)

        calib_tune_nodes = datamodule.split_dict[Stage.CALIBRATION_TUNE]
        N = len(calib_tune_nodes)
        # use only the first half of the calibration set for tuning
        tune_calib_nodes, test_calib_nodes = calib_tune_nodes.split(math.ceil(N / 2))

        overall_results: List[pd.Series] = []
        for _ in range(self.config.n_iterations):
            scores = score_module.pipe_compute(probs)
            iteration_results = []
            for diff_param in torch.arange(0, 1, 0.05).round(decimals=2):
                scores = self.transform(
                    scores, datamodule=datamodule, diffusion_param=diff_param
                )
                label_scores = get_label_scores(
                    labels, scores, tune_calib_nodes, self.defined_args.get("dataset")
                )
                # additional kwargs for tps
                if isinstance(score_module, TPSScore):
                    kwargs = {
                        "labels": labels[tune_calib_nodes],
                        "num_classes": datamodule.num_classes,
                    }
                else:
                    kwargs = {}

                qhat = score_module.compute_quantile(label_scores, **kwargs)
                eff_str = conf_metric_names.efficiency.name

                test_calib_scores = scores[test_calib_nodes]
                prediction_sets = PredSetTransformation(threshold=qhat).pipe_transform(
                    test_calib_scores
                )

                eff = compute_metric(eff_str, prediction_sets, labels[test_calib_nodes])

                assert isinstance(eff, torch.Tensor)

                iteration_results.append(
                    {"diffusion_param": diff_param.item(), eff_str: eff.item()}
                )

            iteration_results = pd.DataFrame(iteration_results)
            baseline_res = iteration_results.loc[
                (iteration_results["diffusion_param"] == 0)
            ][eff_str].values[0]
            iteration_results["improvement"] = iteration_results[eff_str] - baseline_res
            overall_results.append(iteration_results[["improvement"]])

            if self.config.resplit_every_iteration:
                shuffle_idx = torch.randperm(N, generator=gen)
                tune_calib_nodes, test_calib_nodes = calib_tune_nodes[
                    shuffle_idx
                ].split(math.ceil(N / 2))

        overall_results = pd.concat(overall_results, axis=1)
        overall_mean_impr = overall_results.mean(axis=1)

        best_param_sets = iteration_results.loc[overall_mean_impr.idxmin()]
        best_params = {"diffusion_param": best_param_sets["diffusion_param"]}

        return best_params

    def transform(self, x, **kwargs):
        diffusion_param = kwargs.get("diffusion_param", 0)
        datamodule = kwargs.get("datamodule")
        x = x.to(DEFAULT_DEVICE)
        A = datamodule.adj_matrix.to(DEFAULT_DEVICE)

        # TODO: More efficient, possibly batched computation
        degs = torch.matmul(A, torch.ones((A.shape[0])).to(DEFAULT_DEVICE))

        return (
            (1 - diffusion_param) * x
            + diffusion_param
            * (1 / (degs + 1e-10))[:, None]
            * torch.linalg.matmul(A, x)
        ).cpu()
