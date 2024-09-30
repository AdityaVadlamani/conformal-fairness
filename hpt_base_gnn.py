import logging
from contextlib import ExitStack
from itertools import product
from typing import Any, Dict

import conformal_fairness.utils as utils
import numpy as np
import pyrallis.argparsing as pyr_a
import ray.train
from conformal_fairness.config import BaseExptConfig, DatasetSplitConfig
from conformal_fairness.constants import layer_types, sample_type
from conformal_fairness.custom_logger import CustomLogger
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

from hpt_config import BaseTuneExptConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_PREFIX = "base_gnn."
CUSTOM_STEP = "custom_step"
RESULTS_TABLE = "basegnn_tune_table"
TRIAL_PREFIX = "TRIAL_"
DATASET = "dataset"


def get_job_type(config: BaseExptConfig) -> str:
    if (
        config.logging_config is not None
        and config.logging_config.wandb_config is not None
    ):
        return config.logging_config.wandb_config.job_type
    return "tune"


def create_tune_jobid_from_config(config: BaseExptConfig) -> str:
    """Create a job name from the config for easy w&b grouping.
    Since we launch n_trials_per_config for each config, we will use the fixed part of the config to generate a job name.
    """
    loading_style = config.dataset_loading_style
    match loading_style:
        case sample_type.split.name:
            split_fractions = config.dataset_split_fractions
            return f"{config.dataset.name}_{config.base_gnn.model}_{loading_style}_{split_fractions.train}_{split_fractions.valid}_{'_'.join(config.dataset.sens_attrs)}"  # type: ignore
        case sample_type.n_samples_per_class.name:
            return f"{config.dataset.name}_{config.base_gnn.model}_{loading_style}_{config.dataset_n_samples_per_class}_{'_'.join(config.dataset.sens_attrs)}"  # type: ignore
        case _:
            raise ValueError("Unsupported loading style")


def get_aggr_func(aggr: str):
    if hasattr(np, aggr):
        return getattr(np, aggr)
    else:
        raise ValueError(f"Invalid aggregation function {aggr} not in numpy")


def get_aggr_metric_name(aggr: str, metric: str):
    return f"{aggr}_{metric}"


def update_params(base_config: BaseExptConfig, new_config: Dict[str, Any]):
    base_params = {}
    expt_params = {}
    for k, v in new_config.items():
        if BASE_PREFIX in k:
            rem_k = k[len(BASE_PREFIX) :]
            base_params[rem_k] = v
        else:
            expt_params[k] = v

    utils.update_dataclass_from_dict(base_config, expt_params)
    utils.update_dataclass_from_dict(base_config.base_gnn, base_params)


def set_run_name(config: BaseExptConfig, trial_name: str):
    if (
        config.logging_config is not None
        and config.logging_config.wandb_config is not None
    ):
        config.logging_config.wandb_config.run_name = trial_name


def train_func(base_tune_config: BaseTuneExptConfig, new_config: Dict[str, Any]):
    base_config = base_tune_config.expt_config
    update_params(base_config, new_config)

    metric_vals = []

    for idx in range(base_tune_config.n_trials_per_config):
        utils.set_seed_and_precision(idx)
        base_config.seed = idx
        base_config.job_id = create_tune_jobid_from_config(base_config)
        set_run_name(base_config, ray.train.get_context().get_trial_name())
        expt_logger = CustomLogger(base_config.logging_config)

        datamodule = utils.prepare_datamodule(base_config)
        datamodule.setup_sampler(base_config.base_gnn.layers)

        model = utils.setup_base_model(base_config, datamodule)

        trainer = utils.setup_trainer(
            base_config,
            strategy=RayDDPStrategy(),
            plugins=[RayLightningEnvironment()],
            num_sanity_val_steps=0,
        )
        trainer = prepare_trainer(trainer)

        with ExitStack() as stack:
            train_dl, val_dl = utils.enter_cpu_cxs(
                datamodule, ["train_dataloader", "val_dataloader"], stack
            )
            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
            )

        metric_name = base_tune_config.metric_used
        assert (
            metric_name in trainer.logged_metrics
        ), "Metric not found in trainer.logged_metrics"
        metric_val = trainer.logged_metrics.get(metric_name)

        expt_logger.log_hyperparams(vars(base_config))
        expt_logger.log_metrics({metric_name: metric_val, CUSTOM_STEP: 0})
        expt_logger.force_exit()

        metric_vals.append(metric_val)

    aggr_metric = get_aggr_metric_name(
        base_tune_config.metric_aggr, base_tune_config.metric_used
    )
    aggr_metric_val = get_aggr_func(base_tune_config.metric_aggr)(metric_vals)
    ray.train.report({aggr_metric: aggr_metric_val, DATASET: base_config.dataset.name})


def main():
    args = pyr_a.parse(config_class=BaseTuneExptConfig)
    aggr_metric_name = get_aggr_metric_name(args.metric_aggr, args.metric_used)

    print(f"EXPT ARGS: {args.expt_config}")

    t_config = args.tune_split_config
    expt_loop_space = []
    # ensure dataset download before launching
    utils.prepare_datamodule(args.expt_config)

    match t_config.s_type:
        case sample_type.split.name:
            expt_loop_space = list(
                product(args.l_types, t_config.train_fracs, t_config.val_fracs)
            )
        case sample_type.n_samples_per_class.name:
            expt_loop_space = list(product(args.l_types, t_config.samples_per_class))

    # we will intialize the config partially and pass into the tune function
    # all experiments run in this script are generated from this
    # by deafult, we will have the default values
    expt_config = args.expt_config
    expt_config.resume_from_checkpoint = False

    for split_config in expt_loop_space:
        l_type = split_config[0]

        expt_config.base_gnn.model = l_type
        expt_config.dataset_loading_style = t_config.s_type

        match t_config.s_type:
            case sample_type.split.name:
                assert len(split_config) == 3
                expt_config.dataset_split_fractions = DatasetSplitConfig()
                expt_config.dataset_split_fractions.train = split_config[1]
                expt_config.dataset_split_fractions.valid = split_config[2]

            case sample_type.n_samples_per_class.name:
                assert len(split_config) == 2
                expt_config.dataset_n_samples_per_class = split_config[1]

        search_space = {
            f"{BASE_PREFIX}lr": tune.loguniform(1e-4, 1e-1),
            f"{BASE_PREFIX}hidden_channels": tune.choice([16, 32, 64, 128]),
            f"{BASE_PREFIX}layers": tune.choice([1, 2, 4]),
            f"{BASE_PREFIX}dropout": tune.uniform(0.1, 0.8),
        }

        match l_type:
            case layer_types.GAT.name:
                search_space[f"{BASE_PREFIX}heads"] = tune.choice([2, 4, 8])
            case layer_types.GraphSAGE.name:
                search_space[f"{BASE_PREFIX}aggr"] = tune.choice(
                    ["mean", "gcn", "pool", "lstm"]
                )

        scheduler = ASHAScheduler(
            max_t=expt_config.epochs, grace_period=1, reduction_factor=2
        )

        scaling_config = ScalingConfig(
            num_workers=args.n_tune_workers,
            use_gpu=expt_config.resource_config.gpus > 0,
            resources_per_worker={
                "CPU": expt_config.resource_config.cpus,
                "GPU": expt_config.resource_config.gpus,
            }
            | expt_config.resource_config.custom,
        )
        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(num_to_keep=1),
            storage_path=args.tune_output_dir,
        )

        ray_trainer = TorchTrainer(
            lambda new_config: train_func(args, new_config),
            scaling_config=scaling_config,
            run_config=run_config,
        )

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric=aggr_metric_name,
                mode="max",
                num_samples=args.num_samples,
                scheduler=scheduler,
            ),
        )
        res = tuner.fit()

        # log the best run
        base_config = args.expt_config
        base_config.job_id = create_tune_jobid_from_config(base_config)
        expt_logger = CustomLogger(args.expt_config.logging_config)
        # expt_logger.log_hyperparams(vars(base_config))
        best_result_val = 0
        try:
            best_result = res.get_best_result()
            job_type = get_job_type(expt_config)
            best_result_val = best_result.metrics.get(aggr_metric_name, 0)  # type: ignore
        except RuntimeError:
            logger.warn("No best result found for ")

        expt_logger.log_hyperparams(vars(base_config))
        expt_logger.log_table(
            title=RESULTS_TABLE,
            data=[
                [
                    split_config,
                    args.expt_config.dataset.name,
                    f"{job_type}_result",
                    "base",
                    best_result_val,
                ]
                + list(best_result.config.values())
            ],
            columns=[
                "split_config",
                "dataset",
                "job_type",
                "group",
                aggr_metric_name,
            ]
            + [f"best_config.{key}" for key in best_result.config.keys()],
        )
        expt_logger.force_exit()


if __name__ == "__main__":
    # python hpt_base_gnn.py  --config_path="configs/hpt_base_gnn_default.yaml"
    main()
