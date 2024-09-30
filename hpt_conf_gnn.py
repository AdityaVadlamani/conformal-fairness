import logging
import os
from itertools import product
from typing import Any, Dict

import conformal_fairness.utils as utils
import numpy as np
import pyrallis.argparsing as pyr_a
import ray.train
from conformal_fairness.config import ConfExptConfig, DatasetSplitConfig
from conformal_fairness.constants import (
    CREDIT,
    ConformalMethod,
    Stage,
    layer_types,
    sample_type,
)
from conformal_fairness.custom_logger import CustomLogger
from conformal_fairness.models import CFGNN
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

from hpt_config import ConfGNNTuneExptConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CFGNN_PREFIX = "cf_gnn."
CUSTOM_STEP = "custom_step"
RESULTS_TABLE = "cfgnn_tune_table"
TRIAL_PREFIX = "TRIAL_"
DATASET = "dataset"


def get_job_type(config: ConfExptConfig) -> str:
    if (
        config.logging_config is not None
        and config.logging_config.wandb_config is not None
    ):
        return config.logging_config.wandb_config.job_type
    return "tune"


def create_tune_jobid_from_config(config: ConfExptConfig) -> str:
    """Create a job name from the config for easy w&b grouping.
    Since we launch n_trials_per_config for each config, we will use the fixed part of the config to generate a job name.
    """
    loading_style = config.dataset_loading_style
    match loading_style:
        case sample_type.split.name:
            split_fractions = config.dataset_split_fractions
            return f"{config.dataset.name}_{config.confgnn_config.model}_{loading_style}_{split_fractions.train}_{split_fractions.valid}_{'_'.join(config.dataset.sens_attrs)}"  # type: ignore
        case sample_type.n_samples_per_class.name:
            return f"{config.dataset.name}_{config.confgnn_config.model}_{loading_style}_{config.dataset_n_samples_per_class}_{'_'.join(config.dataset.sens_attrs)}"  # type: ignore
        case _:
            raise ValueError("Unsupported loading style")


def get_aggr_func(aggr: str):
    if hasattr(np, aggr):
        return getattr(np, aggr)
    else:
        raise ValueError(f"Invalid aggregation function {aggr} not in numpy")


def get_aggr_metric_name(aggr: str, metric: str):
    return f"{aggr}_{metric}"


def update_params(conf_config: ConfExptConfig, new_config: Dict[str, Any]):
    base_params = {}
    expt_params = {}
    for k, v in new_config.items():
        if CFGNN_PREFIX in k:
            rem_k = k[len(CFGNN_PREFIX) :]
            base_params[rem_k] = v
        else:
            expt_params[k] = v

    utils.update_dataclass_from_dict(conf_config, expt_params)
    utils.update_dataclass_from_dict(conf_config.confgnn_config, base_params)


def set_run_name(config: ConfExptConfig, trial_name: str):
    if (
        config.logging_config is not None
        and config.logging_config.wandb_config is not None
    ):
        config.logging_config.wandb_config.run_name = trial_name


def train_func(cfgnn_tune_config: ConfGNNTuneExptConfig, new_config: Dict[str, Any]):
    conf_config = cfgnn_tune_config.conf_expt_config
    update_params(conf_config, new_config)

    metric_vals = []

    for idx in range(cfgnn_tune_config.n_trials_per_config):
        # make sure that the sampling config for the expt is same as that of the base job
        base_ckpt_dir = cfgnn_tune_config.base_model_dir
        base_expt_config = utils.load_basegnn_config_from_ckpt(base_ckpt_dir)

        cfgnn_tune_config.conf_expt_config.base_job_id = base_expt_config.job_id

        utils.check_sampling_consistent(base_expt_config, conf_config)

        # update conf_config
        conf_config.conformal_seed = idx
        conf_config.job_id = create_tune_jobid_from_config(conf_config)
        set_run_name(conf_config, ray.train.get_context().get_trial_name())

        # setup dataloaders
        utils.set_seed_and_precision(conf_config.seed)
        datamodule = utils.prepare_datamodule(conf_config)

        # reshulle the calibration and test sets if required
        datamodule.resplit_calib_test(conf_config)

        expt_logger = CustomLogger(conf_config.logging_config)

        datamodule = utils.prepare_datamodule(conf_config)

        # Load probs from base
        probs, _ = utils.load_basegnn_outputs(conf_config, base_ckpt_dir)
        assert (
            probs.shape[1] == datamodule.num_classes
        ), f"Loaded probs has {probs.shape[1]} classes, but the dataset has {datamodule.num_classes} classes"

        assert (
            conf_config.confgnn_config is not None
        ), f"confgnn_config cannot be None for CFGNN"

        if datamodule.name != CREDIT:
            _ = utils.set_trained_basegnn_path(
                conf_config, cfgnn_tune_config.base_model_dir
            )
        _, _ = utils.set_conf_ckpt_dir_fname(conf_config, ConformalMethod.CFGNN.value)

        datamodule.split_calib_tune_qscore(
            tune_frac=conf_config.confgnn_config.tuning_fraction
        )
        if conf_config.confgnn_config.load_probs:
            datamodule.update_features(probs)
            datamodule.setup_sampler(conf_config.confgnn_config.layers)
        else:
            datamodule.setup_sampler(
                conf_config.confgnn_config.layers + base_expt_config.base_gnn.layers
            )

        model = CFGNN(
            config=conf_config.confgnn_config,
            alpha=conf_config.alpha,
            num_epochs=conf_config.epochs,
            num_classes=datamodule.num_classes,
        )

        trainer = utils.setup_trainer(
            conf_config,
            strategy=RayDDPStrategy(find_unused_parameters=True),
            plugins=[RayLightningEnvironment()],
            num_sanity_val_steps=0,
        )
        trainer = prepare_trainer(trainer)

        calib_tune_nodes = datamodule.split_dict[Stage.CALIBRATION_TUNE]

        calib_tune_dl = datamodule.custom_nodes_dataloader(
            calib_tune_nodes,
            conf_config.batch_size,
            shuffle=True,
            drop_last=len(calib_tune_nodes) % conf_config.batch_size,
        )

        with utils.dl_affinity_setup(calib_tune_dl)():
            trainer.fit(
                model,
                train_dataloaders=calib_tune_dl,
                val_dataloaders=calib_tune_dl,
                ckpt_path=None,
            )

        metric_name = cfgnn_tune_config.metric_used
        assert (
            metric_name in trainer.logged_metrics
        ), f"Metric={metric_name} not found in trainer.logged_metrics={trainer.logged_metrics}"
        metric_val = trainer.logged_metrics.get(metric_name)

        expt_logger.log_hyperparams(vars(conf_config))
        expt_logger.log_metrics({metric_name: metric_val, CUSTOM_STEP: 0})
        expt_logger.force_exit()

        metric_vals.append(metric_val)

    aggr_metric = get_aggr_metric_name(
        cfgnn_tune_config.metric_aggr, cfgnn_tune_config.metric_used
    )
    aggr_metric_val = get_aggr_func(cfgnn_tune_config.metric_aggr)(metric_vals)
    ray.train.report({aggr_metric: aggr_metric_val, DATASET: conf_config.dataset})

    return {aggr_metric: aggr_metric_val, DATASET: conf_config.dataset}


def main():
    args: ConfGNNTuneExptConfig = pyr_a.parse(config_class=ConfGNNTuneExptConfig)
    aggr_metric_name = get_aggr_metric_name(args.metric_aggr, args.metric_used)

    if (
        args.calib_test_equal
        and args.conf_expt_config.dataset_loading_style == sample_type.split.name
    ):
        args.conf_expt_config.dataset_split_fractions.calib = (
            1
            - args.conf_expt_config.dataset_split_fractions.train
            - args.conf_expt_config.dataset_split_fractions.valid
        ) / 2

    t_config = args.tune_split_config
    expt_loop_space = []
    # ensure dataset download before launching
    utils.prepare_datamodule(args.conf_expt_config)

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
    expt_config = args.conf_expt_config
    expt_config.resume_from_checkpoint = False

    for split_config in expt_loop_space:
        l_type = split_config[0]

        expt_config.confgnn_config.model = l_type
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
            f"{CFGNN_PREFIX}lr": tune.loguniform(1e-4, 1e-1),
            f"{CFGNN_PREFIX}hidden_channels": tune.choice([16, 32, 64, 128]),
            f"{CFGNN_PREFIX}layers": tune.choice([1, 2, 3, 4]),
            f"{CFGNN_PREFIX}dropout": tune.uniform(0.1, 0.8),
            f"{CFGNN_PREFIX}temperature": tune.loguniform(1e-3, 1e1),
        }

        match l_type:
            case layer_types.GAT.name:
                search_space[f"{CFGNN_PREFIX}heads"] = tune.choice([2, 4])
            case layer_types.GraphSAGE.name:
                search_space[f"{CFGNN_PREFIX}aggr"] = tune.choice(
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
            placement_strategy="SPREAD",
        )

        if expt_config.dataset_loading_style == sample_type.split.name:
            name = f"{expt_config.dataset}/{expt_config.dataset_loading_style}/{expt_config.dataset_split_fractions.train}_{expt_config.dataset_split_fractions.valid}_{expt_config.confgnn_config.model}"
        else:
            name = f"{expt_config.dataset}/{expt_config.dataset_loading_style}/{expt_config.dataset_n_samples_per_class}_{expt_config.confgnn_config.model}"

        run_config = RunConfig(
            name=name,
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
                mode=args.metric_mode,
                num_samples=args.num_samples,
                scheduler=scheduler,
                reuse_actors=True,
            ),
        )
        res = tuner.fit()

        # log the best run
        expt_config.job_id = create_tune_jobid_from_config(expt_config)
        expt_logger = CustomLogger(args.conf_expt_config.logging_config)
        # expt_logger.log_hyperparams(vars(base_config))
        best_result_val = 0
        try:
            best_result = res.get_best_result(
                metric=aggr_metric_name, mode=args.metric_mode
            )
            job_type = get_job_type(expt_config)
            best_result_val = best_result.metrics.get(aggr_metric_name, 0)  # type: ignore
        except RuntimeError:
            logger.warning("No best result found for ")

        expt_logger.log_hyperparams(vars(expt_config))
        expt_logger.log_table(
            title=RESULTS_TABLE,
            data=[
                [
                    split_config,
                    f"{expt_config.dataset.name}_{'_'.join(expt_config.dataset.sens_attrs)}",
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
    # python hpt_conf_gnn.py  --config_path="configs/hpt_conf_gnn_default.yaml"
    main()
