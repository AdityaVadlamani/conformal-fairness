import logging
import os
import shutil

from conformal_fairness import utils
import pyrallis.argparsing as pyr_a
from conformal_fairness.config import BaseExptConfig
from conformal_fairness.constants import *
from conformal_fairness.custom_logger import CustomLogger
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_params(base_config: BaseExptConfig, new_config):
    utils.update_dataclass_from_dict(base_config.base_model_config, new_config)


def train_func(config, base_config: BaseExptConfig, num_trials):
    update_params(base_config, config)
    sum_acc = 0
    base_seed = base_config.seed
    force_reprep = base_config.dataset.force_reprep
    base_config.dataset.force_reprep = False
    for i in range(num_trials):
        base_config.seed = i
        utils.set_seed_and_precision(i)
        datamodule = utils.prepare_datamodule(base_config)
        model: XGBClassifier = utils.load_basexgb(base_config)  # get model
        model = utils.train_basexgb(model, datamodule)  # fit the model
        _, valid_acc = utils.basexgb_valid_outputs(model, datamodule)
        sum_acc += valid_acc.item()
    # results = utils.run_xgb_inference_alldl(model, datamodule)  # get results
    base_config.seed = base_seed
    base_config.dataset.force_reprep = force_reprep
    train.report({"accuracy": sum_acc / num_trials})


def train_func_list(config, base_config: BaseExptConfig, num_trials, datamodule_list):
    update_params(base_config, config)
    sum_acc = 0
    base_seed = base_config.seed
    base_config.dataset.force_reprep = False
    for i in range(num_trials):
        base_config.seed = i
        utils.set_seed_and_precision(i)
        datamodule = datamodule_list[i]
        model: XGBClassifier = utils.load_basexgb(base_config)  # get model
        model = utils.train_basexgb(model, datamodule)  # fit the model
        _, valid_acc = utils.basexgb_valid_outputs(model, datamodule)
        sum_acc += valid_acc.item()
    # results = utils.run_xgb_inference_alldl(model, datamodule)  # get results
    base_config.seed = base_seed
    train.report({"accuracy": sum_acc / num_trials})


def main():
    args = pyr_a.parse(config_class=BaseExptConfig)

    # Callbacks
    # setup checkpointing
    ckpt_dir, ckpt_filename = utils.get_base_ckpt_dir_fname(
        args.output_dir, args.dataset.name, args.job_id
    )

    if not args.resume_from_checkpoint:
        # delete existing chekpoint dir if it exists
        logger.warning(
            f"Existing checkpoint for {args.dataset}/{args.job_id} will be overwritten."
        )
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
    else:
        logger.warning("Resuming from checkpoint")
        args = utils.load_base_config_from_ckpt(ckpt_dir, args)

    utils.prepare_datamodule(args)

    # datamodule.setup_sampler(args.base_model_config.layers)

    # create logger and log expt hyperparams
    expt_logger = CustomLogger(args.logging_config)
    expt_logger.log_hyperparams(vars(args))

    config = {
        "n_estimators": tune.randint(2, 500),
        "max_depth": tune.randint(2, 30),
        "lr": tune.loguniform(1e-4, 1e-1),
        # 'grow_policy': tune.choice(['depthwise','lossguide']),
        # 'booster': tune.choice(['gbtree', 'dart']),
        "gamma": tune.uniform(0, 1),
        "colsample_bytree": tune.uniform(0.25, 1.0),
        "colsample_bylevel": tune.uniform(0.25, 1.0),
        "colsample_bynode": tune.uniform(0.25, 1.0),
        "subsample": tune.uniform(0.5, 1.0),
    }

    scheduler = ASHAScheduler(max_t=500, grace_period=1, reduction_factor=2)

    tune_config = tune.TuneConfig(
        metric="accuracy",
        mode="max",
        num_samples=25,
        max_concurrent_trials=1,
        scheduler=scheduler,
    )
    # scaling_config = tune.ScalingConfig(num_workers=27,use_gpu=True,
    #                                     resources_per_worker={"CPU": 5, "GPU": 0.2,})
    num_seeds = 5
    datamodule_list = []
    force_reprep = args.dataset.force_reprep
    args.dataset.force_reprep = True
    base_seed = args.seed
    for i in range(num_seeds):
        args.seed = i
        utils.set_seed_and_precision(i)
        datamodule = utils.prepare_datamodule(args)
        datamodule_list.append(datamodule)
        args.dataset.force_reprep = False
    # results = utils.run_xgb_inference_alldl(model, datamodule)  # get results
    args.seed = base_seed
    args.dataset.force_reprep = force_reprep
    utils.set_seed_and_precision(base_seed)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_func_list,
                base_config=args,
                num_trials=num_seeds,
                datamodule_list=datamodule_list,
            ),
            resources={"cpu": 27, "gpu": 1},
        ),
        param_space=config,
        tune_config=tune_config,
    )

    # # Retrieve the test and validation masks for each model, once (instead of doing it in each loop)
    # tuner = tune.Tuner(
    #     tune.with_resources(tune.with_parameters(train_func, base_config=args, num_trials=num_seeds),
    #                         resources = {'cpu':27, 'gpu': 1}),
    #     param_space=config,
    #     tune_config=tune_config,
    #     )
    res = tuner.fit()

    best_result = res.get_best_result()
    best_config = args
    update_params(best_config, best_result.metrics["config"])
    utils.output_base_model_config(ckpt_dir, best_config)


if __name__ == "__main__":
    # python hpt_base_xgb.py  --config_path="configs/base_xgb_default.yaml"
    main()
