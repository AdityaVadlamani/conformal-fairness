import glob
import logging
import os
from contextlib import ExitStack
from typing import Dict, Optional, Tuple

import lightning.pytorch as L
import pyrallis.cfgparsing as pyr_c
import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from xgboost import XGBClassifier

from ..config import BaseExptConfig, ConfExptConfig, SharedBaseConfig
from ..constants import *
from ..custom_logger import CustomLogger
from ..data import BaseDataModule, GraphDataModule, TabularDataModule
from ..models import GNN, MLP, BaseModel
from .sys_utils import enter_cpu_cxs

logging.basicConfig(level=logging.INFO)


# Helper functions for base gnn
def set_seed_and_precision(seed: int):
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision(PYTORCH_PRECISION)


def _get_output_directory(output_directory, dataset, job_id) -> str:
    return os.path.join(output_directory, dataset, job_id)


def prepare_datamodule(args: SharedBaseConfig) -> BaseDataModule:
    rank_zero_info("Setting up data module")

    # TODO: Generalize this first condition to multiple dataset
    if args.dataset.name in TABULAR_DATASETS and args.dataset.name in GRAPH_DATASETS:
        if args.dataset.type == DatasetType.TABULAR:
            datamodule = TabularDataModule(config=args)
        elif args.dataset.type == DatasetType.GRAPH:
            datamodule = GraphDataModule(config=args)

    elif args.dataset.name in TABULAR_DATASETS:
        datamodule = TabularDataModule(config=args)
    elif args.dataset.name in GRAPH_DATASETS:
        datamodule = GraphDataModule(config=args)
    else:
        raise NotImplementedError()

    datamodule.prepare_data()
    assert (
        args.dataset_split_fractions is not None
    ), f"Dataset split fractions must be provided"

    datamodule.setup(args)

    rank_zero_info("Finished setting up data module")
    return datamodule


def setup_base_model(args: BaseExptConfig, datamodule: BaseDataModule) -> BaseModel:
    rank_zero_info("Setting up lightning module")
    model = None
    if args.dataset.name in TABULAR_DATASETS:
        model = MLP(
            config=args.base_model_config,
            num_features=datamodule.num_features,
            num_classes=datamodule.num_classes,
        )
    else:
        model = GNN(
            config=args.base_model_config,
            num_features=datamodule.num_features,
            num_classes=datamodule.num_classes,
        )
    rank_zero_info("Finished setting up lightning module")
    return model


def _get_ckpt_dir_fname(output_dir, dataset, job_id, ckpt_prefix) -> Tuple[str, str]:
    ckpt_dir = os.path.join(output_dir, dataset, job_id)
    ckpt_filename = f"{ckpt_prefix}_{{val_acc:.4f}}"
    return ckpt_dir, ckpt_filename


def get_base_ckpt_dir_fname(output_dir, dataset, job_id) -> Tuple[str, str]:
    return _get_ckpt_dir_fname(output_dir, dataset, job_id, BASE_MODEL_CKPT_PREFIX)


def set_conf_ckpt_dir_fname(
    args: ConfExptConfig, conformal_method_str
) -> Tuple[str, str]:
    if not args.confgnn_config.ckpt_dir or args.confgnn_config.ckpt_filename:
        args.confgnn_config.ckpt_dir, args.confgnn_config.ckpt_filename = (
            _get_ckpt_dir_fname(
                args.output_dir,
                args.dataset.name,
                args.job_id,
                conformal_method_str,
            )
        )
    return args.confgnn_config.ckpt_dir, args.confgnn_config.ckpt_filename


def output_base_model_config(output_dir: str, args: BaseExptConfig):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, BASE_MODEL_CKPT_CONFIG_FILE), "w") as f:
        pyr_c.dump(args, f)


def load_base_config_from_ckpt(
    ckpt_dir: str, default_args: Optional[BaseExptConfig] = None
) -> BaseExptConfig:
    """Default args used if yaml not found"""
    yaml_path = os.path.join(ckpt_dir, BASE_MODEL_CKPT_CONFIG_FILE)
    logging.info(f"Attempting base model config load from {yaml_path}")
    if os.path.exists(yaml_path):
        if default_args is not None:
            logging.warning(
                f"Config will be overwritten by existing {default_args.dataset.name}/{default_args.job_id}"
            )
        # load BaseExpt config from yaml
        with open(yaml_path, "r") as f:
            return pyr_c.load(BaseExptConfig, f)
    else:
        assert (
            default_args is not None
        ), "No default args provided and no config file found"
        return default_args


def _base_ckpt_path(job_output_dir: str):
    return glob.glob(os.path.join(job_output_dir, f"{BASE_MODEL_CKPT_PREFIX}*.ckpt"))


def set_trained_base_model_path(args: ConfExptConfig, ckpt_dir: Optional[str] = None):
    if ckpt_dir is not None:
        args.confgnn_config.base_model_path = _base_ckpt_path(ckpt_dir)[0]
    elif not args.confgnn_config.base_model_path:
        job_output_dir = _get_output_directory(
            args.output_dir, args.dataset.name, args.base_job_id
        )
        # TODO: We assume that the first checkpoint found in the job id will be the one to use
        args.confgnn_config.base_model_path = _base_ckpt_path(job_output_dir)[0]
    return args.confgnn_config.base_model_path


def load_base_model(ckpt_dir: str, args: BaseExptConfig, datamodule) -> BaseModel:
    base_ckpt_path = _base_ckpt_path(ckpt_dir)
    if args.resume_from_checkpoint:
        if (
            args.dataset.name not in TABULAR_DATASETS
        ):  # isinstance(args.base_model_config,BaseGNNConfig):
            if len(base_ckpt_path) > 0:
                base_ckpt_path = base_ckpt_path[0]
                logging.info(f"Resuming from checkpoint: {base_ckpt_path}")
                model = MLP.load_from_checkpoint(base_ckpt_path)
                return model
            else:
                logging.warning(
                    "No checkpoint found for resuming. Training from scratch."
                )
        else:
            if len(base_ckpt_path) > 0:
                base_ckpt_path = base_ckpt_path[0]
                logging.info(f"Resuming from checkpoint: {base_ckpt_path}")
                model = GNN.load_from_checkpoint(base_ckpt_path)
                return model
            else:
                logging.warning(
                    "No checkpoint found for resuming. Training from scratch."
                )
    return setup_base_model(args, datamodule)


def load_basexgb(args: BaseExptConfig) -> XGBClassifier:
    # base_ckpt_path = _base_ckpt_path(ckpt_dir)
    n_estimators = args.base_model_config.n_estimators
    max_depth = args.base_model_config.max_depth
    max_leaves = args.base_model_config.max_leaves
    lr = args.base_model_config.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    objective = "multi:softprob"
    verbose = 2

    grow_policy = args.base_model_config.grow_policy
    booster = args.base_model_config.booster
    gamma = args.base_model_config.gamma
    colsample_bytree = args.base_model_config.colsample_bytree
    colsample_bylevel = args.base_model_config.colsample_bylevel
    colsample_bynode = args.base_model_config.colsample_bynode
    subsample = args.base_model_config.subsample
    seed = args.seed

    xgbmodel = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_leaves=max_leaves,
        device=device,
        objective=objective,
        verbose=verbose,
        learning_rate=lr,
        grow_policy=grow_policy,
        booster=booster,
        gamma=gamma,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        colsample_bynode=colsample_bynode,
        subsample=subsample,
        random_state=seed,
    )
    # eval_metric=accuracy_score)
    return xgbmodel


def train_basexgb(model, datamodule) -> XGBClassifier:
    train_labels = (
        datamodule.y[datamodule.split_dict[Stage.TRAIN]].cpu().detach().numpy()
    )
    train_features = (
        datamodule.X[datamodule.split_dict[Stage.TRAIN], :].cpu().detach().numpy()
    )
    valid_labels = (
        datamodule.y[datamodule.split_dict[Stage.VALIDATION]].cpu().detach().numpy()
    )
    valid_features = (
        datamodule.X[datamodule.split_dict[Stage.VALIDATION], :].cpu().detach().numpy()
    )
    return model.fit(
        train_features, train_labels, eval_set=[(valid_features, valid_labels)]
    )


def setup_trainer(
    args: SharedBaseConfig,
    expt_logger: Optional[CustomLogger] = None,
    /,
    strategy="auto",
    callbacks=None,
    plugins=None,
    **kwargs,
) -> L.Trainer:
    rank_zero_info("Setting up trainer")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.resource_config.gpus,
        num_nodes=args.resource_config.nodes,
        max_epochs=args.epochs,
        strategy=strategy,
        callbacks=callbacks,
        plugins=plugins,
        logger=expt_logger,
        log_every_n_steps=100,
        check_val_every_n_epoch=1,
        **kwargs,
    )
    rank_zero_info("Finished setting up trainer")
    return trainer


def output_base_model_results(args: BaseExptConfig, results: Dict[str, torch.Tensor]):
    assert NODE_IDS_KEY in results
    # assert that results[NODE_IDS_KEY] is sorted
    assert torch.all(
        results[NODE_IDS_KEY].argsort() == torch.arange(len(results[NODE_IDS_KEY]))
    )
    assert LABELS_KEY in results and PROBS_KEY in results

    job_output_dir = _get_output_directory(
        args.output_dir, args.dataset.name, args.job_id
    )
    os.makedirs(job_output_dir, exist_ok=True)
    torch.save(results, os.path.join(job_output_dir, ALL_OUTPUTS_FILE))


def run_inference_alldl(model: GNN, trainer: L.Trainer, ckpt_path, datamodule):
    with ExitStack() as stack:
        dl = enter_cpu_cxs(datamodule, ["all_dataloader"], stack)
        trainer.test(model, dataloaders=dl, ckpt_path=ckpt_path, verbose=False)

        # return model.latest_test_results
        num_points = datamodule.num_points
        num_classes = datamodule.num_classes

        updated_test_results = {
            NODE_IDS_KEY: torch.arange(num_points),
            LABELS_KEY: torch.fill(torch.empty(num_points, dtype=torch.long), -1),
            PROBS_KEY: torch.fill(torch.empty(num_points, num_classes), -1),
        }

        updated_test_results[LABELS_KEY][model.latest_test_results[NODE_IDS_KEY]] = (
            model.latest_test_results[LABELS_KEY]
        )
        updated_test_results[PROBS_KEY][model.latest_test_results[NODE_IDS_KEY]] = (
            model.latest_test_results[PROBS_KEY]
        )

        return updated_test_results


def run_xgb_inference_alldl(model: XGBClassifier, datamodule):
    # return model.latest_test_results
    num_points = datamodule.num_points
    # num_classes = datamodule.num_classes
    probs = model.predict_proba(datamodule.X.detach().cpu().numpy())

    updated_test_results = {
        NODE_IDS_KEY: torch.arange(num_points),
        LABELS_KEY: datamodule.y,
        PROBS_KEY: torch.tensor(probs),
    }
    return updated_test_results


def basexgb_valid_outputs(model: XGBClassifier, datamodule):
    valid_mask = datamodule.split_dict[Stage.VALIDATION]
    # return model.latest_test_results
    num_points = len(valid_mask)
    # num_classes = datamodule.num_classes
    probs = model.predict_proba(datamodule.X[valid_mask, :].detach().cpu().numpy())

    updated_test_results = {
        NODE_IDS_KEY: torch.arange(num_points),
        LABELS_KEY: datamodule.y[valid_mask],
        PROBS_KEY: torch.tensor(probs),
    }

    pred_label = torch.argmax(updated_test_results[PROBS_KEY], dim=1)
    acc = (pred_label == updated_test_results[LABELS_KEY]).sum() / num_points
    return updated_test_results, acc


def check_sampling_consistent(
    base_expt_config: BaseExptConfig, expt_config: ConfExptConfig
):
    assert (
        base_expt_config.dataset.name == expt_config.dataset.name
    ), "Dataset must be consistent"
    assert base_expt_config.seed == expt_config.seed, "Seed must be consistent"
    assert (
        base_expt_config.dataset_split_fractions.train
        == expt_config.dataset_split_fractions.train
        and base_expt_config.dataset_split_fractions.valid
        == expt_config.dataset_split_fractions.valid
    ), "Dataset split fractions must be consistent"


def load_base_model_outputs(args: ConfExptConfig, job_output_dir: Optional[str] = None):
    if not job_output_dir:
        job_output_dir = _get_output_directory(
            args.output_dir, args.dataset.name, args.base_job_id
        )
    results = torch.load(os.path.join(job_output_dir, ALL_OUTPUTS_FILE))
    probs, labels = results[PROBS_KEY], results[LABELS_KEY]
    assert isinstance(probs, torch.Tensor) and isinstance(labels, torch.Tensor)
    return probs, labels


def update_dataclass_from_dict(dataclass, update_dict):
    for k, v in update_dict.items():
        if hasattr(dataclass, k):
            setattr(dataclass, k, v)
        else:
            logging.warning(
                f"Attempted to update {k} in {type(dataclass)} but it does not exist."
            )
