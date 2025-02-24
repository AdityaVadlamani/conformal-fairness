import glob
import logging
import os
import sys
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import lightning.pytorch as L
import psutil
import pyrallis.cfgparsing as pyr_c
import torch
import xgboost as xgb
from dgl.dataloading import DataLoader
from lightning_utilities.core.rank_zero import rank_zero_info
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from .conf_metrics import calc_coverage, calc_efficiency
from .config import BaseExptConfig, ConfExptConfig, ConfFairExptConfig, SharedBaseConfig
from .conformal_predictor import (
    ConformalMethod,
    ScoreMultiSplitConformalClassifier,
    ScoreSplitConformalClassifer,
)
from .conformal_predictor_fair import ScoreSplitFairConformalClassifer
from .conformal_risk_control import ScoreSplitConformalRiskClassifier
from .constants import (
    ALL_OUTPUTS_FILE,
    BASEGNN_CKPT_CONFIG_FILE,
    BASEGNN_CKPT_PREFIX,
    CPU_AFF,
    FEATURE_FIELD,
    LABEL_FIELD,
    LABELS_KEY,
    NODE_IDS_KEY,
    NON_GRAPH_DATASETS,
    PROBS_KEY,
    PYTORCH_PRECISION,
    SENS_FIELD,
    Stage,
    fairness_metric,
    sample_type,
)
from .custom_logger import CustomLogger
from .data_module import DataModule
from .models import GNN, MLP

logging.basicConfig(level=logging.INFO)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def dl_affinity_setup(dl: DataLoader, avl_affinities: Union[List[int], None] = None):
    # setup cpu affinity for dgl dataloader
    # TODO: multi node issues
    if avl_affinities is None:
        avl_affinities = psutil.Process().cpu_affinity()
    assert avl_affinities is not None, "No available cpu affinities"

    cx = getattr(dl, CPU_AFF)
    cx_fn = partial(
        cx,
        loader_cores=avl_affinities[: dl.num_workers],
        compute_cores=avl_affinities[dl.num_workers :],
        verbose=False,
    )
    # cx_fn = partial(cx, verbose=False)
    return cx_fn


def enter_cpu_cxs(
    datamodule: L.LightningDataModule, dl_strs: List[str], stack: ExitStack
):
    """Enter cpu contexts on stack and return dataloaders"""
    dls = []
    avl_affinities = psutil.Process().cpu_affinity()
    with suppress_stdout():
        for dl_str in dl_strs:
            dl = getattr(datamodule, dl_str)()
            stack.enter_context(dl_affinity_setup(dl, avl_affinities)())
            dls.append(dl)
    return dls


# Helper functions for base gnn
def set_seed_and_precision(seed: int):
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision(PYTORCH_PRECISION)


def _get_output_directory(output_directory, dataset, job_id) -> str:
    return os.path.join(output_directory, dataset, job_id)


def prepare_datamodule(args: SharedBaseConfig) -> DataModule:
    rank_zero_info("Setting up data module")
    datamodule = DataModule(config=args)

    datamodule.prepare_data()
    if args.dataset_loading_style == sample_type.split.name:
        assert (
            args.dataset_split_fractions is not None
        ), f"Dataset split fractions must be provided for loading `{sample_type.split.name}`"

    datamodule.setup(args)

    rank_zero_info("Finished setting up data module")
    return datamodule


def setup_base_model(args: BaseExptConfig, datamodule: DataModule) -> Union[GNN, MLP]:
    rank_zero_info("Setting up lightning module")
    model = None
    if args.dataset.name in NON_GRAPH_DATASETS:
        model = MLP(
            config=args.base_gnn,
            num_features=datamodule.num_features,
            num_classes=datamodule.num_classes,
        )
    else:
        model = GNN(
            config=args.base_gnn,
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
    return _get_ckpt_dir_fname(output_dir, dataset, job_id, BASEGNN_CKPT_PREFIX)


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


def output_basegnn_config(output_dir: str, args: BaseExptConfig):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, BASEGNN_CKPT_CONFIG_FILE), "w") as f:
        pyr_c.dump(args, f)


def load_basegnn_config_from_ckpt(
    ckpt_dir: str, default_args: Optional[BaseExptConfig] = None
) -> BaseExptConfig:
    """Default args used if yaml not found"""
    yaml_path = os.path.join(ckpt_dir, BASEGNN_CKPT_CONFIG_FILE)
    logging.info(f"Attempting basegnn config load from {yaml_path}")
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
    return glob.glob(os.path.join(job_output_dir, f"{BASEGNN_CKPT_PREFIX}*.ckpt"))


def set_trained_basegnn_path(args: ConfExptConfig, ckpt_dir: Optional[str] = None):
    if ckpt_dir is not None:
        args.confgnn_config.base_model_path = _base_ckpt_path(ckpt_dir)[0]
    elif not args.confgnn_config.base_model_path:
        job_output_dir = _get_output_directory(
            args.output_dir, args.dataset.name, args.base_job_id
        )
        # TODO: We assume that the first checkpoint found in the job id will be the one to use
        args.confgnn_config.base_model_path = _base_ckpt_path(job_output_dir)[0]
    return args.confgnn_config.base_model_path


def load_basegnn(ckpt_dir: str, args: BaseExptConfig, datamodule) -> Union[GNN, MLP]:
    base_ckpt_path = _base_ckpt_path(ckpt_dir)
    if args.resume_from_checkpoint:
        if (
            args.dataset.name not in NON_GRAPH_DATASETS
        ):  # isinstance(args.base_gnn,BaseGNNConfig):
            if len(base_ckpt_path) > 0:
                base_ckpt_path = base_ckpt_path[0]
                logging.info(f"Resuming from checkpoint: {base_ckpt_path}")
                model = MLP.load_from_checkpoint(base_ckpt_path)
                return model
            else:
                logging.warn("No checkpoint found for resuming. Training from scratch.")
        else:
            if len(base_ckpt_path) > 0:
                base_ckpt_path = base_ckpt_path[0]
                logging.info(f"Resuming from checkpoint: {base_ckpt_path}")
                model = GNN.load_from_checkpoint(base_ckpt_path)
                return model
            else:
                logging.warn("No checkpoint found for resuming. Training from scratch.")
    return setup_base_model(args, datamodule)


def load_basexgb(args: BaseExptConfig) -> XGBClassifier:
    # base_ckpt_path = _base_ckpt_path(ckpt_dir)
    n_estimators = args.base_gnn.n_estimators
    max_depth = args.base_gnn.max_depth
    max_leaves = args.base_gnn.max_leaves
    lr = args.base_gnn.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    objective = "multi:softprob"
    verbose = 2

    grow_policy = args.base_gnn.grow_policy
    booster = args.base_gnn.booster
    gamma = args.base_gnn.gamma
    colsample_bytree = args.base_gnn.colsample_bytree
    colsample_bylevel = args.base_gnn.colsample_bylevel
    colsample_bynode = args.base_gnn.colsample_bynode
    subsample = args.base_gnn.subsample
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
        datamodule.graph.ndata[LABEL_FIELD][datamodule.split_dict[Stage.TRAIN]]
        .cpu()
        .detach()
        .numpy()
    )
    train_features = (
        datamodule.graph.ndata[FEATURE_FIELD][datamodule.split_dict[Stage.TRAIN], :]
        .cpu()
        .detach()
        .numpy()
    )
    valid_labels = (
        datamodule.graph.ndata[LABEL_FIELD][datamodule.split_dict[Stage.VALIDATION]]
        .cpu()
        .detach()
        .numpy()
    )
    valid_features = (
        datamodule.graph.ndata[FEATURE_FIELD][
            datamodule.split_dict[Stage.VALIDATION], :
        ]
        .cpu()
        .detach()
        .numpy()
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


def output_basegnn_results(args: BaseExptConfig, results: Dict[str, torch.Tensor]):
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


def run_basegnn_inference_alldl(model: GNN, trainer: L.Trainer, ckpt_path, datamodule):
    with ExitStack() as stack:
        dl = enter_cpu_cxs(datamodule, ["all_dataloader"], stack)
        trainer.test(model, dataloaders=dl, ckpt_path=ckpt_path, verbose=False)

        # return model.latest_test_results
        num_nodes = datamodule.num_nodes
        num_classes = datamodule.num_classes

        updated_test_results = {
            NODE_IDS_KEY: torch.arange(num_nodes),
            LABELS_KEY: torch.fill(torch.empty(num_nodes, dtype=torch.long), -1),
            PROBS_KEY: torch.fill(torch.empty(num_nodes, num_classes), -1),
        }

        updated_test_results[LABELS_KEY][model.latest_test_results[NODE_IDS_KEY]] = (
            model.latest_test_results[LABELS_KEY]
        )
        updated_test_results[PROBS_KEY][model.latest_test_results[NODE_IDS_KEY]] = (
            model.latest_test_results[PROBS_KEY]
        )

        return updated_test_results


def run_basexgb_inference_alldl(model: XGBClassifier, datamodule):
    # return model.latest_test_results
    num_nodes = datamodule.num_nodes
    # num_classes = datamodule.num_classes
    probs = model.predict_proba(
        datamodule.graph.ndata[FEATURE_FIELD].detach().cpu().numpy()
    )

    updated_test_results = {
        NODE_IDS_KEY: torch.arange(num_nodes),
        LABELS_KEY: datamodule.graph.ndata[LABEL_FIELD],
        PROBS_KEY: torch.tensor(probs),
    }
    return updated_test_results


def basexgb_valid_outputs(model: XGBClassifier, datamodule):
    valid_mask = datamodule.split_dict[Stage.VALIDATION]
    # return model.latest_test_results
    num_nodes = len(valid_mask)
    # num_classes = datamodule.num_classes
    probs = model.predict_proba(
        datamodule.graph.ndata[FEATURE_FIELD][valid_mask, :].detach().cpu().numpy()
    )

    updated_test_results = {
        NODE_IDS_KEY: torch.arange(num_nodes),
        LABELS_KEY: datamodule.graph.ndata[LABEL_FIELD][valid_mask],
        PROBS_KEY: torch.tensor(probs),
    }

    pred_label = torch.argmax(updated_test_results[PROBS_KEY], dim=1)
    acc = (pred_label == updated_test_results[LABELS_KEY]).sum() / num_nodes
    return updated_test_results, acc


def check_sampling_consistent(
    base_expt_config: BaseExptConfig, expt_config: ConfExptConfig
):
    assert (
        base_expt_config.dataset.name == expt_config.dataset.name
    ), "Dataset must be consistent"
    assert base_expt_config.seed == expt_config.seed, "Seed must be consistent"
    assert (
        base_expt_config.dataset_loading_style == expt_config.dataset_loading_style
    ), "Dataset loading style must be consistent"
    if base_expt_config.dataset_loading_style == sample_type.split.name:
        assert (
            base_expt_config.dataset_split_fractions.train
            == expt_config.dataset_split_fractions.train
            and base_expt_config.dataset_split_fractions.valid
            == expt_config.dataset_split_fractions.valid
        ), "Dataset split fractions must be consistent"
    elif base_expt_config.dataset_loading_style == sample_type.n_samples_per_class.name:
        assert (
            base_expt_config.dataset_n_samples_per_class
            == expt_config.dataset_n_samples_per_class
        ), "Number of samples per class must be consistent"


def load_basegnn_outputs(args: ConfExptConfig, job_output_dir: Optional[str] = None):
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
            logging.warn(
                f"Attempted to update {k} in {type(dataclass)} but it does not exist."
            )


def run_conformal_risk_control(
    args: ConfFairExptConfig,
    datamodule: DataModule,
    base_ckpt_dir: Optional[str] = None,
):
    # Load probs/labels from base for all (not necessarily labeled) nodes
    probs, labels = load_basegnn_outputs(args, base_ckpt_dir)
    assert (
        probs.shape[1] == datamodule.num_classes
    ), f"Loaded probs has {probs.shape[1]} classes, but the dataset has {datamodule.num_classes} classes"

    test_mask = datamodule.graph.ndata[
        Stage.TEST.mask_dstr
    ]  # Test mask of shape (n_nodes,)

    calib_tune = datamodule.graph.ndata[
        Stage.CALIBRATION_TUNE.mask_dstr
    ]  # Test mask of shape (n_nodes,)

    # Perform Conformal Fairness
    split_conf_input = get_split_conf(args)

    crc = ScoreSplitConformalRiskClassifier(args, datamodule)

    calib_tune_scores = crc._get_scores(
        probs[calib_tune], labels[calib_tune], split_conf_input
    )

    groups = datamodule.graph.ndata[SENS_FIELD]
    N = len(labels[test_mask])
    test_labels = labels[test_mask]
    pred_sets = torch.zeros((N, datamodule.num_classes), dtype=torch.bool)
    pred_sets[torch.arange(N), probs[test_mask].argmax(dim=-1)] = True
    print(
        f"Model Test Accuracy with conformal seed {args.conformal_seed}: {calc_coverage(pred_sets, test_labels)}"
    )

    pred_sets, test_labels = crc.run(
        probs=probs,
        labels=labels,
        split_conf_input=split_conf_input,
    )

    print(f"Mean efficiency: {calc_efficiency(pred_sets)}")
    print(f"Coverage: {calc_coverage(pred_sets, test_labels)}")
    groups = datamodule.graph.ndata[SENS_FIELD]

    res = {
        "c": args.closeness_measure,
        "eff": calc_efficiency(pred_sets),
        "coverage": calc_coverage(pred_sets, test_labels),
        "violation": float("-inf"),
    }

    for label in range(1, datamodule.num_classes):
        res_losses = []
        for g_i in range(datamodule.num_sensitive_groups):
            losses = crc._loss_module.evaluate(
                pred_sets=pred_sets,
                labels=test_labels,
                groups=groups[test_mask],
                pos_label=label,
                group_id=g_i,
                num_classes=datamodule.num_classes,
                alpha=args.alpha,
            )
            prior = crc._compute_prior(
                labels[calib_tune], groups[calib_tune], label, g_i, calib_tune_scores
            )

            if isinstance(losses, Tuple) and isinstance(losses, Tuple):
                temp_losses = []
                for l, p in zip(losses, prior):
                    expected_loss = torch.mean(l) / p
                    expected_loss.nan_to_num_(nan=0, posinf=0)
                    temp_losses.append(expected_loss)
                    print(
                        f"Expected loss (adjusted prior) for y_k = {label} and g_i = {g_i} = {expected_loss}"
                    )
                res_losses.append(temp_losses)
            else:
                expected_loss = torch.mean(losses) / prior
                expected_loss.nan_to_num_(nan=0, posinf=0)
                res_losses.append(expected_loss)
                print(
                    f"Expected loss (adjusted prior) for y_k = {label} and g_i = {g_i} = {expected_loss}"
                )
        print()

        if args.fairness_metric == fairness_metric.Equalized_Odds.name:
            print(
                f"Actual Loss Delta = {max(max([x[0] for x in res_losses]) - min([x[0] for x in res_losses]), max([x[1] for x in res_losses]) - min([x[1] for x in res_losses]))}\n"
            )

            res["violation"] = max(
                max(
                    (
                        max([x[0] for x in res_losses])
                        - min([x[0] for x in res_losses]),
                        max([x[1] for x in res_losses])
                        - min([x[1] for x in res_losses]),
                    )
                ),
                res["violation"],
            )

        else:
            print(f"Actual Loss Delta={max(res_losses) - min(res_losses)}\n")
            res["violation"] = (max(res_losses) - min(res_losses), res["violation"])

    return pred_sets, test_labels, res


def run_conformal_fairness(
    args: ConfFairExptConfig,
    datamodule: DataModule,
    base_ckpt_dir: Optional[str] = None,
):
    # Load probs/labels from base for all (not necessarily labeled) nodes
    probs, labels = load_basegnn_outputs(args, base_ckpt_dir)
    assert (
        probs.shape[1] == datamodule.num_classes
    ), f"Loaded probs has {probs.shape[1]} classes, but the dataset has {datamodule.num_classes} classes"

    test_mask = datamodule.graph.ndata[
        Stage.TEST.mask_dstr
    ]  # Test mask of shape (n_nodes,)

    # Perform Conformal Fairness
    split_conf_input = get_split_conf(args)

    cpf = ScoreSplitFairConformalClassifer(args, datamodule)

    groups = datamodule.graph.ndata[SENS_FIELD]
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
            if args.fairness_metric == fairness_metric.Disparate_Impact.name
            else float("-inf")
        ),
        "eff": calc_efficiency(pred_sets),
        "coverage": calc_coverage(pred_sets, test_labels),
        "violation": (
            float("inf")
            if args.fairness_metric == fairness_metric.Disparate_Impact.name
            else float("-inf")
        ),
    }
    for label in labels:
        coverages = []
        baseline_coverages = []
        for g_i in range(datamodule.num_sensitive_groups):
            match args.fairness_metric:
                case (
                    fairness_metric.Equal_Opportunity.name
                    | fairness_metric.Predictive_Equality.name
                    | fairness_metric.Equalized_Odds.name
                    | fairness_metric.Demographic_Parity.name
                    | fairness_metric.Disparate_Impact.name
                    | fairness_metric.Overall_Acc_Equality.name
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
                            != fairness_metric.Overall_Acc_Equality.name
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
                case fairness_metric.Predictive_Parity.name:
                    try:
                        cpf.config.fairness_metric = (
                            fairness_metric.Equal_Opportunity.name
                        )
                        eo_filtered_test_mask = cpf._get_filter_mask(
                            test_labels, test_groups, label, g_i
                        )

                        cpf.config.fairness_metric = (
                            fairness_metric.Demographic_Parity.name
                        )
                        dp_filtered_test_mask = cpf._get_filter_mask(
                            test_labels, test_groups, label, g_i
                        )
                    finally:
                        cpf.config.fairness_metric = (
                            fairness_metric.Predictive_Parity.name
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

        if args.fairness_metric == fairness_metric.Disparate_Impact.name:
            print(f"Actual Coverage Ratio={min(coverages) / max(coverages)}\n")
            res["violation"] = min(min(coverages) / max(coverages), res["violation"])
            res["base_violation"] = min(
                min(baseline_coverages) / max(baseline_coverages), res["base_violation"]
            )
        elif args.fairness_metric == fairness_metric.Equalized_Odds.name:
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


def run_conformal_fairness_with_avg(
    args: ConfFairExptConfig,
    datamodule: DataModule,
    base_ckpt_dir: Optional[str] = None,
):
    # Load probs/labels from base for all (not necessarily labeled) nodes
    probs, labels = load_basegnn_outputs(args, base_ckpt_dir)
    assert (
        probs.shape[1] == datamodule.num_classes
    ), f"Loaded probs has {probs.shape[1]} classes, but the dataset has {datamodule.num_classes} classes"

    test_mask = datamodule.graph.ndata[
        Stage.TEST.mask_dstr
    ]  # Test mask of shape (n_nodes,)

    # Perform Conformal Fairness
    split_conf_input = get_split_conf(args)

    cpf = ScoreSplitFairConformalClassifer(args, datamodule)

    groups = datamodule.graph.ndata[SENS_FIELD]
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
            if args.fairness_metric == fairness_metric.Disparate_Impact.name
            else float("-inf")
        ),
        "micro_base_violation": 0,
        "macro_base_violation": 0,
        "eff": calc_efficiency(pred_sets),
        "coverage": calc_coverage(pred_sets, test_labels),
        "violation": (
            float("inf")
            if args.fairness_metric == fairness_metric.Disparate_Impact.name
            else float("-inf")
        ),
        "base_violation": (
            float("inf")
            if args.fairness_metric == fairness_metric.Disparate_Impact.name
            else float("-inf")
        ),
        "micro_violation": 0,
        "macro_violation": 0,
    }

    for label in labels:
        coverages = []
        baseline_coverages = []

        for g_i in range(datamodule.num_sensitive_groups):
            match args.fairness_metric:
                case (
                    fairness_metric.Equal_Opportunity.name
                    | fairness_metric.Predictive_Equality.name
                    | fairness_metric.Equalized_Odds.name
                    | fairness_metric.Demographic_Parity.name
                    | fairness_metric.Disparate_Impact.name
                    | fairness_metric.Overall_Acc_Equality.name
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
                            != fairness_metric.Overall_Acc_Equality.name
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
                case fairness_metric.Predictive_Parity.name:
                    try:
                        cpf.config.fairness_metric = (
                            fairness_metric.Equal_Opportunity.name
                        )
                        eo_filtered_test_mask = cpf._get_filter_mask(
                            test_labels, test_groups, label, g_i
                        )

                        cpf.config.fairness_metric = (
                            fairness_metric.Demographic_Parity.name
                        )
                        dp_filtered_test_mask = cpf._get_filter_mask(
                            test_labels, test_groups, label, g_i
                        )
                    finally:
                        cpf.config.fairness_metric = (
                            fairness_metric.Predictive_Parity.name
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

        if args.fairness_metric == fairness_metric.Disparate_Impact.name:
            print(f"Actual Coverage Ratio={min(coverages) / max(coverages)}\n")
            res["violation"] = min((min(coverages) / max(coverages)), res["violation"])
            res["base_violation"] = min(
                (min(baseline_coverages) / max(baseline_coverages)), res["violation"]
            )

            res["micro_violation"] += (min(coverages) / max(coverages)) / len(labels)
            res["micro_base_violation"] += (
                min(baseline_coverages) / max(baseline_coverages)
            ) / len(labels)

            label_perc = (test_labels == label).sum() / (test_labels != 0).sum()

            res["macro_violation"] += (min(coverages) / max(coverages)) * label_perc
            res["macro_base_violation"] += (
                min(baseline_coverages) / max(baseline_coverages)
            ) * label_perc

        elif args.fairness_metric == fairness_metric.Equalized_Odds.name:
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
                res["violation"],
            )

            res["micro_violation"] += max(
                (
                    max([x[0] for x in coverages]) - min([x[0] for x in coverages]),
                    max([x[1] for x in coverages]) - min([x[1] for x in coverages]),
                )
            ) / len(labels)

            res["micro_base_violation"] += max(
                (
                    max([x[0] for x in baseline_coverages])
                    - min([x[0] for x in baseline_coverages]),
                    max([x[1] for x in baseline_coverages])
                    - min([x[1] for x in baseline_coverages]),
                )
            ) / len(labels)

            label_perc = (test_labels == label).sum() / (test_labels != 0).sum()

            res["macro_violation"] += (
                max(
                    (
                        max([x[0] for x in coverages]) - min([x[0] for x in coverages]),
                        max([x[1] for x in coverages]) - min([x[1] for x in coverages]),
                    )
                )
                * label_perc
            )

            res["macro_base_violation"] += (
                max(
                    (
                        max([x[0] for x in baseline_coverages])
                        - min([x[0] for x in baseline_coverages]),
                        max([x[1] for x in baseline_coverages])
                        - min([x[1] for x in baseline_coverages]),
                    )
                )
                * label_perc
            )

        else:
            print(f"Actual Coverage Delta={max(coverages) - min(coverages)}\n")
            res["violation"] = max(max(coverages) - min(coverages), res["violation"])
            res["base_violation"] = max(
                max(baseline_coverages) - min(baseline_coverages), res["base_violation"]
            )

            res["micro_violation"] += (max(coverages) - min(coverages)) / len(labels)
            res["micro_base_violation"] += (
                max(baseline_coverages) - min(baseline_coverages)
            ) / len(labels)

            label_perc = (test_labels == label).sum() / (test_labels != 0).sum()

            res["macro_violation"] += (max(coverages) - min(coverages)) * label_perc
            res["macro_base_violation"] += (
                max(baseline_coverages) - min(baseline_coverages)
            ) * label_perc

    return pred_sets, test_labels, res


def run_conformal(
    args: ConfExptConfig,
    datamodule: DataModule,
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
