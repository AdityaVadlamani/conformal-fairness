from dataclasses import dataclass, field
from typing import List, Optional

from conformal_fairness.config import BaseExptConfig, ConfExptConfig
from conformal_fairness.constants import layer_types, sample_type


@dataclass
class TuneSplitConfig:
    """Config for the split style of dataset loading."""

    # dataset loading expt style (split or n_samples_per_class)
    s_type: str = field(default=sample_type.split.name)
    # num samples per class options to try
    samples_per_class: List[int] = field(default_factory=list)
    # train split fraction options to try
    train_fracs: List[float] = field(default_factory=list)
    # valid split fraction options to try
    val_fracs: List[float] = field(default_factory=list)


@dataclass
class BaseTuneExptConfig:
    """
    Overall config for the base model tuning.
    Each expt uses a single dataset and runs either split or n_samples_per_class style loading.
    """

    # num samples total across the parameter space
    num_samples: int = field(default=20)
    # num trials per config (different seed in every trial)
    # seed will be set from 0 to n_trails_per_config - 1
    n_trials_per_config: int = field(default=10)
    # metric to optimize
    metric_used: str = field(default="val_acc")
    # whether to maximize or minimize metric
    metric_mode: str = field(default="max")
    # aggregation function
    metric_aggr: str = field(default="mean")
    # layer types
    l_types: List[str] = field(default_factory=lambda: [lt.name for lt in layer_types])
    # tuning config
    tune_split_config: TuneSplitConfig = field(default_factory=TuneSplitConfig)

    # Config to use for a specific expt
    expt_config: BaseExptConfig = field(default_factory=BaseExptConfig)

    # tune_output_dir
    tune_output_dir: Optional[str] = None
    # number of tune simultaneous tune workers
    n_tune_workers: int = field(default=1)


@dataclass
class ConfGNNTuneExptConfig(BaseTuneExptConfig):
    """
    Overall config for the conformal GNN model tuning.
    Each expt uses a single dataset and runs either split or n_samples_per_class style loading.
    """

    # Config to use for a specific expt
    conf_expt_config: ConfExptConfig = field(default_factory=ConfExptConfig)
    # base model directory
    base_model_dir: str = field(default="base_model_dir")
    # split calib and test to be 50-50
    calib_test_equal: bool = field(default=True)


@dataclass
class ConfTrialsExptConfig:
    # directory containing all trials to run
    expt_configs_dir: str = field(default="configs")
    # num trials per config
    trials_per_config: int = field(default=50)
    # base model directory
    base_model_dir: str = field(default="base_model_dir")
    # results output directory
    results_output_dir: str = field(default="results_output_dir")
    # expt alpha
    alpha: float = field(default=0.1)
    # split calib and test to be 50-50
    calib_test_equal: bool = field(default=True)
