import dataclasses
import enum
import json
import logging
from typing import Dict, Optional

import wandb
import wandb.plot
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only

from .config import LoggingConfig


# helper for logging dataclasses as dict
# see https://stackoverflow.com/a/51286749
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


class LoggerType(enum.Enum):
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
    TERMINAL = "terminal"


class TerminalLogger:
    def __init__(self, name) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

    def log_hyperparams(self, params):
        self.logger.info(
            f"Hyperparams: {json.dumps(params, cls=EnhancedJSONEncoder, indent=2)}"
        )

    def log_metrics(self, metrics, step):
        self.logger.info(f"Step: {step} Metrics: {json.dumps(metrics, indent=2)}")

    def finalize(self, status):
        self.logger.info(f"Experiment finished with status: {status}")

    def save(self):
        pass

    def log_histogram(self, *args):
        pass


def wrapped_call(loggers_dict, func_name, *args, **kwargs):
    for _, logger in loggers_dict.items():
        getattr(logger, func_name)(*args, **kwargs)


class CustomLogger(Logger):
    def __init__(self, config: Optional[LoggingConfig]) -> None:
        self.config = config
        self.loggers_used: Dict[LoggerType, Logger] = {}

        if config is not None:
            if config.use_wandb:
                assert (
                    config.wandb_config is not None
                ), "Wandb config must be provided if using wandb"
                run_name = config.wandb_config.run_name
                kwargs = {
                    "project": config.wandb_config.project_name,
                    "log_model": config.wandb_config.log_model,
                    "group": config.wandb_config.group,
                    "job_type": config.wandb_config.job_type,
                    "offline": config.wandb_config.offline,
                    "name": run_name,
                }

                self.wandb_logger = WandbLogger(**kwargs)
                if run_name is None:
                    run_name = self.wandb_logger.experiment.name
                    config.wandb_config.run_name = run_name
                print("Run name for wandb logger: ", run_name)

                self.loggers_used[LoggerType.WANDB] = self.wandb_logger

            if config.use_tensorboard:
                raise NotImplementedError

            if config.use_terminal:
                self.terminal_logger = TerminalLogger(self.name)
                self.loggers_used[LoggerType.TERMINAL] = self.terminal_logger

    @property
    def name(self) -> str:
        return "CustomLogger"

    @property
    def version(self) -> str:
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        wrapped_call(self.loggers_used, "log_hyperparams", params)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        wrapped_call(self.loggers_used, "log_metrics", metrics, step)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        wrapped_call(self.loggers_used, "save")

    @rank_zero_only
    def finalize(self, status):
        wrapped_call(self.loggers_used, "finalize", status)

    def update_config(self, expt_config):
        if LoggerType.WANDB in self.loggers_used:
            self.wandb_logger.experiment.config.update(expt_config)

    def log_histogram(self, title, name, values, step=None):
        if LoggerType.WANDB in self.loggers_used:
            data = [[v] for v in values]
            table = wandb.Table(data=data, columns=[name])
            hist = wandb.plot.histogram(table, name, title=title)
            self.wandb_logger.experiment.log({f"{name}_hist": hist}, step=step)

    def log_table(self, title, data, columns, step=None):
        if LoggerType.WANDB in self.loggers_used:
            table = wandb.Table(data=data, columns=columns)
            self.wandb_logger.experiment.log({title: table}, step=step)

    def force_exit(self):
        if LoggerType.WANDB in self.loggers_used:
            self.wandb_logger.experiment.finish()
