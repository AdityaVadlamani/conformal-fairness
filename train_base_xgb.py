import logging
import os
import shutil
from contextlib import ExitStack

import conformal_fairness.utils as utils
import pyrallis.argparsing as pyr_a
from conformal_fairness.config import BaseExptConfig
from conformal_fairness.constants import *
from conformal_fairness.custom_logger import CustomLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        args = utils.load_basegnn_config_from_ckpt(ckpt_dir, args)

    # overwrite any existing config
    utils.output_basegnn_config(ckpt_dir, args)

    utils.set_seed_and_precision(args.seed)
    datamodule = utils.prepare_datamodule(args)

    # datamodule.setup_sampler(args.base_gnn.layers)

    # create logger and log expt hyperparams
    expt_logger = CustomLogger(args.logging_config)
    expt_logger.log_hyperparams(vars(args))

    model: XGBClassifier = utils.load_basexgb(args)  # get model
    model = utils.train_basexgb(model, datamodule)  # fit the model
    results = utils.run_basexgb_inference_alldl(model, datamodule)  # get results
    if results is not None:
        utils.output_basegnn_results(args, results)
    else:
        logger.error("No results to output")
        raise ValueError("No results to output")


if __name__ == "__main__":
    # python train_base_xgb.py  --config_path="configs/base_xgb_default.yaml"
    main()
