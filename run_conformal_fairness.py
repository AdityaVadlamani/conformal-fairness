import logging
import os

from conformal_fairness import utils
import pandas as pd
import pyrallis.argparsing as pyr_a
from conformal_fairness.config import ConfFairExptConfig
from conformal_fairness.constants import ConformalMethod, FairnessMetric

# not adding this to constants since this is only a utility for easy wandb plots
CUSTOM_STEP = "custom_step"  # helper for wandb plots
tern_logger = logging.getLogger(__name__)


def main() -> None:
    args = pyr_a.parse(config_class=ConfFairExptConfig)

    # make sure that the sampling config for the expt is same as that of the base job
    base_ckpt_dir, _ = utils.get_base_ckpt_dir_fname(
        args.output_dir, args.dataset.name, args.base_job_id
    )
    base_expt_config = utils.load_base_config_from_ckpt(base_ckpt_dir)
    utils.check_sampling_consistent(base_expt_config, args)

    if args.calib_test_equal:
        assert args.dataset_split_fractions is not None
        args.dataset_split_fractions.calib = (
            (
                1
                - args.dataset_split_fractions.train
                - args.dataset_split_fractions.valid
            )
        ) / 2

    original_conformal_seed = args.conformal_seed
    original_c = args.closeness_measure
    # setup dataloaders
    utils.set_seed_and_precision(args.seed)
    datamodule = utils.prepare_datamodule(args)

    fairness_trials_dir = f"analysis/fairness_trials/{args.dataset.name}_{'_'.join(args.dataset.sens_attrs)}/{args.fairness_metric}/{args.use_classwise_lambdas}"
    os.makedirs(
        fairness_trials_dir,
        exist_ok=True,
    )

    dfs = []
    try:
        if args.fairness_metric == FairnessMetric.DISPARATE_IMPACT.value:
            c_list = [0.8]
        else:
            c_list = [0.05, 0.1, 0.15, 0.20]

        for c in c_list:
            args.closeness_measure = c
            for c_seed in range(10):
                args.conformal_seed = c_seed
                # reshuffle the calibration and test sets if required
                datamodule.resplit_calib_test(args)

                if args.conformal_method in [
                    ConformalMethod.DAPS,
                    ConformalMethod.DTPS,
                ]:
                    args.diffusion_config.use_tps_classwise = True

                if args.conformal_method in [
                    ConformalMethod.DAPS,
                    ConformalMethod.DTPS,
                    ConformalMethod.CFGNN,
                ]:
                    datamodule.split_calib_tune_qscore(tune_frac=args.tuning_fraction)
                else:
                    # No prior or extra probabilities needed
                    datamodule.split_calib_tune_qscore(tune_frac=0)

                print("\n\nRunning with conformal prediction fairness:")
                _, _, res = utils.run_conformal_fairness(args, datamodule)

                dfs.append(pd.DataFrame({k: [v] for k, v in res.items()}))

            df = pd.concat(dfs)

            if (
                args.conformal_method == ConformalMethod.TPS
                and args.primitive_config.use_tps_classwise
            ):
                df.to_csv(f"{fairness_trials_dir}/tps_classwise.csv")
            elif (
                args.conformal_method == ConformalMethod.APS
                and not args.primitive_config.use_aps_epsilon
            ):
                df.to_csv(f"{fairness_trials_dir}/aps_no_rand.csv")
            else:
                df.to_csv(f"{fairness_trials_dir}/{args.conformal_method}.csv")
    finally:
        args.conformal_seed = original_conformal_seed
        args.closeness_measure = original_c


if __name__ == "__main__":
    # python run_conformal_fairness.py --config_path="configs/fairness_default.yaml"
    main()
