#!/bin/bash

PROJECTDIR="$HOME/conformal-fairness"
CONDAENV=conformal-fairness
TRAIN_FRAC=0.3
VAL_FRAC=0.2
ALPHA=0.1

base_output_dir="$PROJECTDIR/outputs"

for DATASET in ACSIncome ACSEducation
do
    for METRIC in "Equalized_Odds" "Overall_Acc_Equality" "Equal_Opportunity" "Predictive_Equality" "Predictive_Parity" "Demographic_Parity" "Disparate_Impact"
    do
        for METHOD in  "tps" "aps"
        do
            config_path="configs/fair_cp_${DATASET}.yaml"

            for USE_CLASSWISE in False True
            do
                for INVERT_PRIM in False True
                do 
                    if [ "$INVERT_PRIM" = "False" ]; then
                        tps_class=False
                        aps_rand=True
                    else
                        tps_class=True
                        aps_rand=False
                    fi
sbatch <<EOT
#!/bin/bash
#SBATCH --account PAS2030
#SBATCH --partition=gpuserial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-00:00:00
#SBATCH -J c_${DATASET}_${TRAIN_FRAC}_${VAL_FRAC}_${METHOD}_${USE_CLASSWISE}_${INVERT_PRIM}
#SBATCH -e ${PROJECTDIR}/scripts/logs_10/${DATASET}_${TRAIN_FRAC}_${VAL_FRAC}_${METRIC}_closeness_0.01_use_classwise_${USE_CLASSWISE}_inv_prim_${INVERT_PRIM}_j.err
#SBATCH -o ${PROJECTDIR}/scripts/logs_10/${DATASET}_${TRAIN_FRAC}_${VAL_FRAC}_${METRIC}_closeness_0.01_use_classwise_${USE_CLASSWISE}_inv_prim_${INVERT_PRIM}_j.out

echo Job started at `date` on `hostname`
# CONDA SETUP
source ~/.bashrc
conda deactivate
conda activate ${CONDAENV}

export DGLBACKEND=pytorch

cd ${PROJECTDIR}
python run_conformal_fairness.py --config_path=${config_path} \
    --logging_config.use_wandb False \
    --fairness_metric=${METRIC} \
    --conformal_method=${METHOD} \
    --job_id=c_${DATASET}_${TRAIN_FRAC}_${VAL_FRAC}_${METHOD}_${USE_CLASSWISE}_${INVERT_PRIM} \
    --use_classwise_lambdas ${USE_CLASSWISE} \
    --dataset_split_fractions.train ${TRAIN_FRAC} \
    --dataset_split_fractions.valid ${VAL_FRAC}  \
    --alpha $ALPHA \
    --primitive_config.use_tps_classwise $tps_class \
    --primitive_config.use_aps_epsilon $aps_rand
EOT
                done
            done
        done
    done
done
