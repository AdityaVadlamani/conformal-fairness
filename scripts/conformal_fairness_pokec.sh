#!/bin/bash

PROJECTDIR="$HOME/conformal-fairness"
CONDAENV=fairgraph
TRAIN_FRAC=0.3
VAL_FRAC=0.2
ALPHA=0.1

base_output_dir="$PROJECTDIR/outputs"
best_configs_dir="$PROJECTDIR/configs/custom_configs/best_cfgnn_configs"

for DATASET in "Pokec_z" "Pokec_n"
do
    for SENS_ATTR_PREFIX in "" "_gender" "_region" 
    do
        SENS_ATTR="${SENS_ATTR_PREFIX//_}"
        if [ -z "${SENS_ATTR}" ]; then
            SENS_ATTR="region_gender"
        fi
        echo $SENS_ATTR
        for METRIC in "Equalized_Odds" "Equal_Opportunity" "Predictive_Equality" "Predictive_Parity" "Demographic_Parity" "Disparate_Impact" 
        do
            for METHOD in "tps" "aps" "daps" "cfgnn"
            do
                best_base_path="${base_output_dir}/${DATASET}/split/${TRAIN_FRAC}_${VAL_FRAC}${SENS_ATTR_PREFIX}"
                if [ "$METHOD" = "cfgnn" ]; then
                    config_path="${best_configs_dir}/${DATASET}/split/${TRAIN_FRAC}_${VAL_FRAC}_${SENS_ATTR}/cfgnn_config.yaml"
                else
                    config_path="configs/fairness_default.yaml"
                fi
                for USE_CLASSWISE in True
                do
                    base_job_id="best_${DATASET}_split_${TRAIN_FRAC}_${VAL_FRAC}${SENS_ATTR_PREFIX}"
                    trained_model_dir="${base_output_dir}/${DATASET}/${DATASET}_GAT_split_${TRAIN_FRAC}_${VAL_FRAC}_${SENS_ATTR}"
sbatch <<EOT
#!/bin/bash
#SBATCH --account PAS2030
#SBATCH --partition=gpuserial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-00:00:00
#SBATCH -J c_${DATASET}_${TRAIN_FRAC}_${VAL_FRAC}_${METHOD}
#SBATCH -e ${PROJECTDIR}/logs/${DATASET}/${TRAIN_FRAC}_${VAL_FRAC}${SENS_ATTR_PREFIX}/${METRIC}/use_classwise_${USE_CLASSWISE}/c_${METHOD}_%j.err
#SBATCH -o ${PROJECTDIR}/logs/${DATASET}/${TRAIN_FRAC}_${VAL_FRAC}${SENS_ATTR_PREFIX}/${METRIC}/use_classwise_${USE_CLASSWISE}/c_${METHOD}_%j.out

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
    --dataset.name=${DATASET} \
    --job_id=c_${METRIC}_${METHOD}_${USE_CLASSWISE}${SENS_ATTR_PREFIX} \
    --base_job_id=${base_job_id} \
    --use_classwise_lambdas ${USE_CLASSWISE} \
    --dataset_loading_style split \
    --dataset_split_fractions.train ${TRAIN_FRAC} \
    --dataset_split_fractions.valid ${VAL_FRAC}  \
    --dataset.sens_attrs '["${SENS_ATTR}"]' \
    --confgnn_config.train_fn aps \
    --confgnn_config.trained_model_dir $trained_model_dir \
    --epochs 100 \
    --output_dir ${best_base_path} \
    --alpha $ALPHA \
    --primitive_config.use_tps_classwise False \
    --primitive_config.use_aps_epsilon True
EOT
                done
            done
        done
    done
done