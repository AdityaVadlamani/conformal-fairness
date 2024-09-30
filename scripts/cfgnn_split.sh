#!/bin/bash

SCRIPTDIR="$HOME/conformal-fairness"
base_output_dir="$SCRIPTDIR/outputs"
best_configs_dir="$SCRIPTDIR/configs/custom_configs/best_cfgnn_configs"

CONDAENV=fairgraph
ALPHA=0.1
train_frac=0.3
val_frac=0.2


for DATASET in "Pokec_n" "Pokec_z"; do
    for SENS_ATTR in "region_gender" "region" "gender"
    do

        SENS_ATTR_PREFIX="_${SENS_ATTR}"
        if [ "$SENS_ATTR" = "region_gender" ]; then
            BASE_SENS_ATTR_PREFIX=""
        else
            BASE_SENS_ATTR_PREFIX="_${SENS_ATTR}"
        fi

        best_base_path="${base_output_dir}/${DATASET}/split/${train_frac}_${val_frac}${BASE_SENS_ATTR_PREFIX}"
        best_cfgnn_path="${best_configs_dir}/${DATASET}/split/${train_frac}_${val_frac}${SENS_ATTR_PREFIX}/cfgnn_config.yaml"

        trained_model_dir="${base_output_dir}/${DATASET}/${DATASET}_GAT_split_${train_frac}_${val_frac}${SENS_ATTR_PREFIX}"
        if [ ! -f "$best_cfgnn_path" ]; then
            echo "Best cfgnn run not found for ${DATASET} with train_frac=${train_frac} and val_frac=${val_frac}"
            continue
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
#SBATCH -J best_cfgnn_${DATASET}
#SBATCH -o ./logs/best/${DATASET}/cfgnn/best_${train_frac}_${val_frac}${SENS_ATTR_PREFIX}.out
#SBATCH -e ./logs/best/${DATASET}/cfgnn/best_${train_frac}_${val_frac}${SENS_ATTR_PREFIX}.err

echo Job started at `date` on `hostname`
source ~/.bashrc
conda deactivate
conda activate $CONDAENV

export DGLBACKEND=pytorch

cd ${SCRIPTDIR}
srun python run_conformal_fairness.py --config_path=${best_cfgnn_path} \
--logging_config.use_wandb False \
--output_dir ${best_base_path} \
--alpha ${ALPHA} \
--epochs 100 \
--conformal_method cfgnn \
--confgnn_config.train_fn aps \
--confgnn_config.trained_model_dir $trained_model_dir
EOT
    done
done