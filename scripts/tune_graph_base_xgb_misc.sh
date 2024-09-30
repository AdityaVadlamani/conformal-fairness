#!/bin/bash

DATASET="Credit"
SENS_ATTR="Age"
DISCARD_ATTRS="NoDefaultNextMonth,Single"
PRED_ATTRS="EducationLevel"
TRAIN_FRAC=0.3
VAL_FRAC=0.2

SCRIPTDIR="$HOME/conformal-fairness"

sbatch <<EOT
#!/bin/bash
#SBATCH --account PAS2030
#SBATCH --partition=gpuserial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-00:00:00
#SBATCH -J tune_${DATASET}
#SBATCH -o ./logs/tune/${DATASET}/tune_${TRAIN_FRAC}_${VAL_FRAC}_${SENS_ATTR}.out
#SBATCH -e ./logs/tune/${DATASET}/tune_${TRAIN_FRAC}_${VAL_FRAC}_${SENS_ATTR}.err

source ~/.bashrc
conda deactivate
conda activate fairgraph

export DGLBACKEND=pytorch

cd ${SCRIPTDIR}
srun python hpt_base_xgb.py --config_path="configs/base_xgb_default.yaml" --dataset.name ${DATASET} --dataset.sens_attrs '["${SENS_ATTR}"]' --dataset.sens_attrs '["${SENS_ATTR}"]' --dataset.discard_attrs '["${DISCARD_ATTRS}"]' --dataset.pred_attrs '["${PRED_ATTRS}"]' --dataset_loading_style split --dataset_split_fractions.train ${TRAIN_FRAC} --dataset_split_fractions.valid ${VAL_FRAC}
EOT
