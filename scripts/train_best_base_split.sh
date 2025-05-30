#!/bin/bash

PROJECTDIR="$HOME/conformal-fairness"
CONDAENV=fairgraph

best_run_dir="${PROJECTDIR}/configs/custom_configs/best_base_configs"
sens_attr_prefix="_region" #"_gender" ""

for DATASET in "Pokec_n" "Pokec_z"; do
    for train_frac in 0.2 0.3; do
        for val_frac in 0.1 0.2; do
            best_param_path="${best_run_dir}/${DATASET}/split/${train_frac}_${val_frac}${sens_attr_prefix}/base_model_config.yaml"
            if [ ! -f $best_param_path ]; then
                echo "Best parameter file not found for ${DATASET} with train_frac=${train_frac} and val_frac=${val_frac}"
                continue
            fi
            config_output_dir="$PROJECTDIR/outputs/${DATASET}/split/${train_frac}_${val_frac}${sens_attr_prefix}"
            mkdir -p ${config_output_dir}
            job_id="best_${DATASET}_split_${train_frac}_${val_frac}${sens_attr_prefix}"
sbatch <<EOT
#!/bin/bash
#SBATCH --account PAS2030
#SBATCH --partition=gpuserial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-00:00:00
#SBATCH -J best_${DATASET}
#SBATCH -o ${PROJECTDIR}/logs/best/${DATASET}/best_${train_frac}_${val_frac}${sens_attr_prefix}.out
#SBATCH -e ${PROJECTDIR}/logs/best/${DATASET}/best_${train_frac}_${val_frac}${sens_attr_prefix}.err

echo Job started at `date` on `hostname`
# CONDA SETUP
source ~/.bashrc
conda deactivate
conda activate ${CONDAENV}

export DGLBACKEND=pytorch

cd ${PROJECTDIR}
python train_base_gnn.py --config_path=${best_param_path} --logging_config.use_wandb False --output_dir ${config_output_dir} --dataset_dir ${PROJECTDIR}/datasets --job_id ${job_id} --epochs 100
EOT
        done
    done
done
