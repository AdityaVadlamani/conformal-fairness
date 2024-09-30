#!/bin/bash

PROJECTDIR="$HOME/conformal-fairness"
TRAIN_FRACS=0.3
VAL_FRACS=0.2
best_run_dir="$PROJECTDIR/outputs"

for DATASET in "Pokec_n" "Pokec_z"
do
    for SENS_ATTR_PREFIX in "_region" "_gender" ""
    do
		SENS_ATTR="${SENS_ATTR_PREFIX//_}"
		if [ -z "${SENS_ATTR}" ]; then
			SENS_ATTR="region_gender"
		fi
		echo $SENS_ATTR
	    for L_TYPES in "GCN" "GraphSAGE" "GAT" 
	    do    
			base_job_id="best_${DATASET}_split_${TRAIN_FRACS}_${VAL_FRACS}${SENS_ATTR_PREFIX}"
			job_id="hpt_cfgnn_${DATASET}_split_${TRAIN_FRACS}_${VAL_FRACS}${SENS_ATTR_PREFIX}"
			best_base_path="${best_run_dir}/${DATASET}/split/${TRAIN_FRACS}_${VAL_FRACS}${SENS_ATTR_PREFIX}/${DATASET}/best_${DATASET}_split_${TRAIN_FRACS}_${VAL_FRACS}${SENS_ATTR_PREFIX}"
			if [ ! -d "$best_base_path" ]; then
				echo "Best base run not found for ${DATASET} with train_frac=${TRAIN_FRACS} and val_frac=${VAL_FRACS}"
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
#SBATCH -J tune_${DATASET}
#SBATCH -o ./logs/tune/${DATASET}/cfgnn/tune_${L_TYPES}_${TRAIN_FRACS}_${VAL_FRACS}${SENS_ATTR_PREFIX}.out
#SBATCH -e ./logs/tune/${DATASET}/cfgnn/tune_${L_TYPES}_${TRAIN_FRACS}_${VAL_FRACS}${SENS_ATTR_PREFIX}.err

source ~/.bashrc
conda deactivate
conda activate fairgraph

export DGLBACKEND=pytorch

cd $PROJECTDIR
srun python hpt_conf_gnn.py --config_path="configs/hpt_conf_gnn_default.yaml" --base_model_dir=${best_base_path} --conf_expt_config.dataset.name ${DATASET} --conf_expt_config.dataset.sens_attrs '["${SENS_ATTR}"]' --tune_split_config.s_type split --l_types '["${L_TYPES}"]' --tune_split_config.train_fracs "[${TRAIN_FRACS}]" --tune_split_config.val_fracs "[${VAL_FRACS}]"
EOT
		done
    done
done
