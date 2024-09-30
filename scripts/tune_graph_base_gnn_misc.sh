#!/bin/bash
for DATASET in "Pokec_z" #"Pokec_n" #
do
    for SENS_ATTR in "region" "gender" # ""
    do
	    for L_TYPES in "GCN" "GraphSAGE" "GAT" 
	    do
		    for TRAIN_FRACS in 0.2 0.3
		    do
			    for VAL_FRACS in 0.1 0.2
			    do
    
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
#SBATCH -o ./logs/tune/${DATASET}/tune_${L_TYPES}_${TRAIN_FRACS}_${VAL_FRACS}_${SENS_ATTR}.out
#SBATCH -e ./logs/tune/${DATASET}/tune_${L_TYPES}_${TRAIN_FRACS}_${VAL_FRACS}_${SENS_ATTR}.err

source ~/.bashrc
conda deactivate
conda activate fairgraph

export DGLBACKEND=pytorch

cd ${SCRIPTDIR}
srun python hpt_base_gnn.py --config_path="configs/hpt_base_gnn_default.yaml" --expt_config.dataset.name ${DATASET} --expt_config.dataset.sens_attrs '["${SENS_ATTR}"]' --tune_split_config.s_type split --l_types '["${L_TYPES}"]' --tune_split_config.train_fracs "[${TRAIN_FRACS}]" --tune_split_config.val_fracs "[${VAL_FRACS}]"
EOT
				done
			done
		done
    done
done
