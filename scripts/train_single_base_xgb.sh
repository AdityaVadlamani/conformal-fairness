#!/bin/bash
    
SCRIPTDIR="$HOME/conformal-fairness"

CONFIGFILENAME="optimal_basexgb_acs_income"

sbatch <<EOT
#!/bin/bash
#SBATCH --account PAS2030
#SBATCH --partition=gpuserial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH -J tune_acsincome
#SBATCH -o ${SCRIPTDIR}/scripts/logs9/train_${CONFIGFILENAME}.out
#SBATCH -e ${SCRIPTDIR}/scripts/logs9/train_${CONFIGFILENAME}.err

source ~/.bashrc
conda deactivate
conda activate fairgraph

export DGLBACKEND=pytorch

cd ${SCRIPTDIR}
srun python train_base_xgb.py  --config_path="configs/${CONFIGFILENAME}.yaml"
EOT
