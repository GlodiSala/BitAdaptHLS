#!/bin/sh
#SBATCH --job-name=BitAdapt_MIMO
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=50:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "===================================================="
echo "Job started on $(date)"
echo "Node: $(hostname)"
echo "Directory: $(pwd)"
echo "===================================================="

mkdir -p logs

source /export/tmp/sala/anaconda3/etc/profile.d/conda.sh
conda activate bitadapt

export PYTHONPATH=$PYTHONPATH:$(pwd)

if [ -z "$1" ]; then
    echo "ERREUR: Aucun script specifie"
    exit 1
fi

SCRIPT=$1
shift          # ← NOUVEAU: decale $1, $@ contient maintenant le reste

echo "Script : $SCRIPT"
echo "Args   : $@"
echo "===================================================="

python -u $SCRIPT "$@"   # ← NOUVEAU: "$@" passe tous les arguments

echo "===================================================="
echo "Job finished on $(date)"
echo "===================================================="