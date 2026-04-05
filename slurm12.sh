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

# Verification GPU
echo "GPU disponible: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi non disponible"

if [ -z "$1" ]; then
    echo "ERREUR: Aucun script Python specifie."
    exit 1
fi

SCRIPT=$1
shift          # <-- CRITIQUE: decale les arguments, $@ contient maintenant le reste

echo "Script : $SCRIPT"
echo "Args   : $@"
echo "===================================================="

python -u $SCRIPT "$@"   # <-- "$@" passe TOUS les arguments restants a Python

EXIT_CODE=$?
echo "===================================================="
echo "Job finished on $(date) (exit code: $EXIT_CODE)"
echo "===================================================="
exit $EXIT_CODE