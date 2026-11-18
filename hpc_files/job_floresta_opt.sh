#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --job-name=rfr_opt
#SBATCH --output=resultado_rfr-%j.out
#SBATCH --error=python_rfr_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ilumpy

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Inicio: $(date)"
echo "No: $(hostname)"

python -u floresta_opt.py

echo "Fim: $(date)"
