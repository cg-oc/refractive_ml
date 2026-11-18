#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=10:00:00
#SBATCH --job-name=lin_opt
#SBATCH --output=resultado_lin-%j.out
#SBATCH --error=python_lin_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ilumpy

echo "Inicio: $(date)"
echo "No: $(hostname)"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u lineares_opt.py

echo "Fim: $(date)"
