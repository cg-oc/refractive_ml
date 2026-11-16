#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --job-name=rfr_opt
#SBATCH --output=resultado_rfr-%j.out
#SBATCH --error=python_rfr_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ilumpy

echo "Inicio: $(date)"
echo "No: $(hostname)"

python -u floresta_opt.py

echo "Fim: $(date)"
