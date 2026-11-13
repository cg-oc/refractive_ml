#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2:00:00
#SBATCH --job-name=models_opt
#SBATCH --output=resultado-%j.out
#SBATCH --error=python_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ilumpy

echo "Inicio: $(date)"
echo "No: $(hostname)"

python -u models_opt.py

echo "Fim: $(date)"
