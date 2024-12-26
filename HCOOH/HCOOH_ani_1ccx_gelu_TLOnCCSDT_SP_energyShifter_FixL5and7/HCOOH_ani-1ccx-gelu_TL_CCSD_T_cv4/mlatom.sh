#!/bin/bash
#SBATCH -J fateme
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -G 1
#SBATCH -o Job%J_%x_%N_out
#SBATCH -e Job%J_%x_%N_err
#SBATCH -w b008


#----> Job begins <----
source ~/.bashrc
export PYTHONPATH="/mlatom/home/fatemealavi/mlatom_devBranch"
python train.py > output_cv4