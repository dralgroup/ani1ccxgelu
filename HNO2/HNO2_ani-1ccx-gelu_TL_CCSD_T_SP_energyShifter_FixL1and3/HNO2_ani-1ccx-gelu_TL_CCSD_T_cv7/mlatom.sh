#!/bin/bash
#SBATCH -J fateme
#SBATCH -p cpu
#SBATCH -c 14
#SBATCH -o Job%J_%x_%N_out
#SBATCH -e Job%J_%x_%N_err
#SBATCH -w b006


#----> Job begins <----
source ~/.bashrc
export PYTHONPATH="/mlatom/home/fatemealavi/mlatom_devBranch"
python train.py > output_cv7