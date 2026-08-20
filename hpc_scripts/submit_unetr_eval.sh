#!/bin/bash
#PBS -N unetr_eval
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64g
#PBS -l walltime=00:30:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Initialize Conda
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate GPU-env

# Load CUDA modules
module load cuda

# Force CPU (recommended due to compute node cuDNN environment issues)
export CUDA_VISIBLE_DEVICES=""

# Run the evaluation script
python3 new_work/evaluate_unetr_test.py
