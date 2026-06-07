#!/bin/bash
#PBS -N cc_tune
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64g
#PBS -l walltime=02:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Initialize Conda
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate GPU-env

# Load CUDA modules
module load cuda

# GPU settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the tuning script
python3 tune_cc_non_leaked.py
