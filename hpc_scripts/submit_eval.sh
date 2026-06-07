#!/bin/bash
#PBS -N stroke_eval
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=32g
#PBS -j oe

cd $PBS_O_WORKDIR

# Initialize conda
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate cod-gpu

conda install -y pandas seaborn matplotlib


# FORCE CPU (recommended)
export CUDA_VISIBLE_DEVICES=""

python evaluate.py
