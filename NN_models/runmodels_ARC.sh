#!/bin/bash
#SBATCH --job-name=U.2.128    # create a short name for your job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1             # number of gpus per node

#SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)


module purge
module load Anaconda3
conda activate ${DATA}/myenv

python  C-unimodal_model.py --batch_size 128 --learning_rate 1e-2
