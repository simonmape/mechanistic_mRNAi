#!/bin/bash
#SBATCH --job-name=nolatent    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes            # number of gpus per node
#SBATCH --gres=gpu:1 --constraint='gpu_cc:3.7'
#SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=htc

module purge
module load python/anaconda3/2019.03
source activate ${DATA}/myenv

python  nolatent_fixedvar.py --batch_size 64 --learning_rate 1e-3
