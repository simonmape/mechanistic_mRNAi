#!/bin/bash
#SBATCH --job-name=fixedvar    # create a short name for your job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # number of gpus per node
<<<<<<< HEAD
#SBATCH --ntasks-per-node=1
=======
>>>>>>> 631aa8f580865371508f33e39d7c8436441caf6b
#SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)


module purge
module load Anaconda3
source activate ${DATA}/myenv

python  big_latent_fixedvar.py --batch_size 128 --learning_rate 1e-3
