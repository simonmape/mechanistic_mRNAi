#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --array=1-25
#SBATCH --job-name=data_gen
#SBATCH --ntasks-per-node=1
#SBATCH --partition=htc

#Go into the DeepScratch example
SIM_PATH=${DATA}/pakman-develop/examples/DeepScratch

cd ${SIM_PATH}

./MakeSamples Mock.txt ${SLURM_ARRAY_TASK_ID}
