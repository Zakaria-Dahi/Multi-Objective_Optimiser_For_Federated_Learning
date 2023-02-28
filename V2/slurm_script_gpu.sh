#!/bin/bash

#SBATCH --job-name=optfl_gpu
####SBATCH --time=7-00:0
#SBATCH --mem=200G
#SBATCH --ntasks=1
# ojo, usar los cores que se pidan aquí
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
## SBATCH --array=1,10,20-25,100-105

#SBATCH --output=./logs/log_%A-%a.out    # Standard output and error log
#SBATCH --error=./logs/log_%A-%a.err    # Standard output and error log

date
hostname

#lshw -C display


#lspci -v | less

nvidia-smi

#export SLURM_ARRAYID
#echo SLURM_ARRAYID: $SLURM_ARRAYID
#echo TASKID: $SLURM_ARRAY_TASK_ID
#sleep 10

###time python ./program/program.py

time python ./src/nsga2_alternative.py 1 "DENSE" 4

echo `date` terminado 
