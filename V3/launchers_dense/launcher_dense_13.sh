#!/bin/bash -l


#SBATCH --output=./logs/%A_%a.out
#SBATCH --error=./logs/%A_%a.err

#SBATCH --mem=400G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128

date
hostname

################

# Ensure process affinity is disabled
export SLURM_CPU_BIND=none

# Prepare in the current folder a worker launcher for Scoop 
# The scipt below will 'decorate' the python interpreter command
# Before python is called, modules are loaded
HOSTFILE=$(pwd)/hostfile

#!/bin/bash -l
source $(pwd)/../optfl_env/bin/activate
EOF


# Classical "module load" in the main script
source $(pwd)/../optfl_env/bin/activate

# Save the hostname of the allocated nodes
scontrol show hostnames > $HOSTFILE

# Start scoop with python input script
INPUTFILE=$(pwd)/../src/nsga2.py 
python -m scoop --hostfile $HOSTFILE -n 105 $INPUTFILE 13 "DENSE" 4 $@
