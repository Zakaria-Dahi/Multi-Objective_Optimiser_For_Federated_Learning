#!/bin/bash -l


#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100

#SBATCH --output=./logs/log_%A-%a.out    # Standard output and error log
#SBATCH --error=./logs/log_%A-%a.err    # Standard output and error log

date
hostname

################

# Ensure process affinity is disabled
export SLURM_CPU_BIND=none

# Prepare in the current folder a worker launcher for Scoop 
# The scipt below will 'decorate' the python interpreter command
# Before python is called, modules are loaded
HOSTFILE=$(pwd)/hostfile
SCOOP_WRAPPER=$(pwd)/src/start.sh

cat << EOF > $SCOOP_WRAPPER
#!/bin/bash -l
#module load lang/Python
source $(pwd)/../optfl_env/bin/activate
EOF
echo 'python $@' >> $SCOOP_WRAPPER

chmod +x $SCOOP_WRAPPER

# Classical "module load" in the main script
module load lang/Python
source $(pwd)/../optfl_env/bin/activate

# Save the hostname of the allocated nodes
scontrol show hostnames > $HOSTFILE

# Start scoop with python input script
INPUTFILE=$(pwd)/src/nsga2.py 
python -m scoop --hostfile $HOSTFILE -n 100 --python-interpreter=$SCOOP_WRAPPER $INPUTFILE $@
#-n 100
