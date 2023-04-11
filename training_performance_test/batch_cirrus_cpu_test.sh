#!/bin/bash

# Replace "account=dc116" below with your own
# budget code (e.g. dc116-s1234567)
#
# For more than 2 nodes:
#
# replace "--qos=short" with "--qos=standard"
# delete "--reservation=shortqos"

#SBATCH --account=m22oc-s2311726
#SBATCH --job-name=rl_test
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=72
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --reservation=shortqos

# You can probably leave these options mostly as they are

#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --exclusive
#SBATCH --tasks-per-node=36
#SBATCH --cpus-per-task=1


# Launch the parallel job

module unload intel-20.4/compilers
module load gcc/10.2.0

export OMP_NUM_THREADS=64

source /work/m22oc/m22oc/s2311726/uoeRL/bin/activate

python3 ./train_ddpg.py