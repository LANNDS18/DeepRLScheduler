#!/bin/bash


#SBATCH --account=m22oc
#SBATCH --job-name=rl_test
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=256
#SBATCH --partition=standard
#SBATCH --qos=short
# SBATCH --reservation=shortqos

# You can probably leave these options mostly as they are

#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --exclusive
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1


# Launch the parallel job

# module unload intel-20.4/compilers
# module load gcc/10.2.0

# export OMP_NUM_THREADS=64

module load cray-python

source /work/m22oc/m22oc/s2311726/rl-test-env/bin/activate

python3 ./train_ddpg.py