#!/bin/bash
#

#SBATCH --account=m22oc-s2311726
#SBATCH --job-name=cuda-rl-test
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --qos=short


#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


source /work/m22oc/m22oc/s2311726/uoeRL/bin/activate
python3 ./train_ddpg.py
