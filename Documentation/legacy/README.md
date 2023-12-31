## This directory contains the legacy resources. Such as our first simulator sched_env and our primary research.


## Preliminary Training Performance Test on different devices

Access this preliminary research on performance on different platforms via [training performance test](./training_performance_test) folder. 
 a DDPG agent with the architecture [512, 256, 64, 32] for both actor and critic networks implemented by PyTorch2.0 is trained in the ’PENDULUM’ environment on OpenAI-Gym. The training devices used for performance comparison are Crrius, Archer2 and MacBook Pro M1 16GB.

#### Preliminary Research Dependencies
```
"python>=3.8",
"numpy>=1.18",
"torch>=1.3",
"gym>=0.12,<0.26",
"gym[box2d]",
"tqdm>=4.41",
"pyglet>=1.3",
"pytest>=5.3",
```

#### Run
Submit on ARCHER2: `sbatch batch_archer2_test.sh`

Submit on Cirrus CPU: `sbatch batch_cirrus_cpu_test.sh`

Submit on ARCHER2: `sbatch batch_cirrus_gpu_test.sh`

APPLE MacBook Pro M1: `python3 train_ddpg.py` and modify `device` in `agents.py` to `mps` (Metal GPU) or `cpu`

#### Result

The training speed for the DDPG agent is found to be approximately 34 iterations per second (it/s) on the CPU and 198 it/s on the GPU with CUDA. In ARHCER2, the CPU training speed is approximately 90 it/s. However, on the MacBook Pro M1 16GB, the training speed is around 296 it/s on the CPU and 58 it/s on the GPU (Metal).