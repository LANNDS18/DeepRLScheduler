# hpc_rl_simulator

Welcome to the `hpc_rl_simulator` package! Dive in to see its components and capabilities.

## Package Structure

- **🤖 ./agent**: This sub-package contains the custom actor-critic models implemented using torch.
  
- **🏢 ./cluster**: This sub-package has the cluster module. It's tailored according to the cluster setting of each distinct workload dataset.

- **📦 ./job**: Dive into the smallest unit of our simulation environment. Each job is crafted and transformed by mapping it to each entry in the workload trace dataset.

- **📚 ./workload**: It houses the parser function that reads, loads, and initializes the workload object from `.swf` files. This sub-package is the manager of all jobs loaded from job traces.

- **🔢 ./scorer**: Interested in computing job scores? This module is the go-to. Whether it's SJF, F1, or other priority-based scores during scheduling, this module has got you covered. And yes, it also computes their performance metrics.

- **🌍 ./env**: The heartbeat of our package! In this module, we've crafted the HPC scheduling simulator which optionally supports backfilling. And guess what? We've adapted this simulation environment to openai gym! This means it can be trained and learned, staying true to the basic ethos of the openai gym environment.

- **🧰 ./common**: This sub-package is your utility belt. It's packed with handy functions, including wrappers for training, validation, and evaluation environments.

## Usage

```python
from hpc_rl_simulator.env import GymSchedulerEnv
    
workload_file = './dataset/*****.swf'

env = GymSchedulerEnv(workload_file=workload_file, trace_sample_range=[0, 0.5], back_fill=False, seed=0)
env.reset()
done = False
    
while not done:
    action = env.action_space.sample()
    state, rwd, done, info = env.step(action)
    print(info)
```