# INFR11164: Dissertation Project
## Leveraging Deep Reinforcement Learning on HPC Batch Job Schedulers
### Wuwei Zhang
### s2311726
### Supervisor: Caoimhín Laoide-Kemp

## Abstract: 

This dissertation explores the potential of integrating Deep Reinforcement Learning (DRL) into High Performance Computing (HPC) job scheduling. We began by formu- lating the HPC job scheduling problem as a Markov Decision Process (MDP) within a specialized HPC job scheduling simulation setup. This simulation is constructed with modules such as workload, job, and simulator to allow for an intricate depiction of state space, capturing aspects like the waiting job queue and the current state of the cluster. We crafted an action space consisting of 128 distinct actions to represent var- ious job scheduling situations, including potential non-operative or invalid schedules. Central to our methodology is the sigmoid reward function, governed by a hyperparam- eter k, which standardizes performance metrics to fit a [0, 1] interval. This function is evaluated against the episodic reward function and direct performance metric rewards. Leveraging the Proximal Policy Optimization (PPO) technique from Stable Baseline3, we incorporated a kernel-driven policy network and an MLP-centric value network for the agent’s training process. Our strategies, especially when employing the PPO method, demonstrated exemplary outcomes on the HPC2N dataset, marked by a no- table decrease in the average bounded slowdown. However, transitioning to black-box datasets, such as SDSC-SP2, the agent’s performance showed variability, suggesting its reliance on the initial training setup and reward function. In essence, our research explores DRL’s capability to potentially surpass conventional priority-based scheduling approaches in HPC job scheduling problems, albeit its success being closely linked to the nature of training data and the design of the reward function.

## Table of Contents

- [Installation](#installation)
- [Project Architecture](#project-architecture)
- [Running](#running)

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Project Architecture

- `./Documentation`: This directory contains project meeting minutes and legacy source code.

- `./data`: This folder contains training logs and trained models.

- `./dataset`: Contains all the workload datasets from the Parallel Workload Archive used in this project.

- `./ppo_configs`: This folder contains numerous configurations used in our experiments.

- `./hpc_rl_simulator`: This package contains the implementation of our HPC job scheduling simulator and its OpenAI gym wrappers. Additionally, it includes the customized actor policy networks, callbacks, and additional evaluation methods for training the Stable Baselines3 PPO agent.

- `./result_analysis`: This folder contains the data visualization and result analysis of our experiments.


## Running

### Training the Agent
To train the agent, use the following command:

```bash
python train_ppo.py
```

### Evaluating Priority-Based Scheduling Methods' Performance
To evaluate and export the performance, use:

```bash
python non-RL-score.py
```


### Evaluating Trained Agent Performance
To evaluate and export the trained agent's performance:

```bash
python evaluate_agent.py
```

### Accessing Training Logs
To access the training logs via Tensorboard:

```bash
tensorboard --logdir ./data/logs/*****
```

Note: Replace ***** with the appropriate log directory name.