# INFR11164: Dissertation Project
## Graph Neural Network-based Deep Reinforcement Learning for Tasks Scheduling Problem in HPC System
### Wuwei Zhang
### s2311726

## Temporary Project Proposal: 
The allocation of the submitted task across heterogeneous computing nodes, which strives to in- crease the number of tasks in the same time interval in the HPC system, is a crucial issue[1]. At present, related works for deep reinforcement learning (DRL) show success in Job Shop Schedul- ing Problems (JSSP) and other control flow problems [2][3]. However, the current DRL agent requires a significantly large amount of time and data to train, thus, it is important to verify the efficiency gained from the agent can offset the training and running costs.

In this project, we want to represent the state of HPC task scheduling using a disjunctive graph. Then we are interested in integrating Graph Neural Network(GNN) into DRL agents and design- ing an effective reward-action space for Task Scheduling in the HPC system because GNN enables agents to enable better generalization, and reasoning ability in graph-structured data [4]. We aim to apply GNN in DRL agents to learn node features that encode the spatial structure of the Task Scheduling Problem represented as a graph. The computational cost will also be evaluated to verify the efficiency optimized from the DRL + GNN.