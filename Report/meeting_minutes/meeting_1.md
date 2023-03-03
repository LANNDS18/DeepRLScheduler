# Meeting Minutes
## Meeting Information
**Meeting Date/Time:** 
24/02/23, 30min

**Meeting Purpose:** 
To start with the project, discuss the prerequisites for the project.


**Meeting Location:** EPCC meeting room

## Attendees
People who attended:
- s2311726@ed.ac.uk
- C.Laoide-Kemp@epcc.ed.ac.uk

## Discussed Items
- Prepare for Literature Review: writing: It can be a chapter in the dissertation to identify the background and improve the understanding of the project.
- Indentify the Resource Allocation: Make a request to EPCC, determine a estimated CPU And GPU hours, identify the software dependencies.
- Make a Project Plan in Advance: The project plan should be made in advance, and the project should be divided into several stages, and the time should be estimated for each stage. Draw a Gantt chart to show the progress of the project.

## Action Items
1. By testing the training and evaluation speed on login node, the CPUs are relatively slow to use, therefore GPUs are needed. QoS for GPU: long, 5 jobs, 14 days, 8 GPUs
2. GARLSched: Generative adversarial deep reinforcement learning task scheduling optimization for large-scale high performance computing systems: The network size is very small, sometimes it can be easy to train on CPU
3. RLScheduler: We timed both computations on our evaluation platform (Intel Xeon Silver 4109T CPU and 32GB DDR4 DRAM) and presented the results in Table IX. In summary, the trained RLScheduler DNN can make a decision for 128 pending jobs in 0.3ms, compared to SJF sorting the same 128 jobs in 0.7ms 2. The decision making of RLScheduler is comparably fast. In addition, such a time cost will not grow even when more jobs are pending in the system as more jobs will first be cut-off to MAX_OBSV_SIZE (i.e., 128). During RLScheduler training, one epoch takes around 123 seconds and it typically takes less than 100 epochs to converge. Specifically, it took 1.1h to converge our training on Lublin- 1 job trace. The computation will be much faster on GPU