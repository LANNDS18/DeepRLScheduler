## Fault-Aware, Utility-Based Job Scheduling on Blue Gene/P Systems

### Problem identification

The aim of Batch Job Scheduling Problem can be described as maximizing resource utilization, maximizing job throughput, or minimizing job wait time. In practical, our HPC schedulers make scheduling decisions via heuristic priority functions. The job was scheduled based on its priority.

### Rule-based scheduling algorithm

In traditional utility method, scheduling policies are described as job-scoring functions called utility functions that can be changed on the fly. During each scheduling iteration, each job’s score is evaluated, allowing the scheduler to take the appropriate action.

```
def defined_utililty_func(jobinfo)
    score = [define the utility function]
    TH = [define the fallback threshold]
    return (score, score * TH)
```

Some example scheduler methods based on priority:

1. First Come First Server, FCFS. Priority according to the submission time. Si = qi
2. Shortest Job First, SJF. Priority according to the job duration.
3. UNICEF: Si = qi/(log2(ni) ∗ ti)
4. WFP3 Si = −(wt/rt)3 ∗ n

FAT, WFP3, UNICEF scheduling algorithm based on simple non- linear operation. These algorithms sort the priority to reduce the average waiting time.

### Evaluation Metrics

#### Time (Will be used in RL scheduler)

* Minimize Average waiting time. The job waiting time is the time period between the job’s arrival time and the time of job start.
* Minimize Average response time (RESP). The time period between job’s arrival and successful completion.

### Fault (Will not be used in RL scheduler)

* Job failure rate (JFR). JFR is defined as the ratio between the number of failed jobs and the total number of jobs submitted.
* Service unit loss rate (SULR). SULR is defined as the ratio of wasted service units (i.e., product of job running hours and number of computing nodes) to the entire service units in a given time span.

## RLScheduler: An Automated HPC Batch Job Scheduler Using Reinforcement Learning

### Background

#### Batch Job Scheduling

Job Attributes: On HPC platforms, a job presents several attributes. A more complete list of job attributes can be found in the Standard Workload Format (SWF).


| Name                 | Symbol | Description                                           |
| -------------------- | ------ | ----------------------------------------------------- |
| Job ID               | id_t   | the id of job                                         |
| User ID              | u_t    | the user’s ID                                        |
| Group ID             | g_t    | the group’s ID                                       |
| Executable Id        | app_t  | ID of the job’s executable file                      |
| Submit Time          | s_t    | job submission time                                   |
| Requested Processors | n_t    | the number of processors that a job requests.         |
| Requested Time       | r_t    | job’s runtime estimation (or upper bound) from users |
| Requested Memory     | m_t    | the requested memory per processor                    |

Workloads: In the context of HPC batch job scheduling, workload usually includes a number of batch jobs and the timestamps addressing their submissions. Characterized by the attributes of jobs and their arrival patterns.

It is hard to accurately model a workload, Representative statistical values to characterize workloads.

#### Two Additional Evaluation Metrics

* Minimize the average bounded slowdown (bsld). Here, slowdown means the ratio of job turnaround time over its execution time ((wj + ej )/ej )
* Maximize resource utilization (util), also called utiliza- tion rate, represents the average percentage of compute nodes allocated normalized by the entirety of nodes in the system over a given period of time.

#### Backfilling

Backfilling can be activated to search for the jobs whose resource allocations can be satisfied now without affecting the planned execution for the waiting job, to improve the efficiency of the system.

### RL

Network: takes the waiting jobs and their features as input, outputs a probability distribution of each job being scheduled next. The job with the highest probability (job8 in this example) should be the selected job.

Order Issue: The RL agent should focus less on the original job order, and more on the job's attributes. A new kernel-based DNN architecture to be insensitive to job orders.

High Variance in Sample issue: First, one ‘bad‘ trajectory will diminish what RL agent has learned as we have discussed. Second, too many ‘good’ trajectories will barely teach RL agent anything during training, because no matter what scheduling policy it currently holds, the slowdown is gonna be 1.



## Deep Reinforcement Agent for Scheduling in HPC

DRAS is built on a novel, hierarchical neural network incorporating special HPC scheduling features such as resource reservation and backfilling. A unique training strategy is presented to enable DRAS to rapidly learn the target environment. Once being provided a specific scheduling objective given by system manager, DRAS automatically learns to improve its policy through interaction with the scheduling environment and dynamically adjusts its policy as workload changes. The experiments with different production workloads demonstrate that DRAS outperforms the existing heuristic and optimization approaches by up to 45%. ?
