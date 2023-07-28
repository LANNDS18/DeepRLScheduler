from deepRL_scheduler.hpc_rl_simulator.workload import Workloads

if __name__ == "__main__":
    print("Loading the workloads...")
    load = Workloads()
    load.parse_swf('../../dataset/NASA-iPSC-1993-3.1-cln.swf')
    print("Finish loading the workloads...", type(load[0]))
    print(load.max_nodes, load.max_procs)
    print(load[0].__feature__())
    print(load[1].__feature__())