import numpy as np


def f1_score(job):
    submit_time = job.submit_time
    request_processors = job.request_number_of_processors
    request_time = job.request_time
    # run_time = job.run_time
    return (np.log10(request_time if request_time > 0 else 0.1) * request_processors + 870 * np.log10(
        submit_time if submit_time > 0 else 0.1))


def f2_score(job):
    submit_time = job.submit_time
    request_processors = job.request_number_of_processors
    request_time = job.request_time
    # run_time = job.run_time
    # f2: r^(1/2)*n + 25600 * log10(s)
    return np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time)


def sjf_score(job):
    # run_time = job.run_time
    request_time = job.request_time
    submit_time = job.submit_time
    # if request_time is the same, pick whichever submitted earlier
    return request_time, submit_time


def smallest_score(job):
    request_processors = job.request_number_of_processors
    submit_time = job.submit_time
    # if request_time is the same, pick whichever submitted earlier
    return request_processors, submit_time


def fcfs_score(job):
    submit_time = job.submit_time
    return submit_time


def average_bounded_slowdown(job):
    runtime = max(job.run_time, 10)
    return max(1.0, (job.scheduled_time - job.submit_time + job.run_time) / runtime)


def average_waiting_time(job):
    return job.scheduled_time - job.submit_time


def average_turnaround_time(job):
    return job.scheduled_time - job.submit_time + job.run_time


def resource_utilization(job):
    return -job.run_time * job.request_number_of_processors


def average_slowdown(job):
    return (job.scheduled_time - job.submit_time + job.run_time) / job.run_time
