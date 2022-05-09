import nvsmi
import psutil
from tabulate import tabulate
import os

def get_username(pid):
    try:
        process = psutil.Process(pid)
        return process.username()
    except psutil.NoSuchProcess:
        return 'NoSuchProcess'


def get_free_gpus(ignore_my_pid = True):
    used_gpu_ids = {p.gpu_id for p in nvsmi.get_gpu_processes() if not ignore_my_pid or p.pid != os.getpid()}

    free_gpus = [g.id for g in nvsmi.get_gpus() if g.id not in used_gpu_ids]

    return free_gpus


def print_gpu_usages():
    t = [(g.gpu_id, g.pid, get_username(g.pid), f"{g.used_memory} MiB") for g in nvsmi.get_gpu_processes()]
    print(tabulate(t,headers=['GPU ID', "PID", "username", 'used memory']))

if __name__ == "__main__":
    print_gpu_usages()

