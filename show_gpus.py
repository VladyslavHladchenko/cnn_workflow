
import nvsmi
import psutil
import sys
import os
from tabulate import tabulate
from copy import deepcopy

my_username = os.getlogin()

def get_username(pid):
    try:
        process = psutil.Process(pid)
        return process.username()
    except psutil.NoSuchProcess:
        return 'NoSuchProcess'


def get_free_gpus():
    return [g.gpu_id for g in nvsmi.get_available_gpus(gpu_util_max=0.0)]  #TODO: wrong  


def print_gpu_usages():
    gpu_id_to_usage_map = {gpu.id: gpu.gpu_util for gpu in nvsmi.get_gpus()}
    
    t = [(g.gpu_id,
          g.pid,
          get_username(g.pid),
          f"{g.used_memory} MiB",
          f"{gpu_id_to_usage_map[g.gpu_id]} %")
          for g in nvsmi.get_gpu_processes()]
    
    filtered = deepcopy(t)
    if 'my' in sys.argv:
        my_gpus = [data[0] for data in t if data[2] == my_username]
        filtered = [data for data in t if data[0] in my_gpus]
    
    
    print(tabulate(filtered, headers=['GPU ID', "PID", "username", 'used memory', 'usage']))


if __name__ == "__main__":
    print_gpu_usages()
