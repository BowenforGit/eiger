import socket
from datetime import datetime
import os
import json

import rmm.mr
from rmm.allocators.cupy import rmm_cupy_allocator
import cupy as cp

def write_results_to_csv(results, basename, header):
    # hostname = socket.gethostname().split('.')[0]
    today = datetime.now().strftime('%Y%m%d')
    filename = os.path.join(os.getenv("OUTPUT_DIR", "."), f"{basename}_{today}.csv")
    with open(filename, "w") as f:
        f.write(header + "\n")
        for line in results:
            f.write(line + "\n")

def write_results_to_json(results_list, basename):
    today = datetime.now().strftime('%Y%m%d')
    filename = os.path.join(os.getenv("OUTPUT_DIR", "."), f"{basename}_{today}.json")
    with open(filename, "w") as f:
        json.dump(results_list, f, indent=4)

def get_bench_params():
    warmup = int(os.getenv("WARMUP_RUNS", 5))
    runs = int(os.getenv("BENCHMARK_RUNS", 10))
    return warmup, runs

def bench_cudf_init():
    current_mr = rmm.mr.get_current_device_resource()
    free_memory, _ = rmm.mr.available_device_memory()
    free_memory = int(round(float(free_memory) * 0.95 / 256) * 256)
    
    if isinstance(current_mr, rmm.mr.CudaMemoryResource):
        print("Current RMM memory resource is CudaMemoryResource")
    else:
        raise RuntimeError("Current RMM memory resource is not CudaMemoryResource")
    
    print(f"Setting PoolMemoryResource with pool size: {free_memory} bytes")
    new_mr = rmm.mr.PoolMemoryResource(
        current_mr, 
        initial_pool_size=free_memory)
    rmm.mr.set_current_device_resource(new_mr)
    cp.cuda.set_allocator(rmm_cupy_allocator)