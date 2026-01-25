"""
Benchmark for cuDF groupby aggregation operations.
Mirrors the groupby_nvbench.cu benchmark for comparison.

Parameters from C++ benchmark:
- Key type: int32_t (4-byte) or int64_t (8-byte)
- num_items: Number of rows in the table
- num_groups: Number of unique groups (keys modulo this value)
- num_aggs: Number of aggregation columns (each using MAX)

Ignored parameters:
- zipf_factor: Always 0.0 (uniform distribution)
- algo: cuDF has single implementation (no algorithm selection)
"""

import pylibcudf as plc
import pyarrow as pa
import cupy as cp
import numpy as np
import timeit
import argparse
from bench_cudf_utils import *

WARMUP_RUNS, BENCHMARK_RUNS = get_bench_params()

bench_cudf_init()

def generate_groupby_keys(
    num_items: int,
    num_groups: int,
    dtype: np.dtype,
    seed: int = 42
) -> plc.Column:
    """
    Generate a key column with uniform distribution.
    Mirrors the C++ generate_groupby_keys with zipf_factor=0.0.
    
    Keys are generated as shuffled values in range [0, num_groups).
    
    Args:
        num_items: Number of rows
        num_groups: Number of unique groups
        dtype: Data type for keys (np.int32 or np.int64)
        seed: Random seed for reproducibility
    
    Returns:
        pylibcudf Column containing key values
    """
    rng = cp.random.default_rng(seed)
    
    if dtype == np.int32:
        pa_type = pa.int32()
        cp_dtype = cp.int32
    else:
        pa_type = pa.int64()
        cp_dtype = cp.int64
    
    # Generate keys: i % num_groups, then shuffle
    # This mirrors the C++ implementation
    keys = cp.arange(num_items, dtype=cp_dtype) % num_groups
    keys = cp.random.permutation(keys)
    
    # Convert to PyArrow via CPU
    keys_cpu = cp.asnumpy(keys)
    pa_array = pa.array(keys_cpu, type=pa_type)
    
    return plc.Column.from_arrow(pa_array)


def generate_groupby_values(
    num_items: int,
    num_aggs: int,
    dtype: np.dtype,
    seed: int = 42
) -> list[plc.Column]:
    """
    Generate value columns for aggregation.
    Mirrors the C++ generate_groupby_vals_seq.
    
    Values are sequential: [0, 1, 2, ..., num_items-1].
    
    Args:
        num_items: Number of rows
        num_aggs: Number of aggregation columns to generate
        dtype: Data type for values (np.int32 or np.int64)
        seed: Random seed (not used, values are sequential)
    
    Returns:
        List of pylibcudf Columns for aggregation
    """
    if dtype == np.int32:
        pa_type = pa.int32()
        cp_dtype = cp.int32
    else:
        pa_type = pa.int64()
        cp_dtype = cp.int64
    
    columns = []
    for i in range(num_aggs):
        # Sequential values: 0, 1, 2, ..., num_items-1
        vals = cp.arange(num_items, dtype=cp_dtype)
        vals_cpu = cp.asnumpy(vals)
        pa_array = pa.array(vals_cpu, type=pa_type)
        columns.append(plc.Column.from_arrow(pa_array))
    
    return columns


def bench_groupby_max(
    key_table: plc.Table, 
    value_columns: list[plc.Column]
) -> float:
    """
    Benchmark groupby with MAX aggregation.
    
    Args:
        key_table: Table containing the key column(s)
        value_columns: List of columns to aggregate
    
    Returns:
        Elapsed time in seconds
    """
    start_time = timeit.default_timer()
    
    # Create GroupBy object
    groupby = plc.groupby.GroupBy(key_table)
    
    # Create aggregation requests for each value column
    # Each column gets MAX aggregation
    requests = []
    for col in value_columns:
        agg = plc.aggregation.max()
        request = plc.groupby.GroupByRequest(col, [agg])
        requests.append(request)
    
    # Perform aggregation
    keys_result, agg_results = groupby.aggregate(requests)
    
    # Ensure GPU work is complete
    cp.cuda.Device().synchronize()
    end_time = timeit.default_timer()
    
    return end_time - start_time


def run_benchmark(
    key_table: plc.Table, 
    value_columns: list[plc.Column]
) -> tuple[float, float, float, list[float]]:
    """
    Run benchmark for groupby aggregation.
    
    Args:
        key_table: Table containing the key column(s)
        value_columns: List of columns to aggregate
    
    Returns:
        Tuple of (average_time, min_time, max_time, list of individual run times)
    """
    results = []
    
    # Warmup runs
    for _ in range(WARMUP_RUNS):
        _ = bench_groupby_max(key_table, value_columns)
    
    # Benchmark runs
    for _ in range(BENCHMARK_RUNS):
        elapsed_time = bench_groupby_max(key_table, value_columns)
        results.append(elapsed_time)
    
    avg_time = sum(results) / len(results)
    min_time = min(results)
    max_time = max(results)
    print(f"Min Time: {min_time*1000} ms, Max Time: {max_time*1000} ms, Avg Time: {avg_time*1000} ms")
    
    return avg_time, min_time, max_time, results


def benchmark_groupby_max(
    results: list, 
    json_results: list,
    key_types: list[str], 
    num_items_list: list[int],
    num_groups_list: list[int], 
    num_aggs_list: list[int]
):
    """
    Benchmark: groupby_max with varying parameters.
    Mirrors bench_groupby_max from C++ benchmark.
    """
    print("\n=== Benchmark: groupby_max ===")
    
    for key_type in key_types:
        dtype = np.int32 if key_type == "int32" else np.int64
        
        for num_items in num_items_list:
            for num_groups in num_groups_list:
                # Skip if num_groups > num_items
                if num_groups > num_items:
                    continue
                
                for num_aggs in num_aggs_list:
                    print(f"  KeyType={key_type}, num_items={num_items}, "
                          f"num_groups={num_groups}, num_aggs={num_aggs}")
                    
                    try:
                        # Generate keys
                        key_col = generate_groupby_keys(num_items, num_groups, dtype, seed=42)
                        key_table = plc.Table([key_col])
                        
                        # Generate value columns
                        value_columns = generate_groupby_values(num_items, num_aggs, dtype, seed=42)
                        
                        # Run benchmark
                        avg_time, min_time, max_time, measurements = run_benchmark(key_table, value_columns)
                        
                        # Throughput: elements per second
                        throughput = num_items / avg_time
                        
                        print(f"    Avg: {avg_time*1000:.3f} ms, "
                              f"Throughput: {throughput/1e6:.3f} MElements/s")
                        
                        results.append(
                            f"cudf,groupby_max,{key_type},{num_items},"
                            f"{num_groups},{num_aggs},{avg_time},{min_time},{max_time},{throughput}"
                        )
                        json_results.append({
                            "impl": "cudf",
                            "benchmark": "groupby_max",
                            "key_type": key_type,
                            "num_items": num_items,
                            "num_groups": num_groups,
                            "num_aggs": num_aggs,
                            "avg_time": avg_time,
                            "min_time": min_time,
                            "max_time": max_time,
                            "throughput": throughput,
                            "measurements": measurements
                        })
                    except Exception as e:
                        print(f"    ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark cuDF groupby aggregation")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmark with reduced parameters")
    args = parser.parse_args()
    
    all_results = []
    json_results = []
    
    # Parameter configurations matching C++ benchmark
    if args.quick:
        # Quick mode: fewer configurations for testing
        key_types = ["int32"]
        # num_items: 2^24, 2^26
        num_items_list = [2**28]
        # num_groups: 2^4, 2^12, 2^20
        num_groups_list = [2**2, 2**4, 2**6]
        num_aggs_list = [2]
    else:
        # Full mode: match C++ benchmark parameters
        key_types = ["int32", "int64"]
        # num_items: 2^28, 2^29 (from C++ benchmark)
        num_items_list = [2**28]
        # num_groups: 2^2 to 2^26, step 2
        num_groups_list = [2**i for i in range(2, 27, 2)]
        # num_aggs: 1, 2, 4
        num_aggs_list = [1, 2, 4]
    
    # Run benchmark
    benchmark_groupby_max(all_results, json_results, key_types, num_items_list, 
                          num_groups_list, num_aggs_list)
    
    # Write results to CSV
    header = "impl,benchmark,key_type,num_items,num_groups,num_aggs,avg_time,min_time,max_time,throughput"
    write_results_to_csv(all_results, "cudf_groupby_max", header)
    write_results_to_json(json_results, "cudf_groupby_max")
    
    print(f"\nBenchmark complete. {len(all_results)} configurations tested.")
    print("Results written to CSV.")


if __name__ == "__main__":
    main()
