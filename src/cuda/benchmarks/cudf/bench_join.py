"""
Benchmark for cuDF inner join operations.
Mirrors the ieq_join_nvbench.cu benchmark for comparison.

Benchmark Suites (matching C++ benchmark):
1. vary_left_table_size - Varies left table size with fixed right table size (2^28)
2. vary_right_table_size - Varies right table size across different left table sizes

Parameters from C++ benchmark:
- Key type: int32_t (4-byte) or int64_t (8-byte)
- left_num_items: Number of rows in the left (primary key) table
- right_num_items: Number of rows in the right (foreign key) table
- left_ncols, right_ncols: Total number of columns (including key column)

Ignored parameters (cuDF has only one join implementation):
- join_algo, mat_algo, match_ratio, zipf_factor
"""

import pylibcudf as plc
import pyarrow as pa
import cupy as cp
import numpy as np
import timeit
import argparse
from bench_cudf_utils import *

bench_cudf_init()

WARMUP_RUNS, BENCHMARK_RUNS = get_bench_params()


def generate_pk_table(
    num_rows: int,
    num_cols: int,
    dtype: np.dtype,
    seed: int = 42
) -> plc.Table:
    """
    Generate a primary key table with unique keys in the first column.
    Mirrors gamax::datagen::device::pk_uniform_id.
    
    Args:
        num_rows: Number of rows (also number of unique keys)
        num_cols: Total number of columns (including key column)
        dtype: Data type for columns (np.int32 or np.int64)
        seed: Random seed for reproducibility
    
    Returns:
        pylibcudf Table with unique keys in first column
    """
    rng = cp.random.default_rng(seed)
    columns = []
    
    if dtype == np.int32:
        pa_type = pa.int32()
        cp_dtype = cp.int32
    else:
        pa_type = pa.int64()
        cp_dtype = cp.int64
    
    # First column: unique primary keys (0 to num_rows-1, shuffled)
    pk_data = cp.arange(num_rows, dtype=cp_dtype)
    pk_data = cp.random.permutation(pk_data)
    pk_cpu = cp.asnumpy(pk_data)
    pk_array = pa.array(pk_cpu, type=pa_type)
    columns.append(plc.Column.from_arrow(pk_array))
    
    # Remaining columns: random payload values
    for i in range(1, num_cols):
        col_rng = cp.random.default_rng(seed + i)
        payload_data = col_rng.integers(0, num_rows, size=num_rows, dtype=cp_dtype)
        payload_cpu = cp.asnumpy(payload_data)
        payload_array = pa.array(payload_cpu, type=pa_type)
        columns.append(plc.Column.from_arrow(payload_array))
    
    return plc.Table(columns)


def generate_fk_table(
    num_unique_keys: int,
    num_rows: int,
    num_cols: int,
    dtype: np.dtype,
    seed: int = 42
) -> plc.Table:
    """
    Generate a foreign key table with keys referencing primary keys.
    Mirrors gamax::datagen::device::fk_uniform_id with match_ratio=1.0.
    
    Args:
        num_unique_keys: Number of unique keys to reference (from PK table)
        num_rows: Number of rows in this table
        num_cols: Total number of columns (including key column)
        dtype: Data type for columns (np.int32 or np.int64)
        seed: Random seed for reproducibility
    
    Returns:
        pylibcudf Table with foreign keys in first column
    """
    rng = cp.random.default_rng(seed)
    columns = []
    
    if dtype == np.int32:
        pa_type = pa.int32()
        cp_dtype = cp.int32
    else:
        pa_type = pa.int64()
        cp_dtype = cp.int64
    
    # First column: foreign keys (random values from 0 to num_unique_keys-1)
    fk_data = rng.integers(0, num_unique_keys, size=num_rows, dtype=cp_dtype)
    fk_cpu = cp.asnumpy(fk_data)
    fk_array = pa.array(fk_cpu, type=pa_type)
    columns.append(plc.Column.from_arrow(fk_array))
    
    # Remaining columns: random payload values
    for i in range(1, num_cols):
        col_rng = cp.random.default_rng(seed + i)
        payload_data = col_rng.integers(0, num_rows, size=num_rows, dtype=cp_dtype)
        payload_cpu = cp.asnumpy(payload_data)
        payload_array = pa.array(payload_cpu, type=pa_type)
        columns.append(plc.Column.from_arrow(payload_array))
    
    return plc.Table(columns)


def bench_inner_join(left_table: plc.Table, right_table: plc.Table,
                     left_keys: plc.Table, right_keys: plc.Table) -> float:
    """
    Benchmark inner join between two tables with materialization.
    
    The cuDF inner_join takes key tables and returns row indices.
    We then use gather to materialize the full result tables.
    
    Args:
        left_table: Full left table (for materialization)
        right_table: Full right table (for materialization)
        left_keys: Left table (key columns only for join)
        right_keys: Right table (key columns only for join)
    
    Returns:
        Elapsed time in seconds
    """
    start_time = timeit.default_timer()
    
    # Step 1: Perform the join to get indices
    left_indices, right_indices = plc.join.inner_join(
        left_keys,
        right_keys,
        plc.types.NullEquality.EQUAL
    )
    
    # Step 2: Materialize the result using gather
    left_result = plc.copying.gather(
        left_table,
        left_indices,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK
    )
    right_result = plc.copying.gather(
        right_table,
        right_indices,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK
    )
    
    # Ensure GPU work is complete
    cp.cuda.Device().synchronize()
    end_time = timeit.default_timer()
    
    return end_time - start_time


def run_benchmark(left_table: plc.Table, right_table: plc.Table,
                  left_keys: plc.Table, right_keys: plc.Table) -> tuple[float, float, float, list[float]]:
    """
    Run benchmark for inner join with materialization.
    
    Args:
        left_table: Full left table
        right_table: Full right table
        left_keys: Left key table
        right_keys: Right key table
    
    Returns:
        Tuple of (average_time, min_time, max_time, list of individual run times)
    """
    results = []
    
    # Warmup runs
    for _ in range(WARMUP_RUNS):
        _ = bench_inner_join(left_table, right_table, left_keys, right_keys)
    
    # Benchmark runs
    for _ in range(BENCHMARK_RUNS):
        elapsed_time = bench_inner_join(left_table, right_table, left_keys, right_keys)
        results.append(elapsed_time)
    
    avg_time = sum(results) / len(results)
    min_time = min(results)
    max_time = max(results)
    
    return avg_time, min_time, max_time, results


def run_single_config(
    key_type: str,
    left_num_items: int,
    right_num_items: int,
    left_ncols: int,
    right_ncols: int
) -> dict | None:
    """
    Run a single benchmark configuration and return results.
    
    Args:
        key_type: "int32" or "int64"
        left_num_items: Number of rows in left table
        right_num_items: Number of rows in right table
        left_ncols: Number of columns in left table
        right_ncols: Number of columns in right table
    
    Returns:
        Dictionary with benchmark results or None on error
    """
    dtype = np.int32 if key_type == "int32" else np.int64
    
    # Skip if left > right (matches C++ behavior)
    if left_num_items > right_num_items:
        return None
    
    try:
        # Generate tables
        left_table = generate_pk_table(left_num_items, left_ncols, dtype, seed=42)
        right_table = generate_fk_table(
            left_num_items, right_num_items, right_ncols, dtype, seed=39
        )
        
        # Extract key columns
        left_keys = plc.Table([left_table.columns()[0]])
        right_keys = plc.Table([right_table.columns()[0]])
        
        # Run benchmark
        avg_time, min_time, max_time, measurements = run_benchmark(
            left_table, right_table, left_keys, right_keys
        )
        
        # Throughput: total elements processed
        total_elements = left_num_items + right_num_items
        throughput = total_elements / avg_time
        
        return {
            "key_type": key_type,
            "left_num_items": left_num_items,
            "right_num_items": right_num_items,
            "left_ncols": left_ncols,
            "right_ncols": right_ncols,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "throughput": throughput,
            "measurements": measurements
        }
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def format_result(result: dict) -> str:
    """Format a result dictionary as a CSV line."""
    return (
        f"cudf,inner_join,{result['key_type']},"
        f"{result['left_num_items']},{result['right_num_items']},"
        f"{result['left_ncols']},{result['right_ncols']},"
        f"{result['avg_time']},{result['min_time']},"
        f"{result['max_time']},{result['throughput']}"
    )


def benchmark_vary_left_table_size(
    key_types: list[str],
    left_num_items_list: list[int],
    right_num_items: int,
    ncols_list: list[int]
) -> tuple[list[str], list[dict]]:
    """
    Benchmark suite: vary_left_table_size
    Varies left table size with fixed right table size.
    Mirrors the C++ benchmark suite.
    
    C++ params:
    - left_num_items: 2^10 to 2^28, step 1
    - right_num_items: 2^28 (fixed)
    - left_ncols, right_ncols: [2, 4]
    """
    print("\n=== Benchmark Suite: vary_left_table_size ===")
    csv_results = []
    json_results = []
    
    for key_type in key_types:
        for left_num_items in left_num_items_list:
            for left_ncols in ncols_list:
                for right_ncols in ncols_list:
                    print(f"  KeyType={key_type}, left={left_num_items}, "
                          f"right={right_num_items}, left_ncols={left_ncols}, "
                          f"right_ncols={right_ncols}")
                    
                    result = run_single_config(
                        key_type, left_num_items, right_num_items,
                        left_ncols, right_ncols
                    )
                    
                    if result:
                        print(f"    Avg: {result['avg_time']*1000:.3f} ms, "
                              f"Throughput: {result['throughput']/1e6:.3f} MElements/s")
                        csv_results.append(format_result(result))
                        json_results.append(result)
    
    return csv_results, json_results


def benchmark_vary_right_table_size(
    key_types: list[str],
    left_num_items_list: list[int],
    right_num_items_list: list[int],
    ncols_list: list[int]
) -> tuple[list[str], list[dict]]:
    """
    Benchmark suite: vary_right_table_size
    Varies right table size across different left table sizes.
    Mirrors the C++ benchmark suite.
    
    C++ params:
    - left_num_items: 2^12, 2^16, 2^20, 2^24 (step 4)
    - right_num_items: 2^12 to 2^28, step 1
    - left_ncols, right_ncols: [2, 4]
    """
    print("\n=== Benchmark Suite: vary_right_table_size ===")
    csv_results = []
    json_results = []
    
    for key_type in key_types:
        for left_num_items in left_num_items_list:
            for right_num_items in right_num_items_list:
                for left_ncols in ncols_list:
                    for right_ncols in ncols_list:
                        print(f"  KeyType={key_type}, left={left_num_items}, "
                              f"right={right_num_items}, left_ncols={left_ncols}, "
                              f"right_ncols={right_ncols}")
                        
                        result = run_single_config(
                            key_type, left_num_items, right_num_items,
                            left_ncols, right_ncols
                        )
                        
                        if result:
                            print(f"    Avg: {result['avg_time']*1000:.3f} ms, "
                                  f"Throughput: {result['throughput']/1e6:.3f} MElements/s")
                            csv_results.append(format_result(result))
                            json_results.append(result)
    
    return csv_results, json_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark cuDF inner join")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmark with reduced parameters")
    parser.add_argument("--suite", type=str, default="all",
                        choices=["all", "vary_left", "vary_right"],
                        help="Which benchmark suite to run")
    args = parser.parse_args()
    
    # CSV header for all suites
    header = ("impl,benchmark,key_type,left_num_items,right_num_items,"
              "left_ncols,right_ncols,"
              "avg_time,min_time,max_time,throughput")
    
    # Parameter configurations matching C++ benchmark
    if args.quick:
        # Quick mode: fewer configurations for testing
        key_types = ["int32"]
        # vary_left_table_size
        left_items_vary_left = [2**16, 2**20, 2**24]
        right_items_fixed = 2**24
        # vary_right_table_size
        left_items_vary_right = [2**16, 2**20]
        right_items_list = [2**16, 2**20, 2**24]
        # ncols
        ncols_list = [2, 4]
    else:
        # Full mode: match C++ benchmark parameters
        key_types = ["int32", "int64"]
        # vary_left_table_size: left from 2^10 to 2^28, right fixed at 2^28
        left_items_vary_left = [2**i for i in range(10, 29, 1)]
        right_items_fixed = 2**28
        # vary_right_table_size: left at 2^12/16/20/24, right from 2^12 to 2^28
        left_items_vary_right = [2**i for i in range(12, 25, 4)]
        right_items_list = [2**i for i in range(12, 29, 1)]
        # ncols: 2 and 4
        ncols_list = [2, 4]
    
    # Run selected benchmark suites
    if args.suite in ["all", "vary_left"]:
        csv_results, json_results = benchmark_vary_left_table_size(
            key_types, left_items_vary_left, right_items_fixed, ncols_list
        )
        write_results_to_csv(csv_results, "cudf_join_vary_left_table_size", header)
        write_results_to_json(json_results, "cudf_join_vary_left_table_size")
        print(f"  Suite 'vary_left_table_size': {len(csv_results)} configurations")
    
    if args.suite in ["all", "vary_right"]:
        csv_results, json_results = benchmark_vary_right_table_size(
            key_types, left_items_vary_right, right_items_list, ncols_list
        )
        write_results_to_csv(csv_results, "cudf_join_vary_right_table_size", header)
        write_results_to_json(json_results, "cudf_join_vary_right_table_size")
        print(f"  Suite 'vary_right_table_size': {len(csv_results)} configurations")
    
    print("\nBenchmark complete. Results written to CSV files.")


if __name__ == "__main__":
    main()
