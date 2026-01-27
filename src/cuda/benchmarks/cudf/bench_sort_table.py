"""
Benchmark for cuDF table sorting operations.
Mirrors the sort_table_nvbench.cu benchmark for comparison.

Benchmarks:
1. sort_table_numeric - Numeric columns only, varying row counts and column counts
2. sort_table_string_key - String key column with varying string lengths
3. sort_table_string_payload - Numeric key with string payload columns
"""

import pylibcudf as plc
import pyarrow as pa
import cupy as cp
import numpy as np
import timeit
import argparse
from typing import Optional
from bench_cudf_utils import *

bench_cudf_init()

WARMUP_RUNS, BENCHMARK_RUNS = get_bench_params()


def generate_random_numeric_column(
    num_rows: int, 
    dtype: np.dtype, 
    seed: int = 42
) -> plc.Column:
    """
    Generate a pylibcudf Column containing random numeric values using CuPy.
    
    Args:
        num_rows: Number of rows to generate
        dtype: NumPy dtype (np.int32 or np.int64)
        seed: Random seed for reproducibility
    
    Returns:
        pylibcudf Column containing random values
    """
    rng = cp.random.default_rng(seed)
    
    if dtype == np.int32:
        data = rng.integers(0, num_rows, size=num_rows, dtype=cp.int32)
        pa_type = pa.int32()
    else:  # np.int64
        data = rng.integers(0, num_rows, size=num_rows, dtype=cp.int64)
        pa_type = pa.int64()
    
    # Convert to PyArrow via CPU (unavoidable for pylibcudf)
    data_cpu = cp.asnumpy(data)
    pa_array = pa.array(data_cpu, type=pa_type)
    
    return plc.Column.from_arrow(pa_array)


def generate_random_strings_column(
    num_strings: int, 
    string_length: int, 
    seed: int = 42
) -> plc.Column:
    """
    Generate a pylibcudf Column containing random strings using CuPy.
    
    Mirrors the C++ generate_random_strings implementation:
    - Generates lowercase letters 'a' to 'z' (26 characters)
    - Each character is independently random
    
    Args:
        num_strings: Number of strings to generate
        string_length: Length of each string
        seed: Random seed for reproducibility
    
    Returns:
        pylibcudf Column containing the generated strings
    """
    rng = cp.random.default_rng(seed)
    
    # Generate random character codes (lowercase letters 'a' to 'z': ASCII 97-122)
    char_codes = rng.integers(0, 26, size=(num_strings, string_length), dtype=cp.uint8)
    char_codes += 97  # Convert to ASCII 'a'-'z'
    
    # Convert to bytes on CPU
    char_codes_cpu = cp.asnumpy(char_codes)
    
    # Convert to list of strings
    strings = [row.tobytes().decode('ascii') for row in char_codes_cpu]
    
    # Create PyArrow array
    pa_array = pa.array(strings, type=pa.string())
    
    return plc.Column.from_arrow(pa_array)


def generate_benchmark_table(
    num_rows: int,
    num_key_cols: int,
    num_nonkey_cols: int,
    key_dtype: np.dtype,
    has_string_key: bool = False,
    has_string_nonkey: bool = False,
    string_length: int = 16,
    base_seed: int = 42
) -> plc.Table:
    """
    Generate a table with numeric and/or string columns for benchmarking.
    
    Mirrors the C++ generate_benchmark_table function.
    
    Args:
        num_rows: Number of rows in the table
        num_key_cols: Number of key columns
        num_nonkey_cols: Number of non-key (payload) columns
        key_dtype: Data type for numeric columns (np.int32 or np.int64)
        has_string_key: Whether the first key column should be a string
        has_string_nonkey: Whether the first non-key column should be a string
        string_length: Length of strings if any string columns
        base_seed: Base random seed
    
    Returns:
        pylibcudf Table with the specified schema
    """
    columns = []
    
    # Generate key columns
    for i in range(num_key_cols):
        if i == 0 and has_string_key:
            col = generate_random_strings_column(num_rows, string_length, base_seed + i)
        else:
            col = generate_random_numeric_column(num_rows, key_dtype, base_seed + i)
        columns.append(col)
    
    # Generate non-key (payload) columns
    for i in range(num_nonkey_cols):
        if i == 0 and has_string_nonkey:
            col = generate_random_strings_column(num_rows, string_length, base_seed + 100 + i)
        else:
            col = generate_random_numeric_column(num_rows, key_dtype, base_seed + 100 + i)
        columns.append(col)
    
    return plc.Table(columns)


def bench_sort_table(plc_table: plc.Table, num_key_cols: int) -> float:
    """
    Benchmark sorting a table by key columns using sort_by_key.
    
    Args:
        plc_table: The pylibcudf Table to sort (key columns followed by payload columns)
        num_key_cols: Number of key columns (first num_key_cols columns are keys)
    
    Returns:
        Elapsed time in seconds
    """
    # Split table into keys and values
    all_columns = plc_table.columns()
    key_columns = list(all_columns[:num_key_cols])
    value_columns = list(all_columns[num_key_cols:])
    
    keys_table = plc.Table(key_columns)
    values_table = plc.Table(value_columns)
    
    # Sort order applies only to key columns
    column_order = [plc.types.Order.ASCENDING] * num_key_cols
    null_precedence = [plc.types.NullOrder.AFTER] * num_key_cols
    
    start_time = timeit.default_timer()
    # TODO: this does not sort the keys, fix this
    result = plc.sorting.sort_by_key(values_table, keys_table, column_order, null_precedence)
    # Ensure GPU work is complete
    cp.cuda.Device().synchronize()
    end_time = timeit.default_timer()

    # print(ColumnBase.from_pylibcudf(plc.sorting.sorted_order(keys_table, column_order, null_precedence)))
    
    return end_time - start_time


def run_benchmark(plc_table: plc.Table, num_key_cols: int) -> tuple[float, float, float, list[float]]:
    """
    Run benchmark for table sorting.
    
    Args:
        plc_table: The pylibcudf Table to sort
        num_key_cols: Number of key columns
    
    Returns:
        Tuple of (average_time, min_time, max_time, list of individual run times)
    """
    results = []
    
    # Warmup runs
    for _ in range(WARMUP_RUNS):
        _ = bench_sort_table(plc_table, num_key_cols)
    
    # Benchmark runs
    for _ in range(BENCHMARK_RUNS):
        elapsed_time = bench_sort_table(plc_table, num_key_cols)
        results.append(elapsed_time)
    
    avg_time = sum(results) / len(results)
    min_time = min(results)
    max_time = max(results)
    
    return avg_time, min_time, max_time, results


def benchmark_numeric(results: list, json_results: list, key_types: list[str], row_counts: list[int], 
                      num_key_cols_list: list[int], num_nonkey_cols_list: list[int]):
    """
    Benchmark: sort_table with numeric columns only.
    Mirrors sort_table_numeric from C++ benchmark.
    """
    print("\n=== Benchmark: sort_table_numeric ===")
    
    for key_type in key_types:
        dtype = np.int32 if key_type == "int32" else np.int64
        
        for num_rows in row_counts:
            for num_key_cols in num_key_cols_list:
                for num_nonkey_cols in num_nonkey_cols_list:
                    print(f"  KeyType={key_type}, num_rows={num_rows}, "
                          f"num_key_cols={num_key_cols}, num_nonkey_cols={num_nonkey_cols}")
                    
                    try:
                        # Generate table
                        table = generate_benchmark_table(
                            num_rows=num_rows,
                            num_key_cols=num_key_cols,
                            num_nonkey_cols=num_nonkey_cols,
                            key_dtype=dtype,
                            has_string_key=False,
                            has_string_nonkey=False
                        )
                        
                        # Run benchmark
                        avg_time, min_time, max_time, measurements = run_benchmark(table, num_key_cols)
                        throughput = num_rows / avg_time
                        
                        print(f"    Avg: {avg_time*1000:.3f} ms, "
                              f"Throughput: {throughput/1e6:.3f} MRows/s")
                        
                        results.append(
                            f"cudf,sort_table_numeric,{key_type},{num_rows},"
                            f"{num_key_cols},{num_nonkey_cols},0,{avg_time},{min_time},{max_time},{throughput}"
                        )
                        json_results.append({
                            "impl": "cudf",
                            "benchmark": "sort_table_numeric",
                            "key_type": key_type,
                            "num_rows": num_rows,
                            "num_key_cols": num_key_cols,
                            "num_nonkey_cols": num_nonkey_cols,
                            "string_length": 0,
                            "avg_time": avg_time,
                            "min_time": min_time,
                            "max_time": max_time,
                            "throughput": throughput,
                            "measurements": measurements
                        })
                    except Exception as e:
                        print(f"    ERROR: {e}")


def benchmark_string_key(results: list, json_results: list, row_counts: list[int], 
                         num_key_cols_list: list[int], num_nonkey_cols_list: list[int], 
                         string_lengths: list[int]):
    """
    Benchmark: sort_table with string key column.
    Mirrors sort_table_string_key from C++ benchmark.
    """
    print("\n=== Benchmark: sort_table_string_key ===")
    
    for num_rows in row_counts:
        for num_key_cols in num_key_cols_list:
            for num_nonkey_cols in num_nonkey_cols_list:
                for string_length in string_lengths:
                    print(f"  num_rows={num_rows}, num_key_cols={num_key_cols}, "
                          f"num_nonkey_cols={num_nonkey_cols}, string_length={string_length}")
                    
                    try:
                        # Generate table with string key
                        table = generate_benchmark_table(
                            num_rows=num_rows,
                            num_key_cols=num_key_cols,
                            num_nonkey_cols=num_nonkey_cols,
                            key_dtype=np.int32,  # Payload type
                            has_string_key=True,
                            has_string_nonkey=False,
                            string_length=string_length
                        )
                        
                        # Run benchmark
                        avg_time, min_time, max_time, measurements = run_benchmark(table, num_key_cols)
                        throughput = num_rows / avg_time
                        
                        print(f"    Avg: {avg_time*1000:.3f} ms, "
                              f"Throughput: {throughput/1e6:.3f} MRows/s")
                        
                        results.append(
                            f"cudf,sort_table_string_key,int32,{num_rows},"
                            f"{num_key_cols},{num_nonkey_cols},{string_length},{avg_time},{min_time},{max_time},{throughput}"
                        )
                        json_results.append({
                            "impl": "cudf",
                            "benchmark": "sort_table_string_key",
                            "key_type": "int32",
                            "num_rows": num_rows,
                            "num_key_cols": num_key_cols,
                            "num_nonkey_cols": num_nonkey_cols,
                            "string_length": string_length,
                            "avg_time": avg_time,
                            "min_time": min_time,
                            "max_time": max_time,
                            "throughput": throughput,
                            "measurements": measurements
                        })
                    except Exception as e:
                        print(f"    ERROR: {e}")


def benchmark_string_payload(results: list, json_results: list, key_types: list[str], row_counts: list[int],
                             num_key_cols_list: list[int], num_nonkey_cols_list: list[int], 
                             string_lengths: list[int]):
    """
    Benchmark: sort_table with string payload columns.
    Mirrors sort_table_string_payload from C++ benchmark.
    """
    print("\n=== Benchmark: sort_table_string_payload ===")
    
    for key_type in key_types:
        dtype = np.int32 if key_type == "int32" else np.int64
        
        for num_rows in row_counts:
            for num_key_cols in num_key_cols_list:
                for num_nonkey_cols in num_nonkey_cols_list:
                    for string_length in string_lengths:
                        print(f"  KeyType={key_type}, num_rows={num_rows}, "
                              f"num_key_cols={num_key_cols}, num_nonkey_cols={num_nonkey_cols}, "
                              f"string_length={string_length}")
                        
                        try:
                            # Generate table with string payload
                            table = generate_benchmark_table(
                                num_rows=num_rows,
                                num_key_cols=num_key_cols,
                                num_nonkey_cols=num_nonkey_cols,
                                key_dtype=dtype,
                                has_string_key=False,
                                has_string_nonkey=True,
                                string_length=string_length
                            )
                            
                            # Run benchmark
                            avg_time, min_time, max_time, measurements = run_benchmark(table, num_key_cols)
                            throughput = num_rows / avg_time
                            
                            print(f"    Avg: {avg_time*1000:.3f} ms, "
                                  f"Throughput: {throughput/1e6:.3f} MRows/s")
                            
                            results.append(
                                f"cudf,sort_table_string_payload,{key_type},{num_rows},"
                                f"{num_key_cols},{num_nonkey_cols},{string_length},{avg_time},{min_time},{max_time},{throughput}"
                            )
                            json_results.append({
                                "impl": "cudf",
                                "benchmark": "sort_table_string_payload",
                                "key_type": key_type,
                                "num_rows": num_rows,
                                "num_key_cols": num_key_cols,
                                "num_nonkey_cols": num_nonkey_cols,
                                "string_length": string_length,
                                "avg_time": avg_time,
                                "min_time": min_time,
                                "max_time": max_time,
                                "throughput": throughput,
                                "measurements": measurements
                            })
                        except Exception as e:
                            print(f"    ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark cuDF table sorting")
    parser.add_argument("--benchmark", type=str, default="numeric",
                        choices=["all", "numeric", "string_key", "string_payload"],
                        help="Which benchmark to run")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmark with reduced parameters")
    args = parser.parse_args()
    
    all_results = []
    json_results = []
    
    # Parameter configurations matching C++ benchmark
    if args.quick:
        # Quick mode: fewer configurations for testing
        key_types = ["int32"]
        row_counts_numeric = [2**28]
        row_counts_string = [2**20, 2**22]
        num_key_cols_list = [2]
        num_nonkey_cols_list = [2]
        string_lengths = [16, 64]
    else:
        # Full mode: match C++ benchmark parameters
        key_types = ["int32", "int64"]
        # num_rows: 2^16 to 2^28, step 2
        row_counts_numeric = [2**i for i in range(26, 29, 1)]
        # String benchmarks: 2^16 to 2^26, step 2
        row_counts_string = [2**i for i in range(16, 27, 2)]
        num_key_cols_list = [1, 2, 4]
        num_nonkey_cols_list = [1, 2, 4]
        string_lengths = [8, 16, 32, 64, 128]
    
    # Run selected benchmarks
    if args.benchmark in ["all", "numeric"]:
        benchmark_numeric(all_results, json_results, key_types, row_counts_numeric, 
                          num_key_cols_list, num_nonkey_cols_list)
    
    if args.benchmark in ["all", "string_key"]:
        # String key uses only int32 for payload type (matches C++)
        benchmark_string_key(
            all_results, json_results,
            row_counts_string,
            num_key_cols_list,
            [1, 4],  # num_nonkey_cols for string key
            string_lengths
        )
    
    if args.benchmark in ["all", "string_payload"]:
        benchmark_string_payload(
            all_results, json_results,
            key_types, 
            row_counts_string,
            num_key_cols_list,
            [1, 4],  # num_nonkey_cols for string payload
            [16, 64, 128]  # string lengths for payload
        )
    
    # Write results to CSV
    header = "impl,benchmark,key_type,num_rows,num_key_cols,num_nonkey_cols,string_length,avg_time,min_time,max_time,throughput"
    write_results_to_csv(all_results, "cudf_sort_table", header)
    write_results_to_json(json_results, "cudf_sort_table")
    
    print(f"\nBenchmark complete. {len(all_results)} configurations tested.")
    print("Results written to CSV.")


if __name__ == "__main__":
    main()
