"""
Benchmark for cuDF string sorting operations.
Mirrors the string_sort_nvbench.cu benchmark for comparison.
"""

import pylibcudf as plc
import pyarrow as pa
import cupy as cp
import timeit
from bench_cudf_utils import write_results_to_csv

WARMUP_RUNS = 5
BENCHMARK_RUNS = 10


def generate_random_strings_column(num_strings: int, string_length: int, seed: int = 42) -> tuple[plc.Column, plc.Table]:
    """
    Generate a pylibcudf Column containing random strings using CuPy.
    
    Mirrors the C++ generate_random_strings implementation:
    - Uses random number generator seeded per-string
    - Generates lowercase letters 'a' to 'z' (26 characters)
    - Each character is independently random
    
    Args:
        num_strings: Number of strings to generate
        string_length: Length of each string
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (pylibcudf Column, pylibcudf Table) containing the generated strings
    """
    # Use CuPy for GPU-accelerated random generation
    rng = cp.random.default_rng(seed)
    
    # Generate random character codes (lowercase letters 'a' to 'z': ASCII 97-122)
    # This mirrors the C++ implementation: 'a' + (rng() % 26)
    # Shape: (num_strings, string_length) then flatten
    char_codes = rng.integers(0, 26, size=(num_strings, string_length), dtype=cp.uint8)
    char_codes += 97  # Convert to ASCII 'a'-'z'
    
    # Convert to bytes on CPU (need to materialize strings for PyArrow)
    # This is the bottleneck but unavoidable without low-level cudf string APIs
    char_codes_cpu = cp.asnumpy(char_codes)
    
    # Convert to list of strings using vectorized numpy operations
    # Each row becomes a string
    strings = [row.tobytes().decode('ascii') for row in char_codes_cpu]
    
    # Create PyArrow table with strings
    pa_array = pa.array(strings, type=pa.string())
    pa_table = pa.table({"strings": pa_array})
    plc_table = plc.Table.from_arrow(pa_table)
    
    return plc_table.columns()[0], plc_table

def bench_sort(plc_table: plc.Table) -> float:
    """
    Benchmark sorting a string table.
    
    Args:
        plc_table: The pylibcudf Table containing a string column
    
    Returns:
        Elapsed time in seconds
    """
    start_time = timeit.default_timer()
    result = plc.sorting.sort(
        plc_table, 
        [plc.types.Order.ASCENDING], 
        [plc.types.NullOrder.AFTER]
    )
    end_time = timeit.default_timer()
    return end_time - start_time

def bench_argsort(plc_table: plc.Table) -> float:
    """
    Benchmark computing sorted order (argsort) of a string table.
    
    Args:
        plc_table: The pylibcudf Table containing a string column
    
    Returns:
        Elapsed time in seconds
    """
    start_time = timeit.default_timer()
    result = plc.sorting.sorted_order(
        plc_table, 
        [plc.types.Order.ASCENDING], 
        [plc.types.NullOrder.AFTER]
    )
    end_time = timeit.default_timer()
    return end_time - start_time

op_to_bench_func = {
    "sort": bench_sort,
    "argsort": bench_argsort,
}

def run_benchmark(plc_table: plc.Table, op: str) -> tuple[float, float]:
    """
    Run benchmark for a given operation.
    
    Args:
        plc_table: The pylibcudf Table containing a string column
        op: Operation to benchmark ("sort" or "argsort")
    
    Returns:
        Tuple of (average_time, throughput)
    """
    bench_func = op_to_bench_func[op]
    results = []
    
    # Warmup runs
    for _ in range(WARMUP_RUNS):
        _ = bench_func(plc_table)
    
    # Benchmark runs
    for _ in range(BENCHMARK_RUNS):
        elapsed_time = bench_func(plc_table)
        results.append(elapsed_time)
    
    avg_time = sum(results) / len(results)
    num_strings = plc_table.num_rows()
    throughput = num_strings / avg_time
    
    return avg_time, throughput

def main():
    all_results = []
    
    # Match parameters from string_sort_nvbench.cu
    # string_length: {16, 32, 64, 128, 256}
    # num_strings: {2^20, 2^22, 2^24, 2^26}
    # ops: {"argsort", "sort"}
    
    string_lengths = [16, 32, 64, 128, 256]
    num_strings_list = [2**26]
    ops = ["argsort"]
    
    for op in ops:
        for num_strings in num_strings_list:
            for string_length in string_lengths:
                print(f"Benchmarking {op} with num_strings={num_strings}, string_length={string_length}")
                
                # Generate random strings using CuPy (GPU-accelerated)
                print("  Generating random strings...")
                _, plc_table = generate_random_strings_column(num_strings, string_length)
                print(f"  Generated {num_strings} strings of length {string_length}")
                
                # Run benchmark
                avg_time, throughput = run_benchmark(plc_table, op)
                
                print(f"  Avg Time: {avg_time*1000:.3f} ms, Throughput: {throughput/1e6:.3f} MStrings/s")
                
                # Record results: impl, op, num_strings, string_length, avg_time, throughput
                all_results.append(f"cudf,{op},{num_strings},{string_length},{avg_time},{throughput}")
    
    # Write results to CSV
    write_results_to_csv(
        all_results,
        "cudf_string_sort",
        "impl,op,num_strings,string_length,avg_time,thpt"
    )
    
    print("\nBenchmark complete. Results written to CSV.")

if __name__ == "__main__":
    main()
