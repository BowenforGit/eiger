import pylibcudf as plc
import pyarrow as pa
import timeit
import string
from cudf.core.column.column import ColumnBase
from bench_cudf_utils import *

bench_cudf_init()

WARMUP_RUNS, BENCHMARK_RUNS = get_bench_params()

def bench_startswith(col, pattern):
    start_time = timeit.default_timer()
    result = plc.strings.find.starts_with(col, pattern)
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    return result, elapsed_time

def bench_endswith(col, pattern):
    start_time = timeit.default_timer()
    result = plc.strings.find.ends_with(col, pattern)
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    return result, elapsed_time

def bench_contains(col, pattern, use_kmp=False):
    start_time = timeit.default_timer()
    result = plc.strings.find.contains(col, pattern)
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    return result, elapsed_time

op_to_bench_func = {
    "equal": bench_startswith,  # Reuse startswith for equal benchmarking
    "startswith": bench_startswith,
    "endswith": bench_endswith,
    "contains": bench_contains,
}

def generate_plc_col(str, num_strings):
    pa_table = pa.table({"string": [str]})
    plc_table = plc.Table.from_arrow(pa_table)
    gather_map = plc.column_factories.make_numeric_column(
        plc.DataType.from_arrow(pa.int32()), 
        num_strings, 
        plc.types.MaskState.UNALLOCATED)
    plc.filling.fill_in_place(
        gather_map, 
        0, num_strings, 
        plc.Scalar.from_arrow(pa.scalar(0, pa.int32())))
    result = plc.copying.gather(plc_table, gather_map, plc.copying.OutOfBoundsPolicy.DONT_CHECK)
    return result.columns()[0]

def generate_startswith_strings(num_strings, str_len, pattern):
    string = pattern + 'a' * (str_len - len(pattern))
    return generate_plc_col(string, num_strings)

def generate_endswith_strings(num_strings, str_len, pattern):
    string = 'a' * (str_len - len(pattern)) + pattern
    return generate_plc_col(string, num_strings)

def generate_contains_strings(num_strings, str_len, pattern):
    string = [pattern[i % len(pattern)] for i in range(str_len - len(pattern))]
    for i in range(str_len - len(pattern)):
        if i % len(pattern) == len(pattern) - 1:
            string[i] = " " # introduce a mismatch
    string = ''.join(string) + pattern
    return generate_plc_col(string, num_strings)

def generate_strings(op, num_strings, str_len, pattern):
    if op == "startswith" or op == "equal":
        return generate_startswith_strings(num_strings, str_len, pattern)
    elif op == "endswith":
        return generate_endswith_strings(num_strings, str_len, pattern)
    elif op == "contains":
        return generate_contains_strings(num_strings, str_len, pattern)
    else:
        raise ValueError(f"Unsupported operation: {op}")
    
def generate_pattern(str_len):
    pattern = ''.join([string.ascii_lowercase[i % 26] for i in range(str_len)])
    pa_pattern = pa.scalar(pattern, pa.string())
    return pattern, plc.Scalar.from_arrow(pa_pattern)

def print_plc_col(plc_col):
    cudf_col = ColumnBase.from_pylibcudf(plc_col)
    print(cudf_col)
    print(plc_col.size())

def bench(plc_col, plc_pattern, op):
    bench_func = op_to_bench_func[op]

    results = []
    
    # warm up
    for _ in range(WARMUP_RUNS):
        _, elapsed_time = bench_func(plc_col, plc_pattern)

    # Benchmark
    for _ in range(BENCHMARK_RUNS):
        _, elapsed_time = bench_func(plc_col, plc_pattern)
        results.append(elapsed_time)
    
    avg_time = sum(results) / len(results)
    min_time = min(results)
    max_time = max(results)
    print(f"Min Time: {min_time*1000} ms, Max Time: {max_time*1000} ms, Avg Time: {avg_time*1000} ms")
    num_strings = plc_col.size()
    thpt = num_strings / avg_time
    return avg_time, min_time, max_time, thpt, results

if __name__ == "__main__":
    all_results = []
    json_results = []
    
    for plen in [5, 50]:
        pattern, plc_pattern = generate_pattern(plen)
        for op in ["startswith", "endswith", "contains"]:
            for num_strings in [2**27]:
                for strlen in [30, 45, 60, 75, 90, 105, 120, 135] + [17, 34, 51, 68, 85, 102, 119, 136]:
                    print(f"Benchmarking {op} with strlen={strlen}, plen={plen}, num_strings={num_strings}")
                    if plen > strlen:
                        print("[Skip]")
                        continue
                    plc_col = generate_strings(op, num_strings, strlen, pattern)
                    avg_time, min_time, max_time, thpt, measurements = bench(plc_col, plc_pattern, op)
                    print(f"Avg Time: {avg_time*1000}, Throughput: {thpt/1e9} GStrings/s")
                    all_results.append(f"cudf,{op},{num_strings},{strlen},{plen},{avg_time},{min_time},{max_time},{thpt}")
                    json_results.append({
                        "impl": "cudf",
                        "op": op,
                        "num_strings": num_strings,
                        "fixed_length": strlen,
                        "pattern_length": plen,
                        "avg_time": avg_time,
                        "min_time": min_time,
                        "max_time": max_time,
                        "thpt": thpt,
                        "measurements": measurements
                    })
    
    for op in ["equal"]:
        for num_strings in [2**27]:
            for strlen in [30, 45, 60, 75, 90, 105, 120, 135] + [17, 34, 51, 68, 85, 102, 119, 136]:
                pattern, plc_pattern = generate_pattern(strlen)
                print(f"Benchmarking {op} with strlen={strlen}, plen={strlen}, num_strings={num_strings}")
                plc_col = generate_strings(op, num_strings, strlen, pattern)
                avg_time, min_time, max_time, thpt, measurements = bench(plc_col, plc_pattern, op)
                print(f"Avg Time: {avg_time*1000}, Throughput: {thpt/1e9} GStrings/s")
                all_results.append(f"cudf,{op},{num_strings},{strlen},{strlen},{avg_time},{min_time},{max_time},{thpt}")
                json_results.append({
                    "impl": "cudf",
                    "op": op,
                    "num_strings": num_strings,
                    "fixed_length": strlen,
                    "pattern_length": strlen,
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "thpt": thpt,
                    "measurements": measurements
                })
    
    write_results_to_csv(
        all_results, 
        "cudf_stringops", 
        "impl,op,num_strings,fixed_length,pattern_length,avg_time,min_time,max_time,thpt")
    write_results_to_json(json_results, "cudf_stringops")
