#include <nvbench/nvbench.cuh>

#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>
#include <fstream>

#include "operators/sort/sort.hpp"
#include "data_gen/table_gen.hpp"
#include "data_gen/string_generator.hpp"
#include "string/string_column.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"

namespace {

struct sort_table_params {
    int64_t num_rows;           // Number of rows in the table
    int64_t num_key_cols;       // Number of key columns for sorting
    int64_t num_nonkey_cols;    // Number of non-key (payload) columns
    bool has_string_key;        // Whether one key column is a string
    bool has_string_nonkey;     // Whether one non-key column is a string
    int64_t string_length;      // Length of strings (if any)

    explicit sort_table_params(nvbench::state& state) {
        num_rows = state.get_int64("num_rows");
        num_key_cols = state.get_int64_or_default("num_key_cols", 1);
        num_nonkey_cols = state.get_int64_or_default("num_nonkey_cols", 1);
        has_string_key = state.get_int64_or_default("has_string_key", 0) != 0;
        has_string_nonkey = state.get_int64_or_default("has_string_nonkey", 0) != 0;
        string_length = state.get_int64_or_default("string_length", 16);
    }

    void print(std::ostream& os) const {
        os << "num_rows=" << num_rows
           << ", num_key_cols=" << num_key_cols
           << ", num_nonkey_cols=" << num_nonkey_cols
           << ", has_string_key=" << has_string_key
           << ", has_string_nonkey=" << has_string_nonkey
           << ", string_length=" << string_length
           << std::endl;
    }
};

template<typename KeyT>
bool check_skip(nvbench::state& state, const sort_table_params& params) {
    // String key column sorting with multi-key not supported
    if (params.has_string_key && params.num_key_cols > 1) {
        state.skip("Multi-key sorting with string keys not yet supported.");
        return true;
    }
    return false;
}

template<typename KeyT>
std::unique_ptr<gamax::table> generate_benchmark_table(
    const sort_table_params& params,
    cudaStream_t stream)
{
    std::vector<std::unique_ptr<gamax::column>> columns;
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    
    // Generate key columns
    for (int64_t i = 0; i < params.num_key_cols; ++i) {
        if (i == 0 && params.has_string_key) {
            auto str_col = gamax::datagen::device::generate_random_strings(
                params.num_rows, params.string_length, gamax::ARROW_STRING,
                42 + static_cast<int>(i), stream);
            columns.push_back(std::move(str_col));
        } else {
            auto key_table = gamax::datagen::device::fk_uniform_id(
                key_type, params.num_rows, params.num_rows, 1, 1.0,
                42 + static_cast<int>(i));
            columns.push_back(key_table->remove_column(0));
        }
    }
    
    // Generate non-key (payload) columns
    for (int64_t i = 0; i < params.num_nonkey_cols; ++i) {
        if (i == 0 && params.has_string_nonkey) {
            auto str_col = gamax::datagen::device::generate_random_strings(
                params.num_rows, params.string_length, gamax::ARROW_STRING,
                100 + static_cast<int>(i), stream);
            columns.push_back(std::move(str_col));
        } else {
            auto val_table = gamax::datagen::device::fk_uniform_id(
                key_type, params.num_rows, params.num_rows, 1, 1.0,
                100 + static_cast<int>(i));
            columns.push_back(val_table->remove_column(0));
        }
    }
    
    return gamax::make_table(std::move(columns));
}

} // anonymous namespace

// =============================================================================
// Benchmark: sort_table with numeric key columns
// =============================================================================
template<typename KeyT>
void bench_sort_table_numeric(nvbench::state& state, nvbench::type_list<KeyT>) {
    release_all_mem();

    sort_table_params params(state);
    params.has_string_key = false;
    params.has_string_nonkey = false;
    
    if (check_skip<KeyT>(state, params)) {
        return;
    }

    cudaStream_t stream = cudaStreamDefault;
    auto table = generate_benchmark_table<KeyT>(params, stream);
    cudaDeviceSynchronize();
    
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
    state.add_element_count(params.num_rows, "NumRows");
    state.add_global_memory_reads<KeyT>(
        params.num_rows * (params.num_key_cols + params.num_nonkey_cols),
        "InputElements");

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                auto sorted = gamax::sort::sort_table(
                    table->view(), 0, gamax::sort_order::ASCENDING, launch.get_stream());
                cudaStreamSynchronize(launch.get_stream());
            });
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << "bench_sort_table_numeric error: " << e.what() << "\n";
        params.print(out);
    }
}

// =============================================================================
// Benchmark: sort_table with string key column
// =============================================================================
template<typename KeyT>
void bench_sort_table_string_key(nvbench::state& state, nvbench::type_list<KeyT>) {
    release_all_mem();

    sort_table_params params(state);
    params.has_string_key = true;
    params.num_key_cols = 1;
    
    if (check_skip<KeyT>(state, params)) {
        return;
    }

    cudaStream_t stream = cudaStreamDefault;
    auto table = generate_benchmark_table<KeyT>(params, stream);
    cudaDeviceSynchronize();
    
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
    state.add_element_count(params.num_rows, "NumRows");
    state.add_global_memory_reads<char>(
        params.num_rows * params.string_length, "StringBytes");

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                auto sorted = gamax::sort::sort_table(
                    table->view(), 0, gamax::sort_order::ASCENDING, launch.get_stream());
                cudaStreamSynchronize(launch.get_stream());
            });
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << "bench_sort_table_string_key error: " << e.what() << "\n";
        params.print(out);
    }
}

// =============================================================================
// Benchmark: sort_table with string payload columns (non-key)
// =============================================================================
template<typename KeyT>
void bench_sort_table_string_payload(nvbench::state& state, nvbench::type_list<KeyT>) {
    release_all_mem();

    sort_table_params params(state);
    params.has_string_key = false;
    params.has_string_nonkey = true;
    
    if (check_skip<KeyT>(state, params)) {
        return;
    }

    cudaStream_t stream = cudaStreamDefault;
    auto table = generate_benchmark_table<KeyT>(params, stream);
    cudaDeviceSynchronize();
    
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
    state.add_element_count(params.num_rows, "NumRows");

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                auto sorted = gamax::sort::sort_table(
                    table->view(), 0, gamax::sort_order::ASCENDING, launch.get_stream());
                cudaStreamSynchronize(launch.get_stream());
            });
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << "bench_sort_table_string_payload error: " << e.what() << "\n";
        params.print(out);
    }
}

// =============================================================================
// NVBench Registration: Numeric columns (sweep rows, key types, column counts)
// =============================================================================
NVBENCH_BENCH_TYPES(bench_sort_table_numeric,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
    .set_name("sort_table_numeric")
    .set_type_axes_names({"KeyType"})
    .add_int64_power_of_two_axis("num_rows", nvbench::range(16, 28, 2))
    .add_int64_axis("num_key_cols", {1, 2, 4})
    .add_int64_axis("num_nonkey_cols", {1, 4, 8});

// =============================================================================
// NVBench Registration: String Key (sweep row counts and string lengths)
// =============================================================================
NVBENCH_BENCH_TYPES(bench_sort_table_string_key,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t>))
    .set_name("sort_table_string_key")
    .set_type_axes_names({"PayloadType"})
    .add_int64_power_of_two_axis("num_rows", nvbench::range(10, 26, 2))
    .add_int64_axis("num_nonkey_cols", {1, 4})
    .add_int64_axis("string_length", {8, 16, 32, 64, 128});

// =============================================================================
// NVBench Registration: String Payload (numeric key, string non-key column)
// =============================================================================
NVBENCH_BENCH_TYPES(bench_sort_table_string_payload,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
    .set_name("sort_table_string_payload")
    .set_type_axes_names({"KeyType"})
    .add_int64_power_of_two_axis("num_rows", nvbench::range(10, 26, 2))
    .add_int64_axis("num_key_cols", {1})
    .add_int64_axis("num_nonkey_cols", {1, 4})
    .add_int64_axis("string_length", {16, 64, 128});
