// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <nvbench/nvbench.cuh>

#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>
#include <fstream>

#include "operators/sort/sort.hpp"
#include "data_gen/table_gen.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"

namespace {

struct fused_key_sort_params {
    int64_t num_rows;           // Number of rows in the table
    int64_t num_key_cols;       // Number of key columns (2 or 4)
    bool smart_keys_fusion;     // Whether to use smart key fusion optimization

    explicit fused_key_sort_params(nvbench::state& state) {
        num_rows = state.get_int64("num_rows");
        num_key_cols = state.get_int64("num_key_cols");
        smart_keys_fusion = state.get_int64("smart_keys_fusion") != 0;
    }

    void print(std::ostream& os) const {
        os << "num_rows=" << num_rows
           << ", num_key_cols=" << num_key_cols
           << ", smart_keys_fusion=" << smart_keys_fusion
           << std::endl;
    }
};

/**
 * Create a column with sparse unique values spread across a wide range.
 * Values: 0, stride, 2*stride, ..., (ndv-1)*stride
 * This creates a column where range compression is ineffective (needs many bits)
 * but dictionary encoding is effective (only ndv unique values).
 */
template<typename KeyT>
std::unique_ptr<gamax::column> create_sparse_sequence(
    gamax::size_type ndv,
    uint64_t stride,
    cudaStream_t stream)
{
    std::vector<KeyT> host_values(ndv);
    for (gamax::size_type i = 0; i < ndv; ++i) {
        host_values[i] = static_cast<KeyT>(i * stride);
    }

    auto col = gamax::make_column_from_host(host_values.data(), ndv, stream);
    
    return col;
}

/**
 * Generate a table with key columns suitable for testing smart key fusion.
 * 
 * Half of the columns have a narrow (min, max) range - good for range compression.
 * The other half have a wide distribution but small NDV - good for dictionary encoding.
 * 
 * For num_key_cols=2:
 *   - Column 0: narrow range (values in [0, 256))
 *   - Column 1: wide range, low NDV (100 unique values spread across [0, 100*stride))
 * 
 * For num_key_cols=4:
 *   - Columns 0,1: narrow range
 *   - Columns 2,3: wide range, low NDV
 */
template<typename KeyT>
std::unique_ptr<gamax::table> generate_fused_key_table(
    const fused_key_sort_params& params,
    cudaStream_t stream)
{
    std::vector<std::unique_ptr<gamax::column>> columns;
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    
    const int64_t num_narrow_cols = params.num_key_cols / 2;
    const int64_t num_low_ndv_cols = params.num_key_cols - num_narrow_cols;
    
    // Parameters for narrow-range columns (good for range compression)
    // Range: [0, 256) - fits in 8 bits
    const gamax::size_type narrow_ndv = 256;
    
    // Parameters for wide-range, low-NDV columns (good for dictionary encoding)
    // 100 unique values with stride = 2^24, so range spans ~1.6B (needs 31 bits)
    // but only 100 unique values (needs 7 bits after dictionary encoding)
    const gamax::size_type low_ndv = 100;
    const uint64_t stride = static_cast<uint64_t>(1) << 24;  // 16M stride
    
    // Generate narrow-range columns (first half)
    for (int64_t i = 0; i < num_narrow_cols; ++i) {
        auto col_table = gamax::datagen::device::fk_uniform_id(
            key_type,
            narrow_ndv,           // n_unique_keys: only 256 unique values
            params.num_rows,      // cardinality
            1,                    // ncols
            1.0,                  // match_ratio
            42 + static_cast<int>(i),  // seed
            stream,
            0                     // start: values in [0, 256)
        );
        columns.push_back(col_table->remove_column(0));
    }
    
    // Generate wide-range, low-NDV columns (second half)
    // Create a sparse sequence: {0, stride, 2*stride, ..., (low_ndv-1)*stride}
    // Then sample from it uniformly
    auto sparse_seq = create_sparse_sequence<KeyT>(low_ndv, stride, stream);
    
    for (int64_t i = 0; i < num_low_ndv_cols; ++i) {
        auto col_table = gamax::datagen::device::fk_uniform(
            sparse_seq->view(),   // sequence with sparse values
            params.num_rows,      // cardinality
            1,                    // ncols
            1.0,                  // match_ratio
            100 + static_cast<int>(i),  // seed
            stream
        );
        columns.push_back(col_table->remove_column(0));
    }
    
    // Add one payload column
    auto payload_table = gamax::datagen::device::pk_uniform_id(
        key_type, params.num_rows, 1, 999, stream);
    columns.push_back(payload_table->remove_column(0));
    
    return gamax::make_table(std::move(columns));
}

} // anonymous namespace

// =============================================================================
// Benchmark: sort_table with smart_keys_fusion option
// =============================================================================
template<typename KeyT>
void bench_sort_table_fused_keys(nvbench::state& state, nvbench::type_list<KeyT>) {
    gamax::prewarm_device_memory_resource();

    fused_key_sort_params params(state);

    cudaStream_t stream = cudaStreamDefault;
    auto table = generate_fused_key_table<KeyT>(params, stream);
    cudaDeviceSynchronize();
    
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
    state.add_element_count(params.num_rows, "NumRows");
    state.add_global_memory_reads<KeyT>(
        params.num_rows * (params.num_key_cols + 1),  // +1 for payload
        "InputElements");

    // Prepare key column indices (all columns except the last one are keys)
    std::vector<int> key_col_indices;
    for (int64_t i = 0; i < params.num_key_cols; ++i) {
        key_col_indices.push_back(static_cast<int>(i));
    }

    std::vector<gamax::sort_order> sort_orders(params.num_key_cols, gamax::sort_order::ASCENDING);
    
    gamax::sort::sort_table_options options;
    options.smart_keys_fusion = params.smart_keys_fusion;

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                auto sorted = gamax::sort::sort_table(
                    table->view(), key_col_indices, sort_orders, launch.get_stream(), options);
                cudaStreamSynchronize(launch.get_stream());
            });
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << "bench_sort_table_fused_keys error: " << e.what() << "\n";
        params.print(out);
    }
}

// =============================================================================
// NVBench Registration: Smart Key Fusion benchmark
// Dimensions:
//   - KeyType: int32_t, int64_t
//   - num_rows: 2^24, 2^26, 2^28
//   - num_key_cols: 2, 4
//   - smart_keys_fusion: 0 (off), 1 (on)
// =============================================================================
NVBENCH_BENCH_TYPES(bench_sort_table_fused_keys,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
    .set_name("sort_table_fused_keys")
    .set_type_axes_names({"KeyType"})
    .add_int64_power_of_two_axis("num_rows", nvbench::range(24, 28, 2))
    .add_int64_axis("num_key_cols", {2, 4})
    .add_int64_axis("smart_keys_fusion", {0, 1});
