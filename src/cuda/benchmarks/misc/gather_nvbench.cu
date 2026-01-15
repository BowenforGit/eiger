/**
 * @file gather_nvbench.cu
 * @brief NVBench benchmark for gamax::util::device::gather function.
 * 
 * Dimensions explored:
 * - Data types: int32_t, int64_t
 * - Number of input elements (input column size)
 * - Number of gather map elements (output size)
 * 
 * The gather operation performs: output[i] = input[gather_map[i]]
 * The gather map is always shuffled to ensure random access patterns.
 */

#include <nvbench/nvbench.cuh>

#include <string>
#include <iostream>
#include <memory>

#include "utils/transform.hpp"
#include "data_gen/table_gen.hpp"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"

// ============================================================================
// Benchmark: gather with varying input sizes and gather map sizes
// ============================================================================

template<typename ValueT>
void bench_gather(nvbench::state& state, nvbench::type_list<ValueT>) {
    // Memory cleanup to ensure consistent state
    release_all_mem();

    const gamax::size_type num_inputs = state.get_int64("num_inputs");
    const gamax::size_type gather_map_size = state.get_int64("gather_map_size");
    constexpr int seed = 42;

    // Skip invalid configurations: gather map indices must be within input bounds
    if (gather_map_size != num_inputs) {
        state.skip("Gather map size != input size.");
        return;
    }

    // Generate input column with uniformly distributed values
    auto input_type = gamax::data_type(gamax::type_to_id<ValueT>());
    auto input_table = gamax::datagen::device::pk_uniform_id(
        input_type, num_inputs, 1, seed);
    auto input_col = input_table->get_column(0).view();

    // Generate shuffled gather map using fk_uniform_id
    // This creates indices in [0, num_inputs) with uniform distribution and shuffling
    auto gather_map_type = gamax::data_type(gamax::type_to_id<gamax::size_type>());
    auto gather_map_table = gamax::datagen::device::fk_uniform_id(
        gather_map_type, num_inputs, gather_map_size, 1, 1.0, seed + 1);
    auto gather_map_col = gather_map_table->get_column(0).view();

    cudaDeviceSynchronize();

    // Set up benchmark stream
    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));

    // Add throughput metrics
    state.add_element_count(gather_map_size, "OutputElements");
    state.add_global_memory_reads<ValueT>(num_inputs, "InputRead");
    state.add_global_memory_reads<gamax::size_type>(gather_map_size, "GatherMapRead");
    state.add_global_memory_writes<ValueT>(gather_map_size, "OutputWrite");

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                auto output = gamax::util::device::gather(
                    gather_map_col,
                    input_col,
                    launch.get_stream());
                cudaStreamSynchronize(launch.get_stream());
            });
    } catch (const std::exception& e) {
        std::cerr << "Gather benchmark error: " << e.what() << std::endl;
        state.skip(e.what());
    }
}

// ============================================================================
// NVBench Registration
// ============================================================================

NVBENCH_BENCH_TYPES(bench_gather,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("bench_gather")
.set_type_axes_names({"ValueType"})
.add_int64_power_of_two_axis("num_inputs", nvbench::range(25, 28, 1))
.add_int64_power_of_two_axis("gather_map_size", nvbench::range(25, 28, 1));