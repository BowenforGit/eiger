/**
 * @file read_write_nvbench.cu
 * @brief NVBench benchmark for simple read-write memory bandwidth.
 * 
 * Tests a trivial kernel that reads values from an input column,
 * computes value + 1, and writes to an output column.
 * This establishes baseline memory throughput for comparison with
 * more complex operations.
 * 
 * Dimensions explored:
 * - Data types: int32_t, int64_t
 * - Number of items
 */

#include <nvbench/nvbench.cuh>

#include <string>
#include <iostream>
#include <memory>

#include "data_gen/table_gen.hpp"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"

// ============================================================================
// Kernel: Read value, add 1, write output
// ============================================================================

template<typename T>
__global__ void read_add_write_kernel(const T* __restrict__ input, 
                                       T* __restrict__ output, 
                                       gamax::size_type n) {
    const gamax::size_type idx = static_cast<gamax::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + static_cast<T>(1);
    }
}

// ============================================================================
// Benchmark: Simple read-write with add
// ============================================================================

template<typename ValueT>
void bench_read_write(nvbench::state& state, nvbench::type_list<ValueT>) {
    // Memory cleanup to ensure consistent state
    release_all_mem();

    const gamax::size_type num_items = state.get_int64("num_items");
    constexpr int seed = 42;

    // Generate input column with random values
    auto input_type = gamax::data_type(gamax::type_to_id<ValueT>());
    auto input_table = gamax::datagen::device::fk_uniform_id(
        input_type, num_items, num_items, 1, 1.0, seed);
    auto input_col = input_table->get_column(0).view();

    // Allocate output column
    auto output_col = gamax::make_column<ValueT>(num_items);

    cudaDeviceSynchronize();

    // Set up benchmark stream
    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));

    // Add throughput metrics
    state.add_element_count(num_items, "Elements");
    state.add_global_memory_reads<ValueT>(num_items, "InputRead");
    state.add_global_memory_writes<ValueT>(num_items, "OutputWrite");

    // Kernel launch configuration
    constexpr int block_size = 256;
    const int grid_size = (num_items + block_size - 1) / block_size;

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                read_add_write_kernel<ValueT><<<grid_size, block_size, 0, launch.get_stream()>>>(
                    input_col.data<ValueT>(),
                    output_col->template data<ValueT>(),
                    num_items);
                cudaStreamSynchronize(launch.get_stream());
            });
    } catch (const std::exception& e) {
        std::cerr << "Read-write benchmark error: " << e.what() << std::endl;
        state.skip(e.what());
    }
}

// ============================================================================
// NVBench Registration
// ============================================================================

NVBENCH_BENCH_TYPES(bench_read_write,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("bench_read_write")
.set_type_axes_names({"ValueType"})
.add_int64_power_of_two_axis("num_items", nvbench::range(10, 28, 1));  // 1K to 256M
