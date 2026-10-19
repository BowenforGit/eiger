// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

/**
 * @file gather_green_nvbench.cu
 * @brief Green-context SM-scaling variant of misc/gather_nvbench.cu.
 *
 * Adds a `green_sms` axis: the timed gather runs on a CUDA green-context SM
 * partition of that many SMs (0 = native full device). User-fixed algo=thrust.
 * See green_ctx.hpp for the mechanism.
 */
#include <nvbench/nvbench.cuh>

#include <string>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "../common/nvtx_stub.hpp"
#include "green_ctx.hpp"

#include "utils/transform.hpp"
#include "data_gen/table_gen.hpp"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"

// Simple gather kernel + wrapper (kept for the "ours" algo; unused when algo=thrust).
template <typename GatherMapT, typename ValueT>
__global__ void gather_kernel(const GatherMapT* __restrict__ gather_map,
                              const ValueT* __restrict__ input,
                              ValueT* __restrict__ output,
                              gamax::size_type n) {
    gamax::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    gamax::size_type stride = blockDim.x * gridDim.x;
    for (gamax::size_type i = idx; i < n; i += stride) {
        output[i] = input[gather_map[i]];
    }
}

template <typename ValueT>
std::unique_ptr<gamax::column> simple_gather(
    gamax::column_view gather_map,
    gamax::column_view input,
    cudaStream_t stream) {
    const gamax::size_type n = gather_map.size();
    auto output = std::make_unique<gamax::column>(input.type(), n, stream);

    constexpr int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    gather_kernel<gamax::size_type, ValueT><<<grid_size, block_size, 0, stream>>>(
        gather_map.data<gamax::size_type>(),
        input.data<ValueT>(),
        output->mutable_view().data<ValueT>(),
        n);

    return output;
}

template<typename ValueT>
void bench_gather(nvbench::state& state, nvbench::type_list<ValueT>) {
    gamax::prewarm_device_memory_resource();

    const gamax::size_type num_inputs = state.get_int64("num_inputs");
    const gamax::size_type gather_map_size = state.get_int64("gather_map_size");
    const std::string algo = state.get_string("algo");
    constexpr int seed = 42;

    if (gather_map_size != num_inputs) {
        state.skip("Gather map size != input size.");
        return;
    }

    auto input_type = gamax::data_type(gamax::type_to_id<ValueT>());
    auto input_table = gamax::datagen::device::pk_uniform_id(
        input_type, num_inputs, 1, seed, cudaStreamDefault);
    auto input_col = input_table->get_column(0).view();

    auto gather_map_type = gamax::data_type(gamax::type_to_id<gamax::size_type>());
    auto gather_map_table = gamax::datagen::device::fk_uniform_id(
        gather_map_type, num_inputs, gather_map_size, 1, 1.0, seed + 1,
        cudaStreamDefault);
    auto gather_map_col = gather_map_table->get_column(0).view();

    // Route the timed gather onto an SM partition (0 = native full device).
    GreenCtx green;
    cudaStream_t bench_stream = green_bench_stream(state, green);

    cudaDeviceSynchronize();  // finish datagen before timed op on the green stream
    state.set_cuda_stream(nvbench::make_cuda_stream_view(bench_stream));

    // Untimed green-context correctness/readiness gate. Validate one gathered
    // element before benchmarking so a successful launch cannot hide a wrong
    // result or an allocation/view that is not ready on the green stream.
    auto checked = [](cudaError_t error, char const* operation) {
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string(operation) + " failed: " + cudaGetErrorString(error));
        }
    };
    auto validation_output = gamax::util::device::gather(
        gather_map_col, input_col, bench_stream);
    gamax::size_type gather_index = -1;
    checked(
        cudaMemcpyAsync(
            &gather_index,
            gather_map_col.data<gamax::size_type>(),
            sizeof(gather_index),
            cudaMemcpyDeviceToHost,
            bench_stream),
        "green gather index copy");
    checked(cudaStreamSynchronize(bench_stream), "green gather index synchronization");
    if (gather_index < 0 || gather_index >= input_col.size()) {
        throw std::runtime_error("green gather produced an out-of-range gather index");
    }
    ValueT expected{};
    ValueT actual{};
    checked(
        cudaMemcpyAsync(
            &expected,
            input_col.data<ValueT>() + gather_index,
            sizeof(ValueT),
            cudaMemcpyDeviceToHost,
            bench_stream),
        "green gather expected-value copy");
    checked(
        cudaMemcpyAsync(
            &actual,
            validation_output->view().data<ValueT>(),
            sizeof(ValueT),
            cudaMemcpyDeviceToHost,
            bench_stream),
        "green gather actual-value copy");
    checked(cudaStreamSynchronize(bench_stream), "green gather validation synchronization");
    if (actual != expected) {
        throw std::runtime_error("green gather validation mismatch");
    }
    validation_output.reset();

    state.add_element_count(gather_map_size, "OutputElements");
    state.add_global_memory_reads<ValueT>(num_inputs, "InputRead");
    state.add_global_memory_reads<gamax::size_type>(gather_map_size, "GatherMapRead");
    state.add_global_memory_writes<ValueT>(gather_map_size, "OutputWrite");

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                cudaStream_t stream = launch.get_stream();
                nvtxRangePushA("gather");
                std::unique_ptr<gamax::column> output;
                if (algo == "thrust") {
                    output = gamax::util::device::gather(
                        gather_map_col, input_col, stream);
                } else {
                    output = simple_gather<ValueT>(
                        gather_map_col, input_col, stream);
                }
                cudaStreamSynchronize(stream);
                nvtxRangePop();
            });
    } catch (const std::exception& e) {
        std::cerr << "Gather benchmark error: " << e.what() << std::endl;
        state.skip(e.what());
    }
    green_teardown(green);
}

// User-fixed: algo=thrust. Swept from the CLI: green_sms, num_inputs=gather_map_size.
NVBENCH_BENCH_TYPES(bench_gather,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("bench_gather")
.set_type_axes_names({"ValueType"})
.add_int64_power_of_two_axis("num_inputs", {28})
.add_int64_power_of_two_axis("gather_map_size", {28})
.add_string_axis("algo", {"thrust"})
.add_int64_axis("green_sms", {0});
