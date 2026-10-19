// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// Green-context SM-scaling variant of sort/sort_nvbench.cu.
// Adds a `green_sms` axis: the timed sort runs on a CUDA green-context SM
// partition of that many SMs (0 = native full device). All else mirrors the
// original bench_sort_column_pair. See green_ctx.hpp for the mechanism.
#include <nvbench/nvbench.cuh>

#include <string>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>
#include <fstream>
#include <type_traits>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "../common/nvtx_stub.hpp"
#include "green_ctx.hpp"

#include "operators/sort/sort.hpp"
#include "data_gen/table_gen.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"

template<typename KeyT>
struct sort_params {
    int64_t num_items;
    int64_t begin_bit;
    int64_t end_bit;

    sort_params(nvbench::state& state) {
        num_items = state.get_int64("num_items");
        begin_bit = state.get_int64_or_default("begin_bit", 0);
        auto num_bits = state.get_int64_or_default("num_bits", 8*sizeof(KeyT));
        end_bit = begin_bit + num_bits;
    }
};

template<typename KeyT>
bool check_skip(sort_params<KeyT> const& params) {
    if(params.begin_bit < 0 || params.begin_bit >= 8*sizeof(KeyT)) {
        return true;
    }
    if(params.end_bit <= params.begin_bit || params.end_bit > 8*sizeof(KeyT)) {
        return true;
    }
    return false;
}

template<typename KeyT>
void bench_sort_column_pair(nvbench::state& state, nvbench::type_list<KeyT>) {
    gamax::prewarm_device_memory_resource();

    using ValueT = KeyT;

    sort_params<KeyT> params(state);
    if(check_skip<KeyT>(params)) {
        state.skip("Benchmarking sort column configuration skipped!");
        return;
    }

    // generate input data
    constexpr int ncols = 1;
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto value_type = gamax::data_type(gamax::type_to_id<ValueT>());
    auto keys = gamax::datagen::device::fk_uniform_id(
        key_type, params.num_items, params.num_items, ncols, 1.0, 42, cudaStreamDefault);
    auto values = gamax::datagen::device::fk_uniform_id(
        value_type, params.num_items, params.num_items, ncols, 1.0, 43, cudaStreamDefault);

    // Route the timed sort onto an SM partition (0 = native full device).
    GreenCtx green;
    cudaStream_t bench_stream = green_bench_stream(state, green);

    cudaDeviceSynchronize();  // finish datagen (default stream) before timed op on the green stream
    state.set_cuda_stream(nvbench::make_cuda_stream_view(bench_stream));
    try
    {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
            cudaStream_t stream = launch.get_stream();
            nvtxRangePushA("sort_column_pair");
            auto sorted = gamax::sort::sort_column_pair(
                keys->get_column(0).view(),
                values->get_column(0).view(),
                stream,
                params.begin_bit,
                params.end_bit,
                gamax::sort_order::ASCENDING,
                nullptr,
                0);
            cudaStreamSynchronize(stream);
            nvtxRangePop();
        });
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << e.what() << "\n";
    }
    green_teardown(green);
}

// User-fixed: bench_sort_column_pair, num_items = 2^29.
// Swept from the CLI: green_sms (and optionally Key). Keep num_items overridable.
NVBENCH_BENCH_TYPES(bench_sort_column_pair,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("bench_sort_column_pair")
.set_type_axes_names({"Key"})
.add_int64_power_of_two_axis("num_items", {29})
.add_int64_axis("green_sms", {0});
