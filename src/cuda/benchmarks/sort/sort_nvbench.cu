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

#ifdef __NVCC__
#include <nvtx3/nvtx3.hpp>
#endif

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
        assert(end_bit <= 8*sizeof(KeyT));
    }
};

template<typename KeyT>
bool check_skip(sort_params<KeyT> const& params) {
    return false;
}

template<typename KeyT>
void bench_sort_column_pair(nvbench::state& state, nvbench::type_list<KeyT>) {
    // memory cleanup
    release_all_mem();

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
    auto keys = gamax::datagen::device::fk_uniform_id(key_type, params.num_items, params.num_items, ncols, 1.0, 42);
    auto values = gamax::datagen::device::fk_uniform_id(value_type, params.num_items, params.num_items, ncols, 1.0, 43);

    cudaDeviceSynchronize();

    // set up a clean stream to be used
    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));
    try
    {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) { 
            auto sorted = gamax::sort::sort_column_pair(
                keys->get_column(0).view(), 
                values->get_column(0).view(), 
                params.begin_bit, 
                params.end_bit, 
                gamax::sort_order::ASCENDING, 
                nullptr, 
                0, 
                launch.get_stream()); 
            cudaStreamSynchronize(launch.get_stream());
        });
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << e.what() << "\n";
    }
}

NVBENCH_BENCH_TYPES(bench_sort_column_pair,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("bench_sort_column_pair")
.set_type_axes_names({"Key"})
.add_int64_power_of_two_axis("num_items", nvbench::range(5, 29));

NVBENCH_BENCH_TYPES(bench_sort_column_pair,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("bench_sort_column_pair_sweep_bits")
.set_type_axes_names({"Key"})
.add_int64_power_of_two_axis("num_items", {29})
.add_int64_axis("begin_bit", {0})
.add_int64_axis("num_bits", nvbench::range(1, 32, 1));
