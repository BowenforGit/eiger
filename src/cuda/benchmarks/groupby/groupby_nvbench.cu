#include <nvbench/nvbench.cuh>

#include <string>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>
#include <fstream>

#include "../common/nvtx_stub.hpp"

#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "groupby_util.cuh"

template<typename KeyT, typename ValueT>
bool check_skip(nvbench::state& state, gamax::groupby::groupby_params const& params) {
    if(sizeof(KeyT) > 4 && params.zipf_factor > 0.0) {
        state.skip("Only study skewness with 32-bit keys.");
        return true;
    }

    if(params.num_aggs != 2 && params.zipf_factor > 0.0) {
        state.skip("Only study skewness with exactly 2 aggregations");
        return true;
    }

    if(params.num_groups > params.num_items) {
        state.skip("Number of groups should not be greater than the number of items.");
        return true;
    }

    return false;
}

template<typename KeyT, typename ValueT = int32_t>
void bench_groupby_max(nvbench::state& state, nvbench::type_list<KeyT>) {
    using namespace gamax::groupby;

    release_all_mem(); // make sure if a test fails, the following test will not be affected

    struct groupby_params params;
    params.num_items = state.get_int64("num_items");
    params.num_groups = state.get_int64("num_groups");
    params.num_aggs = state.get_int64("num_aggs");
    params.zipf_factor = state.get_float64("zipf_factor");

    if(check_skip<KeyT, ValueT>(state, params)) {
        return;
    }

    struct groupby_hint hint;
    hint.groups_upper_bound = params.num_groups;
    hint.zero_based_ids = false;
    hint.optimize_sort = false;

    gamax::table key_table;
    gamax::table val_table;
    generate_groupby_data<KeyT, ValueT>(key_table, val_table, params);

    std::vector<gamax::groupby::aggregation_request> requests;
    for(int i = 0; i < params.num_aggs; i++) {
        requests.push_back({i, gamax::groupby::aggregation_id::MAX});
    }

    auto algo = state.get_string("algo");
    auto f = gamax::groupby::groupby;

    if(algo == "sort_gfur") {
        hint.groupby_algo = gamax::groupby::groupby_algo::SORT_GFUR;
    } else if(algo == "sort") {
        hint.groupby_algo = gamax::groupby::groupby_algo::SORT;
    } else if(algo == "hash") {
        hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    } else if(algo == "sort_opt") {
        hint.groupby_algo = gamax::groupby::groupby_algo::SORT;
        hint.optimize_sort = true;
    }

    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));
    state.add_element_count(params.num_items, "NumElements");

    try
    {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) { 
            auto res = f(key_table.view(), val_table.view(), requests, hint, launch.get_stream()); 
            cudaStreamSynchronize(launch.get_stream());
#ifdef __NVCC__
            nvtx3::mark("GroupBy Done");
#endif
        });
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << e.what() << "\n";
        params.print(out);
    }
}

NVBENCH_BENCH_TYPES(bench_groupby_max,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
  .set_name("groupby_max")
  .set_type_axes_names({"Key"})
  .add_string_axis("algo", {"sort", "sort_opt", "sort_gfur", "hash"})
  .add_int64_power_of_two_axis("num_items", {28, 29})
  .add_int64_power_of_two_axis("num_groups", nvbench::range(2, 26, 2))
  .add_int64_axis("num_aggs", {1, 2, 4})
  .add_float64_axis("zipf_factor", {0.0, 0.5, 1.0, 1.5, 2.0});
