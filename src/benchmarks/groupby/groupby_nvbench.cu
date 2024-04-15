#include <nvbench/nvbench.cuh>

#include <string>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/groupby_sort.hpp"
#include "operators/groupby/groupby_partition.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "groupby_util.cuh"

template<typename KeyT>
void bench_groupby_max(nvbench::state& state, nvbench::type_list<KeyT>) {
    using namespace gamax::groupby;
    
    release_all_mem(); // make sure if a test fails, the following test will not be affected

    struct groupby_params params;
    params.num_items = state.get_int64("num_items");
    params.num_groups = state.get_int64("num_groups");
    params.zipf_factor = state.get_float64("zipf_factor");
    auto algo = state.get_string("algo");

    struct groupby_hint hint;
    hint.groups_upper_bound = params.num_groups;
    
    auto  h_keys = new KeyT[params.num_items];
    auto h_vals = new int[params.num_items];

    generate_groupby_data(h_keys, h_vals, params);

    auto d_keys = gamax::make_column_from_host(h_keys, params.num_items);
    auto d_vals = gamax::make_column_from_host(h_vals, params.num_items);

    std::vector<decltype(d_keys)> keys_vec;
    keys_vec.push_back(std::move(d_keys));
    std::vector<decltype(d_vals)> vals_vec;
    vals_vec.push_back(std::move(d_vals));

    gamax::table key_table(std::move(keys_vec));
    gamax::table val_table(std::move(vals_vec));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::MAX});

    auto f = gamax::groupby::sort::groupby;
    if(algo == "partition") {
        f = gamax::groupby::partition::groupby;
    } else if(algo == "sort_gfur") {
        f = gamax::groupby::sort::groupby_gfur;
    }

    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));
    state.add_element_count(params.num_items, "NumElements");

    state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { 
                auto res = f(key_table, val_table, requests, hint); 
    });

    delete [] h_keys;
    delete [] h_vals;
}

NVBENCH_BENCH_TYPES(bench_groupby_max,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
  .set_name("groupby_max")
  .add_string_axis("algo", {"sort", "partition", "sort_gfur"})
  .add_int64_power_of_two_axis("num_items", {28, 29})
  .add_int64_power_of_two_axis("num_groups", nvbench::range(2, 22, 2))
  .add_float64_axis("zipf_factor", {0.0, 0.5, 1.0, 1.5, 2.0});