#include <nvbench/nvbench.cuh>

#include "operators/groupby/groupby_sort.cuh"
#include "utils/utils.cuh"
#include "types/tuple.cuh"
#include "groupby_util.cuh"

template<typename KeyT>
void bench_groupby_max(nvbench::state& state, nvbench::type_list<KeyT>) {
    using namespace gamax::groupby;

    struct groupby_params params;
    params.num_items = state.get_int64("num_items");
    params.num_groups = state.get_int64("num_groups");
    params.zipf_factor = state.get_float64("zipf_factor");

    struct groupby_hint hint;
    hint.NDV = params.num_groups;

    using AggT = gamax::groupby::Max<int>;
    using ValueT = int;
    
    KeyT*  h_keys = (KeyT*)malloc(params.num_items * sizeof(KeyT));
    int* h_vals = (int*)malloc(params.num_items * sizeof(int));

    generate_groupby_data(h_keys, h_vals, params);

    using keys_tuple_t = struct Chunk<KeyT>;
    using vals_tuple_t = struct Chunk<ValueT>;
    using agg_t = typename AggT::OutputType;
    using out_tuple_t = struct Chunk<KeyT, agg_t>;

    keys_tuple_t tkeys(params.num_items, 0, h_keys);
    vals_tuple_t tvals(params.num_items, 0, h_vals);
    release_mem(tkeys.select_vec);
    release_mem(tvals.select_vec);
    tkeys.select_vec = nullptr;
    tvals.select_vec = nullptr;
    
    const bool use_hint = hint.NDV != -1;
    out_tuple_t tout;
    tout.allocate(use_hint ? hint.NDV : params.num_items);
    cudaMemset(COL(tout, 1), 0, sizeof(agg_t) * tout.num_items);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));
    state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { 
                gamax::groupby::sort::grouped_agg<AggT>(tkeys, tvals, tout); 
                tout.num_items = 0;
    });


    tkeys.free_mem();
    tvals.free_mem();
    tout.free_mem();
}

NVBENCH_BENCH_TYPES(bench_groupby_max,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
  .set_name("groupby_max")
  .add_int64_power_of_two_axis("num_items", {28, 29})
  .add_int64_power_of_two_axis("num_groups", nvbench::range(2, 22, 2))
  .add_float64_axis("zipf_factor",{0.0});