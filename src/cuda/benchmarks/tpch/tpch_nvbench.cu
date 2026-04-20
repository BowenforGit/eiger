#include <nvbench/nvbench.cuh>
#include <string>

#include "tpch/tpch.hpp"
#include "memory/mem_manager.hpp"

using namespace gamax::tpch;

namespace {

// Populate tpch_params::algo_map from nvbench axes automatically.
// Convention: every string axis is an operator parameter.
// The axis name is the algo_map key, the axis value is parsed via
// tpch_params::parse_algo_value.
tpch_params params_from_state(nvbench::state& state) {
    auto sf = state.get_int64("scale_factor");
    tpch_params params(sf);

    auto const& axis_values = state.get_axis_values();
    for (auto const& name : axis_values.get_names()) {
        if (axis_values.get_type(name) == nvbench::named_values::type::string) {
            auto const& value = axis_values.get_string(name);
            params.add_algo(name, tpch_params::parse_algo_value(value));
        }
    }
    return params;
}

void bench_tpch(nvbench::state& state, tpch_query qid) {
    release_all_mem();

    auto params = params_from_state(state);
    auto runner = get_query_runner(qid, params);

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
        [&](nvbench::launch&, auto& timer) {
            // FIXME: this would lead to the tables in the runner to be double-freed
            // when calling reset()
            runner->reset();

            timer.start();
            auto result = runner->run();
            timer.stop();
        });
}

} // namespace

#define FUSE_NO_FUSE_AXES {"fuse", "nofuse"}
#define ALL_GROUPBY_AXES {"hash", "sort_gftr", "sort_gftr_opt", "sort_gfur", "partition_gftr", "partition_gfur"}
#define ALL_JOIN_AXES {"hash_gfur", "sort_merge_gftr", "sort_merge_gfur", "partition_hash_gftr", "partition_hash_gfur"}
#define ALL_SORT_AXES {"skf", "no_skf"}

#define TPCH_NVBENCH(qid) \
    void bench_tpch_q##qid(nvbench::state& state) { bench_tpch(state, tpch_query::Q##qid); } \
    NVBENCH_BENCH(bench_tpch_q##qid) \
        .set_name("tpch_q" #qid) \
        .add_int64_axis("scale_factor", {1})

#define FILTER(OP_NAME) \
    .add_string_axis(OP_NAME, FUSE_NO_FUSE_AXES)

#define PROJECT(OP_NAME) \
    .add_string_axis(OP_NAME, FUSE_NO_FUSE_AXES)

#define GROUPBY(OP_NAME) \
    .add_string_axis(OP_NAME, ALL_GROUPBY_AXES)

#define SORT(OP_NAME) \
    .add_string_axis(OP_NAME, ALL_SORT_AXES)

#define JOIN(OP_NAME) \
    .add_string_axis(OP_NAME, ALL_JOIN_AXES)

// ============================================================
// TPC-H Q1: Pricing Summary Report
// ============================================================
TPCH_NVBENCH(1)
    FILTER("filter")
    PROJECT("project")
    GROUPBY("groupby")
    SORT("sort");

// ============================================================
// TPC-H Q6: Forecasting Revenue Change
// ============================================================
TPCH_NVBENCH(6)
    FILTER("filter")
    PROJECT("project");

TPCH_NVBENCH(14)
    FILTER("filter_lineitem")
    PROJECT("project_lineitem")
    JOIN("join_lineitem_part")
    PROJECT("project_join_lineitem_part");
