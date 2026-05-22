#pragma once

#define CUB_STDERR

#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/groupby.hpp"

namespace gamax::groupby::sort {

// currently only sum and max are supported
std::unique_ptr<gamax::table> groupby(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);

std::unique_ptr<gamax::table> groupby_gfur(
    gamax::table_view key,
    gamax::table_view val,
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);

// Sort-based distinct: one row per unique key combination, materialized
// from the original key columns. Output row order is unspecified by
// contract (callers that need a deterministic order should sort
// downstream). Mirrors `sort::groupby`'s split into a regular variant
// and a GFUR variant — the algorithms diverge enough (key-value sort +
// run-length-encode vs argsort + permutation-gather) to warrant
// separate entry points. `hint.optimize_sort` is intentionally ignored:
// the dictionary-encoded-second-sort optimization in `sort::groupby`
// only pays off when there are multiple payload sorts, which a no-agg
// distinct doesn't have.
//
// Both implementations are stubs for now and throw at call time. The
// dispatch in `phys::distinct` routes SORT/SORT_OPT here, so any bench
// cell exercising sort-based distinct will throw cleanly until the
// stubs are filled in.
std::unique_ptr<gamax::table> distinct(
    gamax::table_view key,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);

std::unique_ptr<gamax::table> distinct_gfur(
    gamax::table_view key,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);

} // namespace gamax::groupby::sort