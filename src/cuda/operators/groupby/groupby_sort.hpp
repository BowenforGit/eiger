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
// downstream). Unlike `sort::groupby` there's no GFUR variant — for a
// no-agg distinct there's no per-payload sort to argsort-and-permute,
// so SORT / SORT_OPT / SORT_GFUR all collapse to the same implementation.
// `hint.optimize_sort` is also ignored for the same reason.
//
// Scope is single-key only: either `key.num_columns() == 1`, or
// `hint.enable_skf` is set AND `multikey_compress` successfully fuses
// the multi-col key into a single packed column. Other shapes (multi-
// col without SKF, multi-col where compression declines, STRING keys)
// throw. The `phys::distinct` dispatch routes SORT / SORT_OPT /
// SORT_GFUR all to this single entry point.
std::unique_ptr<gamax::table> distinct(
    gamax::table_view key,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);

} // namespace gamax::groupby::sort