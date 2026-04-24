#pragma once

#define CUB_STDERR

#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/groupby.hpp"

namespace gamax::groupby::hash {

// currently only sum and max are supported
std::unique_ptr<gamax::table> groupby(
    gamax::table_view key,
    gamax::table_view val,
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);

// Hash-based distinct: return one row per unique combination of the
// key columns. Output column shape matches input exactly. Output row
// order is insertion order (first-encounter); callers that need a
// specific order must sort downstream. Shares its dedup machinery
// with `groupby`, so STRING and multi-key inputs are handled.
std::unique_ptr<gamax::table> distinct(
    gamax::table_view key,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);
} // namespace gamax::groupby::hash