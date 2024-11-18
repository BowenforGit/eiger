#pragma once

#define CUB_STDERR

#include <memory>
#include <vector>
#include <cuda.h>

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
} // namespace gamax::groupby::hash