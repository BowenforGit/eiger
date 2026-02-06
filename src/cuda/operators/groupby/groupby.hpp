#pragma once

#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"

#include "types/data_type.hpp"

namespace gamax {
namespace groupby {

enum groupby_algo {
    AUTO_ALGO, // for runtime selection
    HASH,
    SORT,
    SORT_OPT,
    SORT_GFUR,
    PARTITION,
    PARTITION_GFUR,
    UNSUPPORTED_GROUPBY_ALGO
};

struct groupby_hint {
    gamax::size_type groups_upper_bound {gamax::MAXIMUM_SIZE};
    bool zero_based_ids {false}; 
    bool optimize_sort {false}; // optimize sort for sort-based groupby
    gamax::groupby::groupby_algo groupby_algo {gamax::groupby::AUTO_ALGO};
    bool agg_count {false};
};

std::unique_ptr<gamax::table> groupby(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);
}
}