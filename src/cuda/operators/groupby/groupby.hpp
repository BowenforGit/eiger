// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

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
    bool enable_skf {false}; // smart-key-fusion
    // When true, transparently downgrade `groupby_algo` to HASH if the input
    // shape (STRING key column, or multi-column key) is unsupported by the
    // requested algo. Default off so explicit hints throw on unsupported
    // shapes — useful in run_tpch_ir / tests where silent fallback would mask
    // bugs. Bench harnesses opt in per-query (e.g. Q21, whose final s_name
    // groupby is STRING-keyed but whose inner orderkey groupbys benefit from
    // axis-sweep coverage of non-HASH algos).
    bool allow_hash_fallback {false};
};

std::unique_ptr<gamax::table> groupby(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream);
}
}
