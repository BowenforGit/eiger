// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

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

// Reusable result of the hash-set insertion/group-ID phase. It is deliberately
// independent of aggregate-state columns so distributed execution can build
// group IDs as soon as all shuffled key columns arrive, while aggregate-state
// columns are still in flight.
struct group_id_plan {
    gamax::column_ptr sparse_group_ids;
    gamax::column_ptr groupid_loc;
    gamax::column_ptr leader_tid;
    gamax::size_type num_groups {0};
};

group_id_plan compute_group_ids(
    gamax::table_view key,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream);

// Complete a HASH groupby from a caller-computed group-ID plan. The plan must
// have been computed from `key`; values and requests retain the ordinary HASH
// groupby semantics. This entry point intentionally does not re-run key hashing.
std::unique_ptr<gamax::table> groupby_with_plan(
    gamax::table_view key,
    gamax::table_view val,
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    const group_id_plan& plan,
    cudaStream_t stream);

// currently only sum and max are supported
std::unique_ptr<gamax::table> groupby(
    gamax::table_view key,
    gamax::table_view val,
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream);

// Hash-based distinct: return one row per unique combination of the
// key columns. Output column shape matches input exactly. Output row
// order is insertion order (first-encounter); callers that need a
// specific order must sort downstream. Shares its dedup machinery
// with `groupby`, so STRING and multi-key inputs are handled.
std::unique_ptr<gamax::table> distinct(
    gamax::table_view key,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream);
} // namespace gamax::groupby::hash
