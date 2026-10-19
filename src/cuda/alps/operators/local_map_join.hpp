// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Local (single-GPU) PK-FK inner join via a DEVICE-scope cuco::static_map. The
// non-distributed twin of shared_map_join: build the UNIQUE left side (key->value)
// into a local static_map (cuda_allocator, thread_scope_device), find_async the
// right keys, cub-compact the matches. Returns the SAME 3-column schema as
// shared_map_join -- [found_left_value, right_key, right_value] -- so dist_join's
// NCCL shuffle/broadcast paths can use it as a drop-in local join and be compared
// apples-to-apples against shared_map_join (same static_map find+compact primitive;
// only the data movement differs). No MPI, no fabric, no barriers.
//
// Preconditions (mirror shared_map_join):
//   NARROW (wide=false): left & right have exactly 2 columns; all key/value columns
//     are the same integer type (all INT32 or all INT64); output = 3-col
//     [found_left_value, right_key, right_value].
//   WIDE (wide=true): int64 only; left = [key, P_L payloads], right = [key, P_R
//     payloads] (>=2 columns each, all INT64). The map stores key->local row index;
//     the P_L left payloads and [right_key, P_R right payloads] are GATHERED by index
//     into the output [left payloads, right_key, right payloads] (P_L+1+P_R cols).
// Either way the left (build) keys are LOCALLY unique and keys/values are >= 0
// (empty sentinel is -1).

#include <memory>

#include <cuda_runtime.h>

#include "memory/table.hpp"

namespace alps::join {

struct LocalMapJoinHint {
    gamax::size_type left_key_idx{0};   // join-key column in `left`  (payloads = the others)
    gamax::size_type right_key_idx{0};  // join-key column in `right`
    int              cg{4};             // cooperative-group size (1|2|4)
    double           lf{0.5};           // target load factor of the local map
    bool             wide{false};       // false: narrow 2-col inline; true: int64 multi-payload index+gather
};

std::unique_ptr<gamax::table> local_map_join(gamax::table_view left, gamax::table_view right,
                                             cudaStream_t stream, const LocalMapJoinHint& hint = {});

} // namespace alps::join
