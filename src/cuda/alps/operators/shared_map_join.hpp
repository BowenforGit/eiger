// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Distributed inner equi-join via the fabric-striped shared hash table
// (SharedStaticMap). The counterpart to operators/dist_join.hpp's NCCL-based
// shuffle/broadcast join, for an A/B of the two distribution mechanisms.
//
// Model: there is ONE logical hash table, its slots block-striped across every
// GPU's memory over the MNNVL fabric. Each rank inserts its LOCAL left shard
// (slots may land on any GPU), all ranks barrier, then each rank probes with its
// LOCAL right shard and keeps its local matches. No shuffle, no broadcast -- the
// fabric IS the data movement. Like dist_join, the result stays on the producing
// rank (nothing is gathered back).
//
// Preconditions:
//   - left and right have EXACTLY 2 columns each.
//   - all four key/value columns share one integer type: all INT32 or all INT64.
//   - left keys are GLOBALLY unique (PK side; static_map drops duplicate inserts),
//     and keys & values are >= 0 (the empty sentinel is -1; a value of -1 would be
//     read as a probe miss). The build-side uniqueness is checked (debug builds)
//     via map.size() == global left row count.

#include <memory>
#include <vector>

#include "memory/table.hpp"
#include "alps/context.hpp"
#include "alps/containers/fabric_region.hpp"

namespace alps::join {

struct SharedJoinHint {
    gamax::size_type left_key_idx{0};  // join-key column in `left` (payloads = the others)
    gamax::size_type right_key_idx{0}; // join-key column in `right`
    int              cg{4};            // cooperative-group size (1|2|4); 4 is empirically best
    double           lf{0.5};          // target load factor of the shared map
    bool             wide{false};      // false: narrow 2-col inline value; true: int64 multi-payload
                                       //   (slot stores a padded global left index rank*max_rows+local;
                                       //    left payloads gathered over the fabric -- needs the payload
                                       //    regions arg to shared_map_join()).
};

/// Per-phase timing (ms), to mirror DistJoinTiming for the comparison. The headline
/// join time is build_ms + probe_ms + compact_ms (+ wide: payload_copy_ms +
/// gather_ms); construct_ms (standing up + zeroing the global slot array) is
/// reported separately as amortizable setup (the slot-zero is world-x redundant over
/// the fabric -- see shared_map_bench). payload_copy_ms / gather_ms are 0 in narrow.
struct SharedJoinTiming {
    float construct_ms{0.0f};    // build the SharedStaticMap (allocate + zero slots)
    float payload_copy_ms{0.0f}; // WIDE: copy this rank's P_L left payloads into their fabric slabs
    float build_ms{0.0f};        // insert_async (local left) + cross-rank barrier
    float probe_ms{0.0f};        // find_async (local right)
    float compact_ms{0.0f};      // cub::DeviceSelect::If of the matches (wide: just the select)
    float gather_ms{0.0f};       // WIDE: multi-column gather (left payloads = fabric reads over NVLink)
};

/// Inner equi-join over the shared striped map. Returns this rank's local output
/// shard: NARROW -> 3-col [found_left_value, right_key, right_value]; WIDE -> P_L+1+P_R
/// col [left payloads, right_key, right payloads] (left_key omitted -- for an equi-join
/// it equals right_key). `region` is a reusable fabric backing (construct once via
/// FabricStripedRegion(ctx)); the map is built per call from it. Fully collective:
/// every rank must call it (in lockstep -- preconditions are validated collectively,
/// so invalid input throws on every rank rather than deadlocking).
///
/// WIDE MODE (hint.wide): int64 only; left = [key, P_L payloads], right = [key, P_R
/// payloads]. The slot stores a padded global left index (rank*max_rows+local); the
/// P_L left payloads must be readable by every rank over the fabric, so pass P_L
/// caller-owned reusable regions in `left_payload_regions` (>= P_L pointers). Each is
/// acquire()'d to `world*max_rows*sizeof(int64)`, this rank memcpy's its local payload
/// column into its slab, and the compaction GATHERs left payloads from them by index
/// (remote reads). Ignored (may be empty) in narrow mode.
///
/// REGION REUSE CONSTRAINT: a FabricStripedRegion is sized on its FIRST acquire and
/// never grows -- a later acquire needing more bytes THROWS "request exceeds the
/// created region". The per-call slot map needs ~ global_left_rows / lf slots of
/// sizeof(pair<T,T>) bytes (int64 keys/values are 2x the int32 footprint); each
/// payload region needs world*max_rows*8. So a given region may only be reused for
/// joins whose footprint is <= the first join's. To A/B across element types or
/// growing sizes, use fresh regions per configuration (or pre-size on the largest).
std::unique_ptr<gamax::table> shared_map_join(
    gamax::table_view              left,
    gamax::table_view              right,
    containers::FabricStripedRegion& region,
    const Context&                 ctx,
    cudaStream_t                   stream,
    const SharedJoinHint&          hint   = {},
    SharedJoinTiming*              timing = nullptr,
    const std::vector<containers::FabricStripedRegion*>& left_payload_regions = {});

} // namespace alps::join
