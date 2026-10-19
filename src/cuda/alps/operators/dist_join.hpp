// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Distributed inner equi-join.
//
// One entry point, `inner_eq_join`, dispatching on DistJoinHint::strategy:
//   - shuffle:   hash-shuffle BOTH sides on the join key (identical hash, so
//                equal keys co-locate), then a local gamax inner_eq_join per rank.
//   - broadcast: all-gather the smaller side to every rank, then join it against
//                each rank's shard of the larger side (no shuffle of the large
//                side; wins when one side is small).
// Either way the result stays on the rank that produced it — nothing is gathered
// back — and the output schema is [left cols | right cols].

#include <memory>

#include "memory/table.hpp"
#include "operators/join/join.hpp"
#include "alps/context.hpp"

namespace alps::join {

struct DistJoinHint {
    gamax::join::join_hint local{};          // local-join algo / materialization / hints
    gamax::size_type       left_key_idx{0};  // join-key column in `left`
    gamax::size_type       right_key_idx{0}; // join-key column in `right`

    // Distributed join strategy. `shuffle` hash-shuffles BOTH sides so equal keys
    // co-locate (good when both sides are large / similar size). `broadcast`
    // all-gathers the smaller side to every rank and joins it against each rank's
    // shard of the larger side (good when one side is small -- no shuffle of the
    // large side). `autoselect` is a placeholder that currently runs `shuffle`
    // (TODO: pick by size once the crossover is measured).
    enum class Strategy { shuffle, broadcast, autoselect };
    Strategy strategy{Strategy::shuffle};

    // For Strategy::broadcast -- which side to all-gather. `autosel` gathers the
    // globally smaller side (by MPI_Allreduce'd row count) and swaps the local
    // join's arguments so the output schema stays [left cols | right cols]
    // regardless of which side travelled. `left`/`right` force the gathered side.
    enum class BcastSide { autosel, left, right };
    BcastSide broadcast_side{BcastSide::autosel};

    // All-to-all alignment policy for the shuffle. `autosel` size-gates per side:
    // use the misalignment-resistant shuffle_aligned() (pre/main/post split, ~2x
    // NCCL bandwidth) only when the side's global per-rank row count is large
    // enough that the win beats the 3-sweep overhead (measured crossover ~2^22
    // int32 rows/rank; see docs/ALPS_COMMUNICATION_STUDIES.md). `off`/`on` force
    // the plain / aligned shuffle. The decision uses a global (MPI_Allreduce'd)
    // count so every rank picks the same variant (collective safety).
    enum class ShuffleAlign { autosel, off, on };
    ShuffleAlign shuffle_align{ShuffleAlign::autosel};

    // Opt-in: use the static_map-based local_map_join for the per-rank local join
    // (only when local.unique_left -- a PK-FK join). It mirrors shared_map_join's
    // static_map find + cub-compact, emitting the 3-col [found_left_value, right_key,
    // right_value] output, so the NCCL shuffle/broadcast paths can be compared
    // apples-to-apples against shared_map_join (same local primitive; only the data
    // movement differs). Off -> gamax's general multiset inner_eq_join. Requires the
    // local-join inputs to be 2-col, all-int32/all-int64, with keys/values >= 0.
    bool   local_static_map{false};
    int    local_map_cg{4};      // cooperative-group size for the local map
    double local_map_lf{0.5};    // load factor for the local map

    // WIDE mode (multi-payload, int64-only): left = [key, P_L payloads], right =
    // [key, P_R payloads]. The collectives already move all columns; the per-rank
    // local join routes to local_map_join's wide path (key -> local index, then
    // gather all payloads), emitting [left payloads, right_key, right payloads].
    // The multiset local join is out of scope for the wide comparison.
    bool   wide{false};
};

/// Optional per-stage timing (milliseconds). Meaning depends on the strategy:
///   shuffle:   partition_ms = hash+partition+gather (left+right);
///              comm_ms = NCCL all-to-all (left+right); join_ms = local join.
///   broadcast: partition_ms = 0 (no partitioning); comm_ms = all-gather of the
///              small side; join_ms = local join (full small x local shard).
struct DistJoinTiming {
    float partition_ms{0.0f};
    float comm_ms{0.0f};
    float join_ms{0.0f};
};

/// Distributed inner equi-join. Returns this rank's local output shard.
std::unique_ptr<gamax::table> inner_eq_join(
    gamax::table_view   left,
    gamax::table_view   right,
    const DistJoinHint& hint,
    const Context&      ctx,
    cudaStream_t        stream,
    DistJoinTiming*     timing = nullptr);

} // namespace alps::join
