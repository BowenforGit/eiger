// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Per-rank synthetic data generation for the distributed-join benchmark.
//
// Each rank generates its OWN shard of a PK table and an FK table (two int
// columns: key + payload). The model:
//   - PK keys are globally unique:  rank r owns ids [r*pk_rows, (r+1)*pk_rows).
//   - FK keys reference the GLOBAL PK range [0, size*pk_rows), so after the
//     shuffle matching PK/FK rows co-locate and the local join is correct.
//
// Distribution is uniform for now (see plan §9 / §13). Implementation is
// intentionally left as a stub — the benchmark *harness* is the deliverable;
// fill this in (mirroring datagen::device::pk_uniform_id / fk_uniform_id) to
// run the real sweep.

#include <memory>
#include <utility>

#include "memory/table.hpp"
#include "types/data_type.hpp"

namespace alps::datagen {

struct GenParams {
    gamax::data_type key_type{gamax::type_id::INT32}; // key + payload column type
    gamax::size_type pk_rows_per_rank{0};             // unique build-side rows on this rank
    gamax::size_type fk_rows_per_rank{0};             // probe-side rows on this rank
    gamax::size_type ncols{2};                        // key + payload (fixed at 2 for now)
    double           match_ratio{1.0};                // fraction of FK rows that match a PK
    int              seed{42};
};

/// Generate {pk_table, fk_table} for `rank` of `size`. See header notes.
/// THROWS: not implemented yet (stub).
std::pair<std::unique_ptr<gamax::table>, std::unique_ptr<gamax::table>>
gen_shard(int rank, int size, const GenParams& params, cudaStream_t stream);

} // namespace alps::datagen
