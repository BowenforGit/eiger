#pragma once

#include <memory>
#include <cuda_runtime_api.h>

#include "memory/table.hpp"
#include "operators/join/join.hpp"

namespace gamax::join {

// `LEFT ANTI JOIN` — returns tuples from `left` that have NO match in
// `right` on their join key. Dual of `semi_join` (which keeps the ones
// that DO match). Used in TPC-H Q16/Q21/Q22.
//
// Shape: output is a subset of `left`'s columns, in `left`'s column
// order. Right's columns are discarded — the anti-join only needs
// right for membership testing.
std::unique_ptr<gamax::table> anti_join(
    gamax::table_view left,
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream = cudaStreamDefault);

namespace detail {
    // Hash-based anti-join. Reuses the membership-mask helper that
    // semi_hash_join exposes; the only difference is negating the mask
    // before compacting `left`.
    std::unique_ptr<gamax::table> anti_hash_join(
        gamax::table_view left,
        gamax::table_view right,
        gamax::join::join_hint const& hint,
        cudaStream_t stream = cudaStreamDefault);
} // namespace detail

} // namespace gamax::join
