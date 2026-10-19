// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "inner_join.hpp"

#include <memory>
#include <limits>
#include <cuda.h>
#include <cuda/atomic>
#include <type_traits>

#include "join.hpp"
#include "materialize.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "containers/multiset.cuh"
#include "types/sentinel.hpp"

namespace gamax::join {
namespace {
    using tid_type = gamax::size_type;

    struct hash_join_indices_impl {
        template<typename Key, std::enable_if_t<sizeof(Key) == sizeof(cuda::atomic<Key>) && cuco::is_bitwise_comparable_v<Key>, int> = 0>
        gamax::join::EqJoinMatches operator()(
            gamax::column_view left_key,
            gamax::column_view right_key,
            gamax::join::join_hint const& hint,
            cudaStream_t stream,
            const BeforeProbeHook& before_probe)
        {
            if (left_key.size() == 0 || right_key.size() == 0) {
                if (before_probe) before_probe(stream);
                return EqJoinMatches{
                    gamax::make_column<tid_type>(0, stream),
                    gamax::make_column<tid_type>(0, stream)};
            }

            // Build multiset with left (build) side keys
            rmm_allocator_handle<tid_type> allocator{stream};
            auto set = gamax::containers::multiset<Key, tid_type>(
                left_key,
                0.25, // use a default load factor of 0.25
                gamax::sentinel<tid_type>(),
                allocator.get(),
                stream);

            set.insert_with_tids(stream);

            // Size the probe output buffer.
            //
            // `multiset::probe` allocates two tid columns (one
            // left-side tid per match, one right-side tid per match)
            // eagerly at this size; the actual match count is set via
            // `set_size` afterwards, but the physical backing buffers
            // are not trimmed. At the `get_ieq_join_output_alloc_size`
            // default of MAX_JOIN_OUTPUT_SIZE = 2^30 rows, that's
            // **8 GB per tid column × 2 = 16 GB transient per join**
            // — catastrophic when any query chains joins. Q5 at SF10
            // was running a 94 GiB pool dry.
            //
            // ── Two callers, two contracts ────────────────────────
            //
            // 1. Hint-carrying callers (`hint.max_out` set, or
            //    `unique_left` / `unique_right`) supply their own
            //    tight bound and skip this fallback. This is the
            //    path the nvbench microbenchmarks
            //    (`benchmarks/join/ieq_join_nvbench.cu`) take, so
            //    raw hash-join throughput numbers are unaffected.
            //
            // 2. Hint-free callers (e.g. `phys::eq_join`) would
            //    otherwise hit MAX_JOIN_OUTPUT_SIZE. We instead
            //    probe the multiset's cheap `count()` once to
            //    learn `n_matches` exactly, then size the tid
            //    buffers to `n_matches`.
            //
            // ── Cost the fallback adds ────────────────────────────
            //
            // `multiset::count` is a second pass over the same
            // build hash set with the same probe keys — same
            // asymptotic work as `probe`, minus the tid writes.
            // Empirically ~same wall time as a single probe pass,
            // so ~2× the hash-join hot path vs the hint-carrying
            // path. That cost is worth it at TPC-H scales where
            // the over-allocated tid buffers would either OOM or
            // evict payload-side intermediates; microbenchmarks
            // comparing against external engines (cuDF etc.)
            // should keep supplying a tight hint so the
            // comparison stays apples-to-apples. **Do not remove
            // the `hint.max_out` / `unique_*` settings in
            // `benchmarks/join/ieq_join_nvbench.cu` without also
            // lifting the count-then-probe protection**, or
            // benchmark numbers will silently regress by ~2×.
            constexpr gamax::size_type kUnbounded =
                std::numeric_limits<gamax::size_type>::max();
            const gamax::size_type bound_l =
                hint.unique_left ? right_key.size() : kUnbounded;
            const gamax::size_type bound_r =
                hint.unique_right ? left_key.size() : kUnbounded;
            const gamax::size_type bound = std::min(bound_l, bound_r);
            auto max_output_size = hint.max_out != 0
                ? hint.max_out
                : (bound == kUnbounded ? gamax::join::MAX_JOIN_OUTPUT_SIZE : bound);
            const bool have_tight_bound =
                hint.max_out != 0 || hint.unique_left || hint.unique_right;
            if (!have_tight_bound) {
                max_output_size = static_cast<gamax::size_type>(
                    set.count(right_key, stream));
            }

            if (before_probe) before_probe(stream);

            // Probe with right (probe) side keys, returns (build_tids, probe_tids) = (left_tids, right_tids)
            auto [res_left_tids, res_right_tids] =
                set.probe(right_key, max_output_size, stream);
            return EqJoinMatches{
                std::move(res_left_tids), std::move(res_right_tids)};
        }   

        template<typename Key, std::enable_if_t<sizeof(Key) != sizeof(cuda::atomic<Key>) || !cuco::is_bitwise_comparable_v<Key>, int> = 0>
        gamax::join::EqJoinMatches operator()(
            gamax::column_view left_key,
            gamax::column_view right_key,
            gamax::join::join_hint const& hint,
            cudaStream_t stream,
            const BeforeProbeHook&)
        {
            if constexpr (sizeof(Key) != sizeof(cuda::atomic<Key>)) {
                throw std::runtime_error("Hash join only supports atomic types");
            } else {
                throw std::runtime_error("Hash join only supports bitwise comparable types");
            }
        } 
    };
} // namespace

EqJoinMatches eq_join_indices(
    gamax::column_view left_key,
    gamax::column_view right_key,
    gamax::join::join_hint const& hint,
    cudaStream_t stream)
{
    return eq_join_indices(left_key, right_key, hint, stream, {});
}

EqJoinMatches eq_join_indices(
    gamax::column_view left_key,
    gamax::column_view right_key,
    gamax::join::join_hint const& hint,
    cudaStream_t stream,
    const BeforeProbeHook& before_probe)
{
    if (hint.join_algo != gamax::join::HASH) {
        throw std::invalid_argument(
            "eq_join_indices only supports the HASH join algorithm");
    }
    if (left_key.type() != right_key.type()) {
        throw std::invalid_argument(
            "eq_join_indices requires identical key types");
    }
    return gamax::type_dispatcher(
        left_key.type(), hash_join_indices_impl{},
        left_key, right_key, hint, stream, before_probe);
}

namespace detail {

std::unique_ptr<gamax::table> ieq_hash_join(
    gamax::table_view left, 
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream)
{
    auto matches = gamax::join::eq_join_indices(
        left[0], right[0], hint, stream);

    auto result = std::make_unique<gamax::table>();
    result->add_column(simple_materialize(
        left[0], matches.left_indices->view(), stream));
    for (auto column = 1; column < left.num_columns(); ++column) {
        result->add_column(simple_materialize(
            left[column], matches.left_indices->view(), stream));
    }
    // Every use of the left match indices has now been enqueued on `stream`.
    // Release the potentially multi-gigabyte buffer before allocating the
    // right-side outputs so a stream-ordered resource can reuse it once those
    // left gathers complete. This preserves the pre-eq_join_indices peak-memory
    // behavior of the legacy HASH join.
    matches.left_indices.reset();
    for (auto column = 1; column < right.num_columns(); ++column) {
        result->add_column(simple_materialize(
            right[column], matches.right_indices->view(), stream));
    }
    cudaStreamSynchronize(stream);
    return result;
}
} // namespace detail
} // namespace gamax::join
