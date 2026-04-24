#include "inner_join.hpp"

#include <memory>
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
namespace detail {

namespace {
    using tid_type = gamax::size_type;

    struct hash_join_impl {
        template<typename Key, std::enable_if_t<sizeof(Key) == sizeof(cuda::atomic<Key>) && cuco::is_bitwise_comparable_v<Key>, int> = 0>
        std::unique_ptr<gamax::table> operator()(
            gamax::table_view left,
            gamax::table_view right,
            gamax::join::join_hint const& hint,
            cudaStream_t stream)
        {
            // Build multiset with left (build) side keys
            auto set = gamax::containers::multiset<Key, tid_type>(
                left[0],
                0.25, // use a default load factor of 0.25
                gamax::sentinel<tid_type>(),
                *get_thrust_allocator<char>(stream),
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
            auto max_output_size = get_ieq_join_output_alloc_size(left, right, hint);
            const bool have_tight_bound =
                hint.max_out != 0 || hint.unique_left || hint.unique_right;
            if (!have_tight_bound) {
                max_output_size = static_cast<gamax::size_type>(
                    set.count(right[0], stream));
            }

            // Probe with right (probe) side keys, returns (build_tids, probe_tids) = (left_tids, right_tids)
            auto [res_left_tids, res_right_tids] = set.probe(right[0], max_output_size, stream);

            // materialize
            auto res_table = std::make_unique<gamax::table>();
            
            // Materialize join key from left side using left_tids
            res_table->add_column(simple_materialize(left[0], res_left_tids->view(), stream));
            
            for(auto i = 1; i < left.num_columns(); i++) {
                res_table->add_column(simple_materialize(left[i], res_left_tids->view(), stream));
            }
            res_left_tids.reset(nullptr);
            
            for(auto i = 1; i < right.num_columns(); i++) {
                res_table->add_column(simple_materialize(right[i], res_right_tids->view(), stream));
            }
            
            cudaStreamSynchronize(stream);
            
            return res_table;
        }   

        template<typename Key, std::enable_if_t<sizeof(Key) != sizeof(cuda::atomic<Key>) || !cuco::is_bitwise_comparable_v<Key>, int> = 0>
        std::unique_ptr<gamax::table> operator()(
            gamax::table_view left, 
            gamax::table_view right,
            gamax::join::join_hint const& hint,
            cudaStream_t stream)
        {
            if constexpr (sizeof(Key) != sizeof(cuda::atomic<Key>)) {
                throw std::runtime_error("Hash join only supports atomic types");
            } else {
                throw std::runtime_error("Hash join only supports bitwise comparable types");
            }
        } 
    };
} // namespace

std::unique_ptr<gamax::table> ieq_hash_join(
    gamax::table_view left, 
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream)
{
    return gamax::type_dispatcher(
        left[0].type(), 
        hash_join_impl{}, 
        left, 
        right, 
        hint, 
        stream);
}
} // namespace detail
} // namespace gamax::join