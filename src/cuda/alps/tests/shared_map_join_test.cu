// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_shared_map_join_test — MPI correctness test for alps::join::shared_map_join
// (the fabric-striped shared-hash-table distributed join).
//
// Run under mpirun (-np 2/3/4 single-node, or 8/16 cross-node). Each rank builds a
// globally-unique PK shard (the build/left side) and an FK shard (the probe/right
// side, keys spanning the global range so ~half match). It runs shared_map_join,
// gathers the local output shards to rank 0, and compares against a CPU reference
// inner join — the SAME oracle dist_join_test uses, so a pass proves the shared-map
// join agrees with the NCCL dist_join. Covers both int32 and int64. Nonzero on any
// mismatch.
//
// shared_map_join output is [found_left_value, right_key, right_value], so the join
// key is column 1, the left payload column 0, the right payload column 2. The
// oracle compares the (right_key, found_left_value, right_value) triple DIRECTLY
// (not a min/max wrap): this catches a wrong looked-up left value regardless of the
// PK payload encoding. PK payloads (base 1e6) and FK payloads (base 2e6) and all
// keys are >= 0, satisfying the sentinel (-1) precondition.
//
// Each dtype run uses its OWN FabricStripedRegion: the region is sized on first
// acquire and never grows, and int64 slots are 2x the int32 footprint, so reusing
// one region across the two runs would throw at non-trivial sizes.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/context.hpp"
#include "alps/error.hpp"
#include "alps/containers/fabric_region.hpp"
#include "alps/operators/shared_map_join.hpp"
#include "alps/tests/dist_test_util.hpp"

namespace {

using alps::testutil::col_to_host;
using alps::testutil::gather_to_root;
using alps::testutil::make_table;

constexpr gamax::size_type kPkPerRank = 2000;
constexpr gamax::size_type kFkPerRank = 4000;
constexpr long long        kPkPayBase = 1000000;
constexpr long long        kFkPayBase = 2000000;

template <typename T>
int run_dtype(const alps::Context& ctx, const char* tname) {
    const long long global_pk = static_cast<long long>(ctx.size) * kPkPerRank;

    // PK (left/build): globally-unique keys [rank*kPkPerRank, (rank+1)*kPkPerRank).
    std::vector<T> pk_key(kPkPerRank), pk_pay(kPkPerRank);
    for (gamax::size_type i = 0; i < kPkPerRank; ++i) {
        long long k = static_cast<long long>(ctx.rank) * kPkPerRank + i;
        pk_key[i] = static_cast<T>(k);
        pk_pay[i] = static_cast<T>(kPkPayBase + k);
    }
    // FK (right/probe): keys uniform in [0, 2*global_pk) (~half match), unique payloads.
    std::vector<T> fk_key(kFkPerRank), fk_pay(kFkPerRank);
    uint64_t s = 0xD1B54A32D192ED03ull ^ (static_cast<uint64_t>(ctx.rank + 1) * 0x9e3779b97f4a7c15ull);
    for (gamax::size_type j = 0; j < kFkPerRank; ++j) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        fk_key[j] = static_cast<T>((s >> 33) % static_cast<uint64_t>(2 * global_pk));
        fk_pay[j] = static_cast<T>(kFkPayBase + static_cast<long long>(ctx.rank) * kFkPerRank + j);
    }

    auto pk = make_table<T>(pk_key, pk_pay, ctx.stream);
    auto fk = make_table<T>(fk_key, fk_pay, ctx.stream);

    // CPU reference inner join: tuple (key, left_value, right_value) over all matches.
    auto all_pk_key = gather_to_root<T>(pk_key, ctx);
    auto all_pk_pay = gather_to_root<T>(pk_pay, ctx);
    auto all_fk_key = gather_to_root<T>(fk_key, ctx);
    auto all_fk_pay = gather_to_root<T>(fk_pay, ctx);

    using Tup = std::array<int64_t, 3>; // (key, left_value, right_value)
    std::vector<Tup> ref;
    if (ctx.rank == 0) {
        std::unordered_map<int64_t, int64_t> pkmap;
        pkmap.reserve(all_pk_key.size() * 2);
        for (size_t i = 0; i < all_pk_key.size(); ++i)
            pkmap[static_cast<int64_t>(all_pk_key[i])] = static_cast<int64_t>(all_pk_pay[i]);
        ref.reserve(all_fk_key.size());
        for (size_t j = 0; j < all_fk_key.size(); ++j) {
            auto it = pkmap.find(static_cast<int64_t>(all_fk_key[j]));
            if (it != pkmap.end())
                ref.push_back({static_cast<int64_t>(all_fk_key[j]), it->second,
                               static_cast<int64_t>(all_fk_pay[j])});
        }
        std::sort(ref.begin(), ref.end());
    }

    // Run the shared-map join (collective). Each dtype gets its own region.
    alps::join::SharedJoinHint hint;
    hint.left_key_idx  = 0;
    hint.right_key_idx = 0;
    hint.cg            = 4;   // empirically best
    hint.lf            = 0.5;
    alps::join::SharedJoinTiming timing{};

    int local_fail = 0;
    std::vector<int64_t> d_key, d_left, d_right;
    {
        alps::containers::FabricStripedRegion region(ctx);
        auto out = alps::join::shared_map_join(pk->view(), fk->view(), region,
                                               ctx, ctx.stream, hint, &timing);
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

        if (out->num_columns() == 3) {
            auto o0 = col_to_host<T>(out->get_column(0), ctx.stream); // found left value
            auto o1 = col_to_host<T>(out->get_column(1), ctx.stream); // right key (== join key)
            auto o2 = col_to_host<T>(out->get_column(2), ctx.stream); // right value
            d_key.resize(o1.size()); d_left.resize(o1.size()); d_right.resize(o1.size());
            for (size_t r = 0; r < o1.size(); ++r) {
                d_key[r]   = static_cast<int64_t>(o1[r]);
                d_left[r]  = static_cast<int64_t>(o0[r]);
                d_right[r] = static_cast<int64_t>(o2[r]);
            }
        } else {
            std::fprintf(stderr, "[rank %d] %s: expected 3 output columns, got %lld\n",
                         ctx.rank, tname, static_cast<long long>(out->num_columns()));
            local_fail = 1;
        }
        region.free(); // collective; after the output table is materialized
    }

    auto all_key   = gather_to_root<int64_t>(d_key, ctx);
    auto all_left  = gather_to_root<int64_t>(d_left, ctx);
    auto all_right = gather_to_root<int64_t>(d_right, ctx);

    if (ctx.rank == 0) {
        std::vector<Tup> got(all_key.size());
        for (size_t r = 0; r < all_key.size(); ++r) got[r] = {all_key[r], all_left[r], all_right[r]};
        std::sort(got.begin(), got.end());
        if (got.size() != ref.size()) {
            std::fprintf(stderr, "[rank 0] %s row-count mismatch: ref=%zu shared_map=%zu\n",
                         tname, ref.size(), got.size());
            local_fail = 1;
        } else if (got != ref) {
            std::fprintf(stderr, "[rank 0] %s tuple mismatch (counts equal, contents differ)\n", tname);
            local_fail = 1;
        } else {
            std::printf("[shared_map_join_test] np=%d %-5s out rows=%zu  "
                        "(construct=%.3f build=%.3f probe=%.3f compact=%.3f ms) -> PASS\n",
                        ctx.size, tname, got.size(),
                        timing.construct_ms, timing.build_ms, timing.probe_ms, timing.compact_ms);
        }
    }
    return local_fail;
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process(); // allocator + MPI env defaults; before MPI_Init
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    int local_fail = 0;
    local_fail |= run_dtype<int32_t>(ctx, "i32");
    local_fail |= run_dtype<int64_t>(ctx, "i64");

    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    if (ctx.rank == 0 && global_fail) std::printf("[shared_map_join_test] np=%d -> FAIL\n", ctx.size);

    alps::destroy_context(ctx);
    MPI_Finalize();
    return global_fail;
}
