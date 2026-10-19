// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_dist_join_test — MPI correctness test for alps::join::inner_eq_join.
//
// Run under mpirun (-np 2 / -np 4). Each rank builds a small PK shard (globally
// unique keys, distinct payloads) and an FK shard (keys spanning the global
// range, ~half matching). It runs the distributed shuffle+join, then gathers
// both the inputs and the distributed output to rank 0 and compares the output
// against a CPU reference inner join. Returns nonzero on mismatch.
//
// Payloads are constructed so pk_payload < fk_payload always, which makes the
// (key, min, max) tuple comparison robust to output column ordering.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/join/join.hpp"

#include "alps/context.hpp"
#include "alps/error.hpp"
#include "alps/operators/dist_join.hpp"
#include "alps/tests/dist_test_util.hpp"

namespace {

constexpr gamax::size_type kPkPerRank = 2000;
constexpr gamax::size_type kFkPerRank = 4000;
constexpr int32_t          kPkPayBase = 1000000;
constexpr int32_t          kFkPayBase = 2000000;

// Table/host/gather helpers are shared with shared_map_join_test via the templated
// utilities in dist_test_util.hpp (int32 instantiations here).
using alps::testutil::col_to_host;
using alps::testutil::gather_to_root;
using alps::testutil::make_table;

} // namespace

int main(int argc, char** argv) {
    alps::configure_process(); // allocator + MPI env defaults; before MPI_Init
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    const int32_t global_pk = static_cast<int32_t>(static_cast<gamax::size_type>(ctx.size) * kPkPerRank);

    // PK shard: keys [rank*kPkPerRank, (rank+1)*kPkPerRank), distinct payloads.
    std::vector<int32_t> pk_key(kPkPerRank), pk_pay(kPkPerRank);
    for (gamax::size_type i = 0; i < kPkPerRank; ++i) {
        int32_t k = static_cast<int32_t>(ctx.rank * kPkPerRank + i);
        pk_key[i] = k;
        pk_pay[i] = kPkPayBase + k;
    }

    // FK shard: keys uniform in [0, 2*global_pk) (~half match), distinct payloads.
    std::vector<int32_t> fk_key(kFkPerRank), fk_pay(kFkPerRank);
    uint64_t s = 0xD1B54A32D192ED03ull ^ (static_cast<uint64_t>(ctx.rank + 1) * 0x9e3779b97f4a7c15ull);
    for (gamax::size_type j = 0; j < kFkPerRank; ++j) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        fk_key[j] = static_cast<int32_t>((s >> 33) % (2 * global_pk));
        fk_pay[j] = kFkPayBase + static_cast<int32_t>(ctx.rank * kFkPerRank + j);
    }

    auto pk = make_table(pk_key, pk_pay, ctx.stream);
    auto fk = make_table(fk_key, fk_pay, ctx.stream);

    // Gather inputs to rank 0 once and build the CPU reference inner join, then
    // verify BOTH the gated (default) and forced-aligned shuffle paths produce it.
    auto all_pk_key = gather_to_root(pk_key, ctx);
    auto all_pk_pay = gather_to_root(pk_pay, ctx);
    auto all_fk_key = gather_to_root(fk_key, ctx);
    auto all_fk_pay = gather_to_root(fk_pay, ctx);

    using Tup = std::array<int32_t, 3>;
    std::vector<Tup> ref;
    if (ctx.rank == 0) {
        std::unordered_map<int32_t, int32_t> pkmap;
        pkmap.reserve(all_pk_key.size() * 2);
        for (size_t i = 0; i < all_pk_key.size(); ++i) pkmap[all_pk_key[i]] = all_pk_pay[i];
        ref.reserve(all_fk_key.size());
        for (size_t j = 0; j < all_fk_key.size(); ++j) {
            auto it = pkmap.find(all_fk_key[j]);
            if (it != pkmap.end()) {
                int32_t pay = it->second;
                ref.push_back({all_fk_key[j], std::min(pay, all_fk_pay[j]), std::max(pay, all_fk_pay[j])});
            }
        }
        std::sort(ref.begin(), ref.end());
    }

    int local_fail = 0;

    // Run one distributed join with the given alignment policy, gather its output
    // tuples to rank 0, and compare against `ref`. All ranks call this in lockstep
    // (the collectives inside require it).
    auto run_variant = [&](alps::join::DistJoinHint::Strategy strat,
                           alps::join::DistJoinHint::ShuffleAlign sa, const char* label) {
        alps::join::DistJoinHint hint;
        hint.local.join_algo   = gamax::join::HASH;
        hint.local.mat_algo    = gamax::join::GFUR;
        hint.local.unique_left = true; // PK build side
        hint.local.max_out     = 0;
        hint.left_key_idx  = 0;
        hint.right_key_idx = 0;
        hint.strategy      = strat;
        hint.shuffle_align = sa;

        auto out = alps::join::inner_eq_join(pk->view(), fk->view(), hint, ctx, ctx.stream);
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

        std::vector<int32_t> d_key, d_min, d_max;
        if (out->num_columns() == 3) {
            auto o0 = col_to_host<int32_t>(out->get_column(0), ctx.stream);
            auto o1 = col_to_host<int32_t>(out->get_column(1), ctx.stream);
            auto o2 = col_to_host<int32_t>(out->get_column(2), ctx.stream);
            d_key.resize(o0.size()); d_min.resize(o0.size()); d_max.resize(o0.size());
            for (size_t r = 0; r < o0.size(); ++r) {
                d_key[r] = o0[r];
                d_min[r] = std::min(o1[r], o2[r]);
                d_max[r] = std::max(o1[r], o2[r]);
            }
        } else {
            std::fprintf(stderr, "[rank %d] %s: expected 3 output columns, got %lld\n",
                         ctx.rank, label, static_cast<long long>(out->num_columns()));
            local_fail = 1;
        }

        auto all_d_key = gather_to_root(d_key, ctx);
        auto all_d_min = gather_to_root(d_min, ctx);
        auto all_d_max = gather_to_root(d_max, ctx);

        if (ctx.rank == 0) {
            std::vector<Tup> got(all_d_key.size());
            for (size_t r = 0; r < all_d_key.size(); ++r) got[r] = {all_d_key[r], all_d_min[r], all_d_max[r]};
            std::sort(got.begin(), got.end());
            if (got.size() != ref.size()) {
                std::fprintf(stderr, "[rank 0] %s row count mismatch: ref=%zu dist=%zu\n",
                             label, ref.size(), got.size());
                local_fail = 1;
            } else if (got != ref) {
                std::fprintf(stderr, "[rank 0] %s tuple mismatch (counts equal, contents differ)\n", label);
                local_fail = 1;
            } else {
                std::printf("[dist_join_test] np=%d %s output rows=%zu -> PASS\n", ctx.size, label, got.size());
            }
        }
    };

    using Strat = alps::join::DistJoinHint::Strategy;
    using Align = alps::join::DistJoinHint::ShuffleAlign;
    run_variant(Strat::shuffle,   Align::autosel, "shuffle/gated  ");
    run_variant(Strat::shuffle,   Align::on,      "shuffle/aligned");
    run_variant(Strat::broadcast, Align::autosel, "broadcast      ");

    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    if (ctx.rank == 0 && global_fail) std::printf("[dist_join_test] np=%d -> FAIL\n", ctx.size);

    alps::destroy_context(ctx);
    MPI_Finalize();
    return global_fail;
}
