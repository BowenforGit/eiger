// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <cmath>
#include <cstdio>
#include <vector>

#include <mpi.h>

#include "tpch/distributed_benchmark_support.hpp"

namespace {

bool close(double left, double right) {
    return std::abs(left - right) <= 1e-12;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int local_fail = 0;
    auto require = [&](bool condition, const char* label) {
        if (condition) return;
        std::fprintf(stderr, "[rank %d] failed: %s\n", rank, label);
        local_fail = 1;
    };

    const double local_e2e_ms = 10.0 + 3.0 * rank;
    auto e2e = gamax::tpch::benchmark::collect_rank_timing(
        local_e2e_ms, MPI_COMM_WORLD);
    require(
        close(e2e.maximum_ms, 10.0 + 3.0 * (world - 1)),
        "end-to-end headline uses the slowest rank");
    require(
        rank == 0
            ? e2e.rank_ms.size() == static_cast<std::size_t>(world)
            : e2e.rank_ms.empty(),
        "raw samples are stored only on root");
    if (rank == 0) {
        for (int source_rank = 0; source_rank < world; ++source_rank) {
            require(
                close(
                    e2e.rank_ms[static_cast<std::size_t>(source_rank)],
                    10.0 + 3.0 * source_rank),
                "raw end-to-end samples preserve rank order");
        }
    }

    // Give each synthetic step a different bottleneck rank. This verifies that
    // the per-step maximum is derived independently rather than inherited from
    // a query-level aggregate.
    for (int step = 0; step < 3; ++step) {
        const int bottleneck = step % world;
        const double local_step_ms =
            1.0 + step + (rank == bottleneck ? 100.0 : rank);
        auto sample = gamax::tpch::benchmark::collect_rank_timing(
            local_step_ms, MPI_COMM_WORLD);
        require(
            close(sample.maximum_ms, 101.0 + step),
            "step aggregation reports its own imbalanced slowest rank");
        if (rank == 0) {
            const auto stats = gamax::tpch::benchmark::summarize(
                sample.rank_ms);
            require(
                close(stats.maximum, sample.maximum_ms),
                "step summary and all-reduced maximum agree");
        }
    }

    int global_fail = 0;
    MPI_Allreduce(
        &local_fail, &global_fail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf(
            "[distributed_benchmark_mpi_test] np=%d rank aggregation -> %s\n",
            world, global_fail ? "FAIL" : "PASS");
    }
    MPI_Finalize();
    return global_fail;
}
