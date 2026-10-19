// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_shuffle_test — MPI correctness test for alps::collectives::shuffle.
//
// Run under mpirun (-np 2 / -np 4). Each rank builds a small 2-column table
// with random keys spanning the global key range, shuffles it, and checks:
//   (1) conservation — global received rows == global sent rows;
//   (2) locality     — every received key re-hashes to THIS rank.
// Returns nonzero on any rank's failure (reduced across ranks).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "memory/column.hpp"
#include "memory/table.hpp"

#include "alps/collectives/hash.cuh"
#include "alps/collectives/shuffle.hpp"
#include "alps/context.hpp"
#include "alps/error.hpp"

namespace {

constexpr gamax::size_type kRowsPerRank = 4096;
constexpr int32_t          kKeyRange    = 1 << 20; // keys span [0, kKeyRange)

std::unique_ptr<gamax::table> make_i32_table(const std::vector<int32_t>& key,
                                             const std::vector<int32_t>& payload,
                                             cudaStream_t stream) {
    std::vector<std::unique_ptr<gamax::column>> cols;
    cols.push_back(gamax::make_column_from_host(const_cast<int32_t*>(key.data()),
                                                static_cast<gamax::size_type>(key.size()), stream));
    cols.push_back(gamax::make_column_from_host(const_cast<int32_t*>(payload.data()),
                                                static_cast<gamax::size_type>(payload.size()), stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return gamax::make_table(std::move(cols));
}

std::vector<int32_t> col_to_host(const gamax::column& c, cudaStream_t stream) {
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    std::vector<int32_t> h(c.size());
    if (c.size() > 0) {
        ALPS_CUDACHECK(cudaMemcpy(h.data(), c.data<int32_t>(),
                                  c.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    }
    return h;
}

// Concatenate each rank's int32 vector onto rank 0 (variable counts).
std::vector<int32_t> gather_to_root(const std::vector<int32_t>& local, const alps::Context& ctx) {
    int n = static_cast<int>(local.size());
    std::vector<int> counts(ctx.size), displs(ctx.size);
    ALPS_MPICHECK(MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, ctx.mpi_comm));
    std::vector<int32_t> out;
    int total = 0;
    if (ctx.rank == 0) {
        for (int r = 0; r < ctx.size; ++r) { displs[r] = total; total += counts[r]; }
        out.resize(total);
    }
    ALPS_MPICHECK(MPI_Gatherv(local.data(), n, MPI_INT,
                              out.data(), counts.data(), displs.data(), MPI_INT, 0, ctx.mpi_comm));
    return out;
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process(); // allocator + MPI env defaults; before MPI_Init
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    int local_fail = 0;

    // Build this rank's input: deterministic per-rank LCG over the key range.
    std::vector<int32_t> key(kRowsPerRank), payload(kRowsPerRank);
    uint64_t s = 0x9e3779b97f4a7c15ull ^ (static_cast<uint64_t>(ctx.rank) * 0x100000001b3ull + 1);
    for (gamax::size_type i = 0; i < kRowsPerRank; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        key[i]     = static_cast<int32_t>((s >> 33) % kKeyRange);
        payload[i] = static_cast<int32_t>(ctx.rank * kRowsPerRank + i); // origin tag
    }

    auto in = make_i32_table(key, payload, ctx.stream);
    auto out = alps::collectives::shuffle(in->view(), /*key_idx=*/0, ctx, ctx.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

    const gamax::size_type recv_rows = out->num_rows();

    // (1) conservation: total received == total sent.
    long long sent = kRowsPerRank, recv = recv_rows, total_sent = 0, total_recv = 0;
    ALPS_MPICHECK(MPI_Allreduce(&sent, &total_sent, 1, MPI_LONG_LONG, MPI_SUM, ctx.mpi_comm));
    ALPS_MPICHECK(MPI_Allreduce(&recv, &total_recv, 1, MPI_LONG_LONG, MPI_SUM, ctx.mpi_comm));
    if (total_sent != total_recv) {
        std::fprintf(stderr, "[rank %d] conservation FAIL: sent=%lld recv=%lld\n",
                     ctx.rank, total_sent, total_recv);
        local_fail = 1;
    }

    // (2) locality: every received key must route to this rank.
    const int world = ctx.size;
    auto recv_keys = col_to_host(out->get_column(0), ctx.stream);
    gamax::size_type bad = 0;
    for (int32_t k : recv_keys) {
        if (alps::collectives::dest_rank<int32_t>(k, world) != ctx.rank) ++bad;
    }
    if (bad != 0) {
        std::fprintf(stderr, "[rank %d] locality FAIL: %lld/%zu keys misrouted\n",
                     ctx.rank, static_cast<long long>(bad), recv_keys.size());
        local_fail = 1;
    }

    // --- CE backend: drop-in copy-engine all-to-all. Same delivered rows AND
    //     same row order as the NCCL path, so apply the identical conservation +
    //     locality + per-row (key,payload) checks against the NCCL output. ---
    auto outc = alps::collectives::shuffle(in->view(), /*key_idx=*/0, ctx, ctx.stream,
                                           /*timing=*/nullptr, alps::collectives::Backend::CE);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    {
        long long c_recv = outc->num_rows(), c_total_recv = 0;
        ALPS_MPICHECK(MPI_Allreduce(&c_recv, &c_total_recv, 1, MPI_LONG_LONG, MPI_SUM, ctx.mpi_comm));
        if (c_total_recv != total_sent) {
            std::fprintf(stderr, "[rank %d] CE conservation FAIL: sent=%lld recv=%lld\n",
                         ctx.rank, total_sent, c_total_recv);
            local_fail = 1;
        }
        auto c_keys = col_to_host(outc->get_column(0), ctx.stream);
        gamax::size_type c_bad = 0;
        for (int32_t k : c_keys)
            if (alps::collectives::dest_rank<int32_t>(k, world) != ctx.rank) ++c_bad;
        if (c_bad != 0) {
            std::fprintf(stderr, "[rank %d] CE locality FAIL: %lld/%zu keys misrouted\n",
                         ctx.rank, static_cast<long long>(c_bad), c_keys.size());
            local_fail = 1;
        }
        // CE preserves row order -> must match the NCCL output element-for-element.
        if (outc->num_rows() == out->num_rows()) {
            auto c_pays = col_to_host(outc->get_column(1), ctx.stream);
            auto n_keys = col_to_host(out->get_column(0), ctx.stream);
            auto n_pays = col_to_host(out->get_column(1), ctx.stream);
            for (size_t i = 0; i < c_keys.size(); ++i) {
                if (c_keys[i] != n_keys[i] || c_pays[i] != n_pays[i]) {
                    std::fprintf(stderr, "[rank %d] CE vs NCCL row %zu mismatch: "
                                 "CE %d/%d NCCL %d/%d\n", ctx.rank, i,
                                 c_keys[i], c_pays[i], n_keys[i], n_pays[i]);
                    local_fail = 1; break;
                }
            }
        } else {
            std::fprintf(stderr, "[rank %d] CE row count %lld != NCCL %lld\n", ctx.rank,
                         (long long)outc->num_rows(), (long long)out->num_rows());
            local_fail = 1;
        }
    }

    // --- aligned variant: same delivered rows, order NOT preserved. Checks:
    //     (2') locality per rank, and (3) the GLOBAL multiset of (key,payload)
    //     pairs is conserved exactly (catches lost/duplicated rows AND any
    //     key/payload column desync from the pre/main/post split). ---
    auto outa = alps::collectives::shuffle_aligned(in->view(), /*key_idx=*/0, ctx, ctx.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

    auto a_keys = col_to_host(outa->get_column(0), ctx.stream);
    auto a_pays = col_to_host(outa->get_column(1), ctx.stream);

    gamax::size_type a_bad = 0;
    for (int32_t k : a_keys)
        if (alps::collectives::dest_rank<int32_t>(k, world) != ctx.rank) ++a_bad;
    if (a_bad != 0) {
        std::fprintf(stderr, "[rank %d] aligned locality FAIL: %lld/%zu keys misrouted\n",
                     ctx.rank, static_cast<long long>(a_bad), a_keys.size());
        local_fail = 1;
    }

    // Gather input + aligned-output (key,payload) to root and compare multisets.
    auto in_k  = gather_to_root(key, ctx);
    auto in_p  = gather_to_root(payload, ctx);
    auto out_k = gather_to_root(a_keys, ctx);
    auto out_p = gather_to_root(a_pays, ctx);
    if (ctx.rank == 0) {
        auto zip_sort = [](const std::vector<int32_t>& ks, const std::vector<int32_t>& ps) {
            std::vector<std::pair<int32_t, int32_t>> v(ks.size());
            for (size_t i = 0; i < ks.size(); ++i) v[i] = {ks[i], ps[i]};
            std::sort(v.begin(), v.end());
            return v;
        };
        auto a = zip_sort(in_k, in_p);
        auto b = zip_sort(out_k, out_p);
        if (a != b) {
            std::fprintf(stderr, "[rank 0] aligned (key,payload) multiset mismatch: in=%zu out=%zu\n",
                         a.size(), b.size());
            local_fail = 1;
        }
    }

    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    if (ctx.rank == 0) {
        std::printf("[shuffle_test] np=%d rows/rank=%lld (unaligned+aligned) -> %s\n", ctx.size,
                    static_cast<long long>(kRowsPerRank), global_fail ? "FAIL" : "PASS");
    }

    alps::destroy_context(ctx);
    MPI_Finalize();
    return global_fail;
}
