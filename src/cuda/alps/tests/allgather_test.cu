// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_allgather_test — MPI correctness test for alps::collectives::allgather.
//
// Run under mpirun (-np 2 / 3 / 4). Each rank builds a 2-column int32 chunk with
// a DIFFERENT number of rows (variable per-rank counts — the case the
// broadcast-based all-gather exists for) and globally-unique, self-describing
// content. After all-gather, every rank must hold the concatenation
// [rank0 | rank1 | ... | rank(g-1)] in rank order; each rank verifies its FULL
// output independently against the expected (rank, within-chunk-index) pattern.
// Returns nonzero on any rank's failure (reduced across ranks).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string_column.hpp"

#include "alps/collectives/allgather.hpp"
#include "alps/collectives/ce_pull_shuffle.hpp"
#include "alps/context.hpp"
#include "alps/error.hpp"

namespace {

constexpr int64_t kKeyBase = 1000000; // per-rank key stride (disjoint, fits int32 for a few k rows)

// Deliberately unequal row counts per rank, so the gather exercises variable
// counts / offsets (not the uniform microbench case).
gamax::size_type rows_for_rank(int rank) { return 1000 + 137 * rank; }

std::unique_ptr<gamax::table> make_i32_table(const std::vector<int32_t>& a,
                                             const std::vector<int32_t>& b,
                                             cudaStream_t stream) {
    std::vector<std::unique_ptr<gamax::column>> cols;
    cols.push_back(gamax::make_column_from_host(const_cast<int32_t*>(a.data()),
                                                static_cast<gamax::size_type>(a.size()), stream));
    cols.push_back(gamax::make_column_from_host(const_cast<int32_t*>(b.data()),
                                                static_cast<gamax::size_type>(b.size()), stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return gamax::make_table(std::move(cols));
}

std::vector<int32_t> col_to_host(const gamax::column& c, cudaStream_t stream) {
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    std::vector<int32_t> h(c.size());
    if (c.size() > 0)
        ALPS_CUDACHECK(cudaMemcpy(h.data(), c.data<int32_t>(),
                                  c.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    return h;
}

gamax::column_ptr make_strings(
    const std::vector<std::string>& values, cudaStream_t stream)
{
    std::vector<int32_t> lengths(values.size());
    std::string chars;
    for (std::size_t index = 0; index < values.size(); ++index) {
        lengths[index] = static_cast<int32_t>(values[index].size());
        chars += values[index];
    }
    auto strings = gamax::make_string_column(
        chars.data(), lengths.data(),
        static_cast<gamax::size_type>(values.size()), stream,
        gamax::ARROW_STRING);
    return gamax::column_ptr(strings.release());
}

std::vector<std::string> strings_to_host(
    const gamax::column& column, cudaStream_t stream)
{
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    std::vector<gamax::size_type> offsets(
        static_cast<std::size_t>(column.size()) + 1);
    ALPS_CUDACHECK(cudaMemcpy(
        offsets.data(), column.view().auxiliary_data(),
        offsets.size() * sizeof(gamax::size_type), cudaMemcpyDeviceToHost));
    std::vector<char> chars(static_cast<std::size_t>(offsets.back()));
    if (!chars.empty()) {
        ALPS_CUDACHECK(cudaMemcpy(
            chars.data(), column.data<char>(), chars.size(),
            cudaMemcpyDeviceToHost));
    }
    std::vector<std::string> result(static_cast<std::size_t>(column.size()));
    for (std::size_t row = 0; row < result.size(); ++row) {
        result[row].assign(
            chars.data() + offsets[row],
            static_cast<std::size_t>(offsets[row + 1] - offsets[row]));
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process();
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    // This rank's variable-size chunk: key[i] = rank*kKeyBase + i, pay[i] = key+1.
    const gamax::size_type my_rows = rows_for_rank(ctx.rank);
    std::vector<int32_t> key(my_rows), pay(my_rows);
    for (gamax::size_type i = 0; i < my_rows; ++i) {
        key[i] = static_cast<int32_t>(static_cast<int64_t>(ctx.rank) * kKeyBase + i);
        pay[i] = key[i] + 1;
    }

    auto in  = make_i32_table(key, pay, ctx.stream);
    auto out = alps::collectives::allgather(in->view(), ctx, ctx.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

    // Expected total rows + per-rank chunk offsets (exclusive prefix of counts).
    gamax::size_type total = 0;
    std::vector<gamax::size_type> off(ctx.size);
    for (int s = 0; s < ctx.size; ++s) { off[s] = total; total += rows_for_rank(s); }

    int local_fail = 0;
    if (out->num_rows() != total || out->num_columns() != 2) {
        std::fprintf(stderr, "[rank %d] shape mismatch: rows=%lld (want %lld) cols=%lld\n",
                     ctx.rank, static_cast<long long>(out->num_rows()),
                     static_cast<long long>(total), static_cast<long long>(out->num_columns()));
        local_fail = 1;
    }

    if (!local_fail) {
        auto k = col_to_host(out->get_column(0), ctx.stream);
        auto p = col_to_host(out->get_column(1), ctx.stream);
        // Every rank must hold rank s's chunk at [off[s], off[s]+rows_for_rank(s)).
        for (int s = 0; s < ctx.size && !local_fail; ++s) {
            const gamax::size_type rs = rows_for_rank(s);
            for (gamax::size_type i = 0; i < rs; ++i) {
                const int32_t ek = static_cast<int32_t>(static_cast<int64_t>(s) * kKeyBase + i);
                const gamax::size_type idx = off[s] + i;
                if (k[idx] != ek || p[idx] != ek + 1) {
                    std::fprintf(stderr, "[rank %d] mismatch chunk %d row %lld (global %lld): "
                                 "key=%d pay=%d want %d/%d\n",
                                 ctx.rank, s, static_cast<long long>(i), static_cast<long long>(idx),
                                 k[idx], p[idx], ek, ek + 1);
                    local_fail = 1; break;
                }
            }
        }
    }

    // --- CE backend: drop-in copy-engine all-gather. Same layout + row order as
    //     the NCCL path; apply the identical ordered per-row check. ---
    if (!local_fail) {
        auto outc = alps::collectives::allgather(in->view(), ctx, ctx.stream,
                                                 /*comm_ms=*/nullptr, alps::collectives::Backend::CE);
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
        if (outc->num_rows() != total || outc->num_columns() != 2) {
            std::fprintf(stderr, "[rank %d] CE shape mismatch: rows=%lld (want %lld) cols=%lld\n",
                         ctx.rank, static_cast<long long>(outc->num_rows()),
                         static_cast<long long>(total), static_cast<long long>(outc->num_columns()));
            local_fail = 1;
        } else {
            auto k = col_to_host(outc->get_column(0), ctx.stream);
            auto p = col_to_host(outc->get_column(1), ctx.stream);
            for (int s = 0; s < ctx.size && !local_fail; ++s) {
                const gamax::size_type rs = rows_for_rank(s);
                for (gamax::size_type i = 0; i < rs; ++i) {
                    const int32_t ek = static_cast<int32_t>(static_cast<int64_t>(s) * kKeyBase + i);
                    const gamax::size_type idx = off[s] + i;
                    if (k[idx] != ek || p[idx] != ek + 1) {
                        std::fprintf(stderr, "[rank %d] CE mismatch chunk %d row %lld (global %lld): "
                                     "key=%d pay=%d want %d/%d\n",
                                     ctx.rank, s, static_cast<long long>(i), static_cast<long long>(idx),
                                     k[idx], p[idx], ek, ek + 1);
                        local_fail = 1; break;
                    }
                }
            }
        }
    }

    // --- CE_PULL backend: sender stages its local chunk once and every remote
    //     receiver pulls it. Variable counts and rank-ordered output must match
    //     both NCCL and the older CE push implementation exactly. ---
    if (!local_fail) {
        const auto stats_before = alps::collectives::ce_pull_stats(ctx);
        auto outp = alps::collectives::allgather(
            in->view(), ctx, ctx.stream, /*comm_ms=*/nullptr,
            alps::collectives::Backend::CE_PULL);
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
        const auto stats_after = alps::collectives::ce_pull_stats(ctx);
        const auto expected_remote_bytes = static_cast<std::uint64_t>(
            my_rows * 2 * sizeof(int32_t) * (ctx.size - 1));
        if (stats_after.remote_bytes - stats_before.remote_bytes !=
            expected_remote_bytes) {
            std::fprintf(
                stderr,
                "[rank %d] CE_PULL fixed-width byte count mismatch\n",
                ctx.rank);
            local_fail = 1;
        }
        if (outp->num_rows() != total || outp->num_columns() != 2) {
            std::fprintf(stderr,
                "[rank %d] CE_PULL shape mismatch: rows=%lld (want %lld) cols=%lld\n",
                ctx.rank, static_cast<long long>(outp->num_rows()),
                static_cast<long long>(total),
                static_cast<long long>(outp->num_columns()));
            local_fail = 1;
        } else {
            auto k = col_to_host(outp->get_column(0), ctx.stream);
            auto p = col_to_host(outp->get_column(1), ctx.stream);
            for (int s = 0; s < ctx.size && !local_fail; ++s) {
                const gamax::size_type rs = rows_for_rank(s);
                for (gamax::size_type i = 0; i < rs; ++i) {
                    const int32_t ek = static_cast<int32_t>(
                        static_cast<int64_t>(s) * kKeyBase + i);
                    const gamax::size_type idx = off[s] + i;
                    if (k[idx] != ek || p[idx] != ek + 1) {
                        std::fprintf(stderr,
                            "[rank %d] CE_PULL mismatch chunk %d row %lld "
                            "(global %lld): key=%d pay=%d want %d/%d\n",
                            ctx.rank, s, static_cast<long long>(i),
                            static_cast<long long>(idx), k[idx], p[idx],
                            ek, ek + 1);
                        local_fail = 1;
                        break;
                    }
                }
            }
        }
    }

    // Variable row counts give each sender a different aligned length-array
    // footprint. Verify that CE_PULL uses the sender-advertised character
    // offset rather than assuming every rank stages characters identically.
    if (!local_fail) {
        std::vector<std::string> local_strings(
            static_cast<std::size_t>(my_rows));
        for (gamax::size_type row = 0; row < my_rows; ++row) {
            local_strings[static_cast<std::size_t>(row)] =
                "rank-" + std::to_string(ctx.rank) + "-row-" +
                std::to_string(row) + std::string(
                    static_cast<std::size_t>((ctx.rank + row) % 5), 'x');
        }
        std::vector<gamax::column_ptr> string_columns;
        string_columns.push_back(make_strings(local_strings, ctx.stream));
        auto string_input = gamax::make_table(std::move(string_columns));
        const auto stats_before = alps::collectives::ce_pull_stats(ctx);
        auto gathered_strings = alps::collectives::allgather(
            string_input->view(), ctx, ctx.stream, /*comm_ms=*/nullptr,
            alps::collectives::Backend::CE_PULL);
        const auto stats_after = alps::collectives::ce_pull_stats(ctx);
        std::uint64_t local_char_bytes = 0;
        for (const auto& value : local_strings) {
            local_char_bytes += value.size();
        }
        const auto expected_remote_bytes =
            (static_cast<std::uint64_t>(my_rows) * sizeof(gamax::size_type) +
             local_char_bytes) * static_cast<std::uint64_t>(ctx.size - 1);
        if (stats_after.remote_bytes - stats_before.remote_bytes !=
            expected_remote_bytes) {
            std::fprintf(
                stderr, "[rank %d] CE_PULL STRING byte count mismatch\n",
                ctx.rank);
            local_fail = 1;
        }
        const auto got = strings_to_host(
            gathered_strings->get_column(0), ctx.stream);
        std::vector<std::string> expected;
        expected.reserve(static_cast<std::size_t>(total));
        for (int sender = 0; sender < ctx.size; ++sender) {
            for (gamax::size_type row = 0; row < rows_for_rank(sender); ++row) {
                expected.push_back(
                    "rank-" + std::to_string(sender) + "-row-" +
                    std::to_string(row) + std::string(
                        static_cast<std::size_t>((sender + row) % 5), 'x'));
            }
        }
        if (got != expected) {
            std::fprintf(
                stderr, "[rank %d] CE_PULL STRING allgather mismatch\n",
                ctx.rank);
            local_fail = 1;
        }
    }

    // --- aligned variant: same result SET, row order NOT preserved (the
    //     main/post split moves each chunk's ragged tail to the end). Verify
    //     (a) shape, (b) per-row column integrity (pay == key+1), and (c) the
    //     multiset of keys equals the expected globally-unique key set. ---
    if (!local_fail) {
        auto outa = alps::collectives::allgather_aligned(in->view(), ctx, ctx.stream);
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
        if (outa->num_rows() != total || outa->num_columns() != 2) {
            std::fprintf(stderr, "[rank %d] aligned shape mismatch: rows=%lld (want %lld) cols=%lld\n",
                         ctx.rank, static_cast<long long>(outa->num_rows()),
                         static_cast<long long>(total), static_cast<long long>(outa->num_columns()));
            local_fail = 1;
        } else {
            auto k = col_to_host(outa->get_column(0), ctx.stream);
            auto p = col_to_host(outa->get_column(1), ctx.stream);
            std::vector<int32_t> got, exp;
            got.reserve(k.size());
            for (size_t i = 0; i < k.size(); ++i) {
                if (p[i] != k[i] + 1) { // column desync -> pair broken
                    std::fprintf(stderr, "[rank %d] aligned col desync at row %zu: key=%d pay=%d\n",
                                 ctx.rank, i, k[i], p[i]);
                    local_fail = 1; break;
                }
                got.push_back(k[i]);
            }
            if (!local_fail) {
                exp.reserve(total);
                for (int s = 0; s < ctx.size; ++s) {
                    const gamax::size_type rs = rows_for_rank(s);
                    for (gamax::size_type i = 0; i < rs; ++i)
                        exp.push_back(static_cast<int32_t>(static_cast<int64_t>(s) * kKeyBase + i));
                }
                std::sort(got.begin(), got.end());
                std::sort(exp.begin(), exp.end());
                if (got != exp) {
                    std::fprintf(stderr, "[rank %d] aligned key-set mismatch\n", ctx.rank);
                    local_fail = 1;
                }
            }
        }
    }

    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    if (ctx.rank == 0)
        std::printf("[allgather_test] np=%d total_rows=%lld (unaligned+aligned) -> %s\n", ctx.size,
                    static_cast<long long>(total), global_fail ? "FAIL" : "PASS");

    alps::destroy_context(ctx);
    MPI_Finalize();
    return global_fail;
}
