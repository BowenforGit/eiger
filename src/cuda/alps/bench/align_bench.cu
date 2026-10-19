// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_align_bench — does the misalignment-resistant (pre/main/post) split
// actually recover NCCL bandwidth at the OPERATOR level?
//
// For each per-rank size N it builds a 2-column int32 table with random keys
// spanning a large range, so the hash shuffle produces ARBITRARY per-peer counts
// (mostly not multiples of the 4-element/16B vector width) — the realistic,
// misaligned regime (see docs/ALPS_COMMUNICATION_STUDIES.md). It then times,
// head-to-head:
//   shuffle()          vs  shuffle_aligned()        — comm_ms (NCCL all-to-all)
//   allgather()        vs  allgather_aligned()      — wall time (alloc common-mode)
// Each stage is MAX-reduced across ranks per iter; rank 0 reports mean/median and
// the speedup ratio (unaligned / aligned; >1 means the aligned variant is faster).
// One CSV row per (gpus, op, N) to --out, or a table to stdout if --out omitted.
//
//   mpirun -np 4 ./alps_align_bench --size-exp-min 20 --size-exp-max 26 \
//       --iters 10 --warmup 3 [--out align.csv]
//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "memory/column.hpp"
#include "memory/table.hpp"

#include "alps/collectives/allgather.hpp"
#include "alps/collectives/shuffle.hpp"
#include "alps/context.hpp"
#include "alps/error.hpp"

namespace {

using gamax::size_type;

struct Args {
    int         size_exp_min{20};
    int         size_exp_max{26};
    int         iters{10};
    int         warmup{3};
    int         seed{42};
    std::string out; // optional CSV
};

long long arg_ll(const char* v) { return std::strtoll(v, nullptr, 10); }

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if      (k == "--size-exp-min") a.size_exp_min = (int)arg_ll(next());
        else if (k == "--size-exp-max") a.size_exp_max = (int)arg_ll(next());
        else if (k == "--iters")        a.iters        = (int)arg_ll(next());
        else if (k == "--warmup")       a.warmup       = (int)arg_ll(next());
        else if (k == "--seed")         a.seed         = (int)arg_ll(next());
        else if (k == "--out")          a.out          = next();
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.size_exp_min > a.size_exp_max) throw std::runtime_error("size-exp-min > size-exp-max");
    return a;
}

double mean_of(const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x; return v.empty() ? 0.0 : s / v.size();
}
double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Random-key 2-col int32 table: keys uniform over a large range so per-peer
// shuffle counts are arbitrary (the misaligned regime). payload = origin tag.
std::unique_ptr<gamax::table> make_random_table(size_type n, int rank, int seed, cudaStream_t stream) {
    std::vector<int32_t> key(n), pay(n);
    uint64_t s = 0x9e3779b97f4a7c15ull ^ (static_cast<uint64_t>(rank + 1) * 0x100000001b3ull)
               ^ (static_cast<uint64_t>(seed) * 0xD1B54A32D192ED03ull);
    for (size_type i = 0; i < n; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        key[i] = static_cast<int32_t>((s >> 33) & 0x7fffffff); // full positive int32 range
        pay[i] = key[i] + 1;
    }
    std::vector<std::unique_ptr<gamax::column>> cols;
    cols.push_back(gamax::make_column_from_host(key.data(), n, stream));
    cols.push_back(gamax::make_column_from_host(pay.data(), n, stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return gamax::make_table(std::move(cols));
}

double max_reduce(double local, MPI_Comm comm) {
    double g = 0; MPI_Reduce(&local, &g, 1, MPI_DOUBLE, MPI_MAX, 0, comm); return g;
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process();
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    int exit_code = 0;
    try {
        Args args = parse_args(argc, argv);

        FILE* csv = nullptr;
        if (ctx.rank == 0 && !args.out.empty()) {
            csv = std::fopen(args.out.c_str(), "a");
            if (!csv) throw std::runtime_error("cannot open --out file: " + args.out);
            std::fprintf(csv, "gpus,op,n_per_rank,iters,unaligned_ms_mean,unaligned_ms_median,"
                              "aligned_ms_mean,aligned_ms_median,speedup\n");
            std::fflush(csv);
        }
        if (ctx.rank == 0)
            std::printf("# gpus=%d  iters=%d  (ms = MAX across ranks; speedup = unaligned/aligned)\n", ctx.size, args.iters);

        for (int e = args.size_exp_min; e <= args.size_exp_max; ++e) {
            const size_type N = (size_type{1} << e);
            auto tbl = make_random_table(N, ctx.rank, args.seed + e, ctx.stream);

            // ---- shuffle: comm_ms (NCCL all-to-all) ----
            std::vector<double> su, sa; // unaligned, aligned
            for (int w = 0; w < args.warmup; ++w) {
                auto a = alps::collectives::shuffle(tbl->view(), 0, ctx, ctx.stream);
                auto b = alps::collectives::shuffle_aligned(tbl->view(), 0, ctx, ctx.stream);
                ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
            }
            for (int it = 0; it < args.iters; ++it) {
                alps::collectives::ShuffleTiming tu{}, ta{};
                MPI_Barrier(ctx.mpi_comm);
                auto a = alps::collectives::shuffle(tbl->view(), 0, ctx, ctx.stream, &tu);
                ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
                MPI_Barrier(ctx.mpi_comm);
                auto b = alps::collectives::shuffle_aligned(tbl->view(), 0, ctx, ctx.stream, &ta);
                ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
                double gu = max_reduce(tu.comm_ms, ctx.mpi_comm);
                double ga = max_reduce(ta.comm_ms, ctx.mpi_comm);
                if (ctx.rank == 0) { su.push_back(gu); sa.push_back(ga); }
            }

            // ---- allgather: wall time (alloc common-mode to both) ----
            std::vector<double> gu_v, ga_v;
            for (int w = 0; w < args.warmup; ++w) {
                auto a = alps::collectives::allgather(tbl->view(), ctx, ctx.stream);
                auto b = alps::collectives::allgather_aligned(tbl->view(), ctx, ctx.stream);
            }
            for (int it = 0; it < args.iters; ++it) {
                MPI_Barrier(ctx.mpi_comm);
                double t0 = MPI_Wtime();
                auto a = alps::collectives::allgather(tbl->view(), ctx, ctx.stream);
                double t1 = MPI_Wtime();
                MPI_Barrier(ctx.mpi_comm);
                double t2 = MPI_Wtime();
                auto b = alps::collectives::allgather_aligned(tbl->view(), ctx, ctx.stream);
                double t3 = MPI_Wtime();
                double gu = max_reduce((t1 - t0) * 1000.0, ctx.mpi_comm);
                double ga = max_reduce((t3 - t2) * 1000.0, ctx.mpi_comm);
                if (ctx.rank == 0) { gu_v.push_back(gu); ga_v.push_back(ga); }
            }

            if (ctx.rank == 0) {
                struct Row { const char* op; std::vector<double>& u; std::vector<double>& a; };
                Row rows[] = {{"shuffle", su, sa}, {"allgather", gu_v, ga_v}};
                for (auto& r : rows) {
                    const double um = mean_of(r.u), am = mean_of(r.a);
                    const double speedup = am > 0 ? um / am : 0.0;
                    std::printf("  %-9s N=2^%-2d  unaligned %8.3f ms   aligned %8.3f ms   speedup %.2fx\n",
                                r.op, e, um, am, speedup);
                    if (csv) {
                        std::fprintf(csv, "%d,%s,%lld,%d,%.5f,%.5f,%.5f,%.5f,%.4f\n",
                                     ctx.size, r.op, (long long)N, args.iters,
                                     um, median_of(r.u), am, median_of(r.a), speedup);
                        std::fflush(csv);
                    }
                }
            }
        }
        if (csv) std::fclose(csv);
    } catch (const std::exception& e) {
        if (ctx.rank == 0) std::fprintf(stderr, "[alps_align_bench] %s\n", e.what());
        exit_code = 1;
    }

    alps::destroy_context(ctx);
    MPI_Finalize();
    return exit_code;
}
