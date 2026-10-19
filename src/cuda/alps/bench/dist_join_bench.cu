// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_dist_join_bench — multi-GPU throughput benchmark for the shuffle-based
// distributed inner join (uniform PK/FK). One rank per GPU.
//
// In-process sweep over per-rank size N (powers of two) x local-join configs.
// For each N the PK/FK shards are generated once and all configs run on them;
// each config runs a full distributed join (re-shuffling) so the measured unit
// is the real alps::join::inner_eq_join operator. Per iteration each stage
// (partition / all-to-all comm / local join) is MAX-reduced across ranks; we
// report mean/median/std over the timed iterations. One CSV row per
// (gpus, join_algo, mat_algo, N) is appended to --out (rank 0).
//
// Allocator: run under the RMM arena (optionally set GAMAX_MEM_RESERVE_BYTES)
// to remove per-iteration cudaMalloc/cudaFree from the timing loop.
//
//   mpirun -np 4 ./alps_dist_join_bench --size-exp-min 20 --size-exp-max 28 \
//       --iters 10 --warmup 3 --out results.csv
//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "memory/table.hpp"
#include "operators/join/join.hpp"
#include "types/data_type.hpp"

#include "alps/context.hpp"
#include "alps/datagen/gen_shard.hpp"
#include "alps/error.hpp"
#include "alps/operators/dist_join.hpp"

namespace {

struct JoinConfig {
    gamax::join::join_algo          algo;
    gamax::join::materialization_algo mat;
    const char*                     algo_name;
    const char*                     mat_name;
};

// The 5 local-join configurations to sweep.
const JoinConfig kConfigs[] = {
    {gamax::join::HASH,      gamax::join::GFUR, "hash", "gfur"},
    {gamax::join::SORT,      gamax::join::GFUR, "smj",  "gfur"},
    {gamax::join::SORT,      gamax::join::GFTR, "smj",  "gftr"},
    {gamax::join::PARTITION, gamax::join::GFUR, "phj",  "gfur"},
    {gamax::join::PARTITION, gamax::join::GFTR, "phj",  "gftr"},
};

struct Args {
    int         size_exp_min{20};
    int         size_exp_max{28};
    int         iters{10};
    int         warmup{3};
    int         seed{42};
    std::string key_type{"i32"};
    std::string out;     // CSV output file (required)
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
        else if (k == "--key-type")     a.key_type     = next();
        else if (k == "--out")          a.out          = next();
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.out.empty()) throw std::runtime_error("--out <csv file> is required");
    if (a.size_exp_min > a.size_exp_max) throw std::runtime_error("size-exp-min > size-exp-max");
    return a;
}

gamax::data_type key_type_of(const std::string& s) {
    if (s == "i32") return gamax::data_type{gamax::type_id::INT32};
    if (s == "i64") return gamax::data_type{gamax::type_id::INT64};
    throw std::runtime_error("--key-type must be i32 or i64");
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
double std_of(const std::vector<double>& v, double mean) {
    if (v.size() < 2) return 0.0;
    double s = 0; for (double x : v) s += (x - mean) * (x - mean);
    return std::sqrt(s / (v.size() - 1)); // sample std
}

// Does the CSV file already have content (header already written)?
bool file_nonempty(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fclose(f);
    return sz > 0;
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process(); // allocator + MPI env defaults; before MPI_Init
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    int exit_code = 0;
    try {
        Args args = parse_args(argc, argv);
        const gamax::data_type ktype = key_type_of(args.key_type); // any rank count supported

        FILE* csv = nullptr;
        if (ctx.rank == 0) {
            const bool need_header = !file_nonempty(args.out);
            csv = std::fopen(args.out.c_str(), "a");
            if (!csv) throw std::runtime_error("cannot open --out file: " + args.out);
            if (need_header) {
                std::fprintf(csv,
                    "gpus,key_type,join_algo,mat_algo,pk_rows_per_rank,fk_rows_per_rank,r,"
                    "total_pk,total_fk,total_tuples,iters,global_out_rows,"
                    "partition_ms_mean,partition_ms_median,partition_ms_std,"
                    "comm_ms_mean,comm_ms_median,comm_ms_std,"
                    "join_ms_mean,join_ms_median,join_ms_std,"
                    "total_ms_mean,total_ms_median,total_ms_std,"
                    "throughput_gtuples_s_mean,throughput_gtuples_s_median,throughput_gtuples_s_std\n");
                std::fflush(csv);
            }
        }

        for (int e = args.size_exp_min; e <= args.size_exp_max; ++e) {
            const gamax::size_type N = (gamax::size_type{1} << e); // per-rank pk == fk (r = 1)

            alps::datagen::GenParams gp;
            gp.key_type         = ktype;
            gp.pk_rows_per_rank = N;
            gp.fk_rows_per_rank = N;
            gp.ncols            = 2;
            gp.match_ratio      = 1.0;
            gp.seed             = args.seed;

            auto shard = alps::datagen::gen_shard(ctx.rank, ctx.size, gp, ctx.stream);
            auto& pk = shard.first;
            auto& fk = shard.second;

            const long long total_pk     = (long long)ctx.size * N;
            const long long total_fk     = (long long)ctx.size * N;
            const long long total_tuples = total_pk + total_fk;
            const int       r            = 1;

            for (const JoinConfig& cfg : kConfigs) {
                alps::join::DistJoinHint hint;
                hint.local.join_algo   = cfg.algo;
                hint.local.mat_algo    = cfg.mat;
                hint.local.unique_left = true; // PK build side
                hint.local.max_out     = 0;    // auto-estimate
                hint.left_key_idx  = 0;
                hint.right_key_idx = 0;

                // Collective probe: run one dist-join and decide TOGETHER whether
                // this config is viable. Some local joins throw at large N (e.g.
                // phj "build-side bucket overflow"); that throw happens on the
                // local join AFTER the shuffle's collectives + an implicit sync,
                // so every rank reaches this Allreduce in lockstep. If any rank
                // fails, all ranks skip the config and continue (no divergence).
                int local_fail = 0;
                std::string err;
                try {
                    auto out = alps::join::inner_eq_join(pk->view(), fk->view(), hint, ctx, ctx.stream, nullptr);
                    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
                } catch (const std::exception& e) {
                    local_fail = 1; err = e.what();
                    cudaStreamSynchronize(ctx.stream);
                    cudaGetLastError();
                }
                int global_fail = 0;
                ALPS_MPICHECK(MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
                if (global_fail) {
                    if (ctx.rank == 0) {
                        std::fprintf(stderr, "[alps_dist_join_bench] skip gpus=%d N=%lld %s/%s: %s\n",
                                     ctx.size, (long long)N, cfg.algo_name, cfg.mat_name,
                                     err.empty() ? "failed on another rank" : err.c_str());
                    }
                    continue;
                }

                // Warmup.
                for (int w = 0; w < args.warmup; ++w) {
                    auto out = alps::join::inner_eq_join(pk->view(), fk->view(), hint, ctx, ctx.stream, nullptr);
                    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
                }

                // Timed iterations: per-iter MAX-reduce of each stage to rank 0.
                std::vector<double> part, comm, join; // rank 0 only
                if (ctx.rank == 0) { part.reserve(args.iters); comm.reserve(args.iters); join.reserve(args.iters); }
                gamax::size_type my_out_rows = 0;

                for (int it = 0; it < args.iters; ++it) {
                    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
                    alps::join::DistJoinTiming t{};
                    auto out = alps::join::inner_eq_join(pk->view(), fk->view(), hint, ctx, ctx.stream, &t);
                    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
                    my_out_rows = out->num_rows();

                    float pm = 0, cm = 0, jm = 0;
                    ALPS_MPICHECK(MPI_Reduce(&t.partition_ms, &pm, 1, MPI_FLOAT, MPI_MAX, 0, ctx.mpi_comm));
                    ALPS_MPICHECK(MPI_Reduce(&t.comm_ms,      &cm, 1, MPI_FLOAT, MPI_MAX, 0, ctx.mpi_comm));
                    ALPS_MPICHECK(MPI_Reduce(&t.join_ms,      &jm, 1, MPI_FLOAT, MPI_MAX, 0, ctx.mpi_comm));
                    if (ctx.rank == 0) { part.push_back(pm); comm.push_back(cm); join.push_back(jm); }
                }

                long long my_rows = (long long)my_out_rows, global_out_rows = 0;
                ALPS_MPICHECK(MPI_Reduce(&my_rows, &global_out_rows, 1, MPI_LONG_LONG, MPI_SUM, 0, ctx.mpi_comm));

                if (ctx.rank == 0) {
                    std::vector<double> total(args.iters), thr(args.iters);
                    for (int i = 0; i < args.iters; ++i) {
                        total[i] = part[i] + comm[i] + join[i];
                        thr[i]   = total[i] > 0.0
                                 ? (double)total_tuples / (total[i] / 1000.0) / 1e9 // 10^9 tuples/s
                                 : 0.0;
                    }
                    const double pm_mean = mean_of(part), cm_mean = mean_of(comm), jm_mean = mean_of(join);
                    const double tm_mean = mean_of(total), th_mean = mean_of(thr);

                    std::fprintf(csv,
                        "%d,%s,%s,%s,%lld,%lld,%d,%lld,%lld,%lld,%d,%lld,"
                        "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
                        ctx.size, args.key_type.c_str(), cfg.algo_name, cfg.mat_name,
                        (long long)N, (long long)N, r,
                        total_pk, total_fk, total_tuples, args.iters, global_out_rows,
                        pm_mean, median_of(part), std_of(part, pm_mean),
                        cm_mean, median_of(comm), std_of(comm, cm_mean),
                        jm_mean, median_of(join), std_of(join, jm_mean),
                        tm_mean, median_of(total), std_of(total, tm_mean),
                        th_mean, median_of(thr), std_of(thr, th_mean));
                    std::fflush(csv);

                    if (global_out_rows != total_fk) {
                        std::fprintf(stderr,
                            "[alps_dist_join_bench] WARNING gpus=%d N=%lld %s/%s: "
                            "global_out_rows=%lld != total_fk=%lld\n",
                            ctx.size, (long long)N, cfg.algo_name, cfg.mat_name,
                            global_out_rows, total_fk);
                    }
                }
            }
            // pk, fk released here -> back to the slab for the next size.
        }

        if (csv) std::fclose(csv);
    } catch (const std::exception& e) {
        if (ctx.rank == 0) std::fprintf(stderr, "[alps_dist_join_bench] %s\n", e.what());
        exit_code = 1;
    }

    alps::destroy_context(ctx);
    MPI_Finalize();
    return exit_code;
}
