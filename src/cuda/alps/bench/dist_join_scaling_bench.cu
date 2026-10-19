// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_dist_join_scaling_bench — shuffle vs broadcast distributed join, with a
// strong/weak scaling knob and a per-stage comm/compute breakdown.
//
// Data model (gen_shard): PK is the small/unique side, FK = ratio x PK (FK:PK =
// `ratio`, every FK matches a PK -> output rows == total FK). The exponent `e`
// sizes the LARGE side (FK); pk_rows_per_rank = fk_rows_per_rank / ratio.
//
//   --scaling weak   : fk_rows_per_rank = 2^e (fixed per rank); total = 2^e * P.
//                      -> broadcast's replicated PK grows with P (weak-scaling cost).
//   --scaling strong : total FK = 2^e (fixed); fk_rows_per_rank = 2^e / P.
//                      -> broadcast's replicated PK is a fixed total (strong floor).
// FK/rank is rounded down to a multiple of `ratio` so pk = fk/ratio is integral.
//
// For each (ratio, e, strategy, local-join algo) it runs the distributed join,
// MAX-reduces each stage (partition / comm / join) across ranks per iter, and
// writes mean/median + throughput to --out (one row per
// gpus,scaling,strategy,join_algo,ratio,size). Scaling vs node count comes from
// launching at -np 1 / 2 / 4 / 8.
//
// Broadcast is GATED: it is skipped whenever total_pk > fk_per_rank (i.e. P >
// ratio), because replicating a PK side larger than a rank's own FK shard can
// never beat shuffling that shard. So for FK:PK=ratio, broadcast only runs at
// P <= ratio. Shuffle always runs.
//
// `--algos` sweeps the LOCAL-join algorithm (GFUR materialization): hash | smj
// (sort-merge) | phj (partition-hash). Default `hash`. Use it to compare the
// local-join algorithms' time breakdown under (strong) scaling, e.g.
//   ... --scaling strong --strategy shuffle --ratios 1 --algos hash,smj,phj
//
// IMPORTANT: performance runs use the persistent RMM arena. Direct RMM mode
// uses synchronous cudaMalloc/cudaFree and distorts allocator-sensitive timing:
//   GAMAX_RMM_RESOURCE=arena GAMAX_MEM_RESERVE_BYTES=<bytes> mpirun ...
//
//   GAMAX_RMM_RESOURCE=arena GAMAX_MEM_RESERVE_BYTES=16000000000 \
//   mpirun -np 8 ./alps_dist_join_scaling_bench --scaling weak \
//       --fk-exp-min 22 --fk-exp-max 26 --ratios 4,16,64 --strategy both \
//       --iters 10 --warmup 3 --out scaling.csv
//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include "memory/table.hpp"
#include "operators/join/join.hpp"
#include "types/data_type.hpp"

#include "alps/context.hpp"
#include "alps/datagen/gen_shard.hpp"
#include "alps/error.hpp"
#include "alps/operators/dist_join.hpp"

namespace {

using gamax::size_type;

// --- Order-independent output verification --------------------------------
// gen_shard builds both tables as ncols identical copies of the key column
// (data_gen/table_gen.cu generate_table) and match_ratio=1, so EVERY row is
// (key, key). A correct inner equi-join emits [pk|fk] only when pk_key==fk_key,
// hence every correct output row has all four columns equal. We check that, plus
// an absolute value-sensitive no-loss/no-dup checksum (sum of the output key
// equals the sum over the whole FK input, since each FK row joins exactly once).
// Both are order-independent reductions, so they hold regardless of how the
// shuffle/broadcast permuted rows. Unsigned 64-bit: keys < 2^30, <= 2^30 rows ->
// sums < 2^60, exact (no overflow).

// Sum of an int32 column (as u64) over [0, n). Reads one column.
unsigned long long colsum_u64(const int32_t* c, size_type n, cudaStream_t stream) {
    if (n <= 0) return 0ull;
    gamax::rmm_sync_exec_policy_handle policy{stream};
    return thrust::transform_reduce(
        policy.get(),
        thrust::counting_iterator<size_type>(0), thrust::counting_iterator<size_type>(n),
        [c] __device__(size_type i) -> unsigned long long {
            return static_cast<unsigned long long>(static_cast<uint32_t>(c[i]));
        },
        0ull, thrust::plus<unsigned long long>());
}

// Count of rows where the four output columns are NOT all equal.
unsigned long long bad_rows_u64(const int32_t* c0, const int32_t* c1,
                                const int32_t* c2, const int32_t* c3,
                                size_type n, cudaStream_t stream) {
    if (n <= 0) return 0ull;
    gamax::rmm_sync_exec_policy_handle policy{stream};
    return thrust::transform_reduce(
        policy.get(),
        thrust::counting_iterator<size_type>(0), thrust::counting_iterator<size_type>(n),
        [c0, c1, c2, c3] __device__(size_type i) -> unsigned long long {
            const int32_t v = c0[i];
            return (v == c1[i] && v == c2[i] && v == c3[i]) ? 0ull : 1ull;
        },
        0ull, thrust::plus<unsigned long long>());
}

struct Args {
    std::string              scaling{"weak"};   // weak | strong
    std::string              strategy{"both"};  // shuffle | broadcast | both
    std::vector<int>         ratios{16};        // FK:PK ratios (integer, power-of-two recommended)
    std::vector<std::string> algos{"hash"};     // local-join algos (GFUR): hash | smj | phj
    long long        total_fk{0};               // absolute total FK rows (strong); 0 = use fk-exp sweep
    int              fk_exp_min{22};
    int              fk_exp_max{26};
    int              iters{10};
    int              warmup{3};
    int              seed{42};
    std::string      out;                  // required CSV
};

long long arg_ll(const char* v) { return std::strtoll(v, nullptr, 10); }

std::vector<int> parse_ratios(const std::string& s) {
    std::vector<int> r;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) r.push_back(static_cast<int>(std::strtol(tok.c_str(), nullptr, 10)));
    }
    if (r.empty()) throw std::runtime_error("--ratios: empty list");
    return r;
}

std::vector<std::string> parse_strs(const std::string& s) {
    std::vector<std::string> r;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) r.push_back(tok);
    if (r.empty()) throw std::runtime_error("--algos: empty list");
    return r;
}

// Local-join algorithm by name; materialization is always GFUR for this study.
gamax::join::join_algo algo_of(const std::string& s) {
    if (s == "hash") return gamax::join::HASH;
    if (s == "smj")  return gamax::join::SORT;
    if (s == "phj")  return gamax::join::PARTITION;
    throw std::runtime_error("--algos must be hash|smj|phj (got '" + s + "')");
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if      (k == "--scaling")     a.scaling    = next();
        else if (k == "--strategy")    a.strategy   = next();
        else if (k == "--ratios")      a.ratios     = parse_ratios(next());
        else if (k == "--algos")       a.algos      = parse_strs(next());
        else if (k == "--total-fk")    a.total_fk   = arg_ll(next());
        else if (k == "--fk-exp-min")  a.fk_exp_min = (int)arg_ll(next());
        else if (k == "--fk-exp-max")  a.fk_exp_max = (int)arg_ll(next());
        else if (k == "--iters")       a.iters      = (int)arg_ll(next());
        else if (k == "--warmup")      a.warmup     = (int)arg_ll(next());
        else if (k == "--seed")        a.seed       = (int)arg_ll(next());
        else if (k == "--out")         a.out        = next();
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.out.empty()) throw std::runtime_error("--out <csv file> is required");
    if (a.scaling != "weak" && a.scaling != "strong") throw std::runtime_error("--scaling must be weak|strong");
    if (a.strategy != "shuffle" && a.strategy != "broadcast" && a.strategy != "both")
        throw std::runtime_error("--strategy must be shuffle|broadcast|both");
    for (const auto& al : a.algos) (void)algo_of(al); // validate names early
    if (a.fk_exp_min > a.fk_exp_max) throw std::runtime_error("fk-exp-min > fk-exp-max");
    return a;
}

double mean_of(const std::vector<double>& v) { double s=0; for (double x:v) s+=x; return v.empty()?0.0:s/v.size(); }
double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n & 1) ? v[n/2] : 0.5*(v[n/2-1]+v[n/2]);
}
bool file_nonempty(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fclose(f);
    return sz > 0;
}

size_type round_down_mult(size_type n, int m) { return (n / m) * static_cast<size_type>(m); }

} // namespace

int main(int argc, char** argv) {
    alps::configure_process();
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    int exit_code = 0;
    try {
        Args args = parse_args(argc, argv);
        const int P = ctx.size;

        // Direct cuda_memory_resource is a diagnostic mode and adds synchronous
        // allocation cost to timed work. Performance runs use the RMM arena.
        if (ctx.rank == 0) {
            const char* mode = std::getenv("GAMAX_RMM_RESOURCE");
            if (mode && std::strcmp(mode, "arena") != 0)
                std::fprintf(stderr,
                    "[scaling_bench] WARNING: direct RMM mode is diagnostic and distorts "
                    "allocator-sensitive timings. Re-run with GAMAX_RMM_RESOURCE=arena.\n");
        }

        std::vector<alps::join::DistJoinHint::Strategy> strategies;
        if (args.strategy == "both" || args.strategy == "shuffle")
            strategies.push_back(alps::join::DistJoinHint::Strategy::shuffle);
        if (args.strategy == "both" || args.strategy == "broadcast")
            strategies.push_back(alps::join::DistJoinHint::Strategy::broadcast);

        FILE* csv = nullptr;
        if (ctx.rank == 0) {
            const bool need_header = !file_nonempty(args.out);
            csv = std::fopen(args.out.c_str(), "a");
            if (!csv) throw std::runtime_error("cannot open --out file: " + args.out);
            if (need_header) {
                std::fprintf(csv,
                    "gpus,scaling,strategy,join_algo,ratio,fk_per_rank,pk_per_rank,total_fk,total_pk,"
                    "iters,global_out_rows,verify_ok,"
                    "partition_ms_mean,partition_ms_median,comm_ms_mean,comm_ms_median,"
                    "join_ms_mean,join_ms_median,total_ms_mean,total_ms_median,"
                    "throughput_gtuples_s_mean\n");
                std::fflush(csv);
            }
        }

        for (int ratio : args.ratios) {
            // Per-rank FK size points: an absolute --total-fk (one point, strong),
            // else the 2^e sweep. Each rounded to a multiple of `ratio` so pk is
            // integral (gen_shard needs fk % pk == 0).
            std::vector<size_type> fk_sizes;
            if (args.total_fk > 0) {
                fk_sizes.push_back(round_down_mult(static_cast<size_type>(args.total_fk) / P, ratio));
            } else {
                for (int e = args.fk_exp_min; e <= args.fk_exp_max; ++e) {
                    const size_type f = (args.scaling == "weak") ? (size_type{1} << e)
                                                                 : ((size_type{1} << e) / P);
                    fk_sizes.push_back(round_down_mult(f, ratio));
                }
            }
            for (size_type fk_per_rank : fk_sizes) {
                const size_type pk_per_rank = fk_per_rank / ratio;
                if (pk_per_rank < 1 || fk_per_rank < 1) {
                    if (ctx.rank == 0)
                        std::fprintf(stderr, "[scaling_bench] skip ratio=%d: pk/fk per rank < 1\n", ratio);
                    continue;
                }

                alps::datagen::GenParams gp;
                gp.key_type         = gamax::data_type{gamax::type_id::INT32};
                gp.pk_rows_per_rank = pk_per_rank;
                gp.fk_rows_per_rank = fk_per_rank;
                gp.ncols            = 2;
                gp.match_ratio      = 1.0;
                gp.seed             = args.seed + ratio;

                auto shard = alps::datagen::gen_shard(ctx.rank, P, gp, ctx.stream);
                auto& pk = shard.first;
                auto& fk = shard.second;

                const long long total_fk = (long long)fk_per_rank * P;
                const long long total_pk = (long long)pk_per_rank * P;

                // Expected output key-checksum: sum of the FK key over the whole
                // (distributed) FK input. Each FK row joins exactly once, so the
                // output key sum must equal this regardless of strategy/algo.
                unsigned long long fk_keysum_local =
                    colsum_u64(fk->view().get_column(0).data<int32_t>(), fk->num_rows(), ctx.stream);
                ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
                unsigned long long fk_keysum_global = 0;
                ALPS_MPICHECK(MPI_Allreduce(&fk_keysum_local, &fk_keysum_global, 1,
                                            MPI_UNSIGNED_LONG_LONG, MPI_SUM, ctx.mpi_comm));

                for (auto strat : strategies) {
                  const char* sname =
                      (strat == alps::join::DistJoinHint::Strategy::broadcast) ? "broadcast" : "shuffle";

                  // Broadcast gate: replicating the (small) PK side to every rank
                  // only pays off when the replicated total is no larger than the
                  // per-rank FK shard it spares from the shuffle. Once total_pk >
                  // fk_per_rank (equivalently P > ratio, since total_pk = total_fk/
                  // ratio and fk_per_rank = total_fk/P) every rank holds more PK
                  // than its own FK -- broadcasting is pointless, so skip it. The
                  // condition is deterministic from the args, so all ranks agree
                  // (no collective needed).
                  if (strat == alps::join::DistJoinHint::Strategy::broadcast &&
                      total_pk > (long long)fk_per_rank) {
                      if (ctx.rank == 0)
                          std::fprintf(stderr,
                              "[scaling_bench] skip broadcast PK:FK=1/%d fk/rk=%lld: total_pk=%lld > "
                              "fk_per_rank=%lld (P=%d > ratio=%d) -- broadcast not sensible\n",
                              ratio, (long long)fk_per_rank, total_pk, (long long)fk_per_rank, P, ratio);
                      continue;
                  }

                  for (const std::string& aname : args.algos) {
                    alps::join::DistJoinHint hint;
                    hint.local.join_algo   = algo_of(aname);
                    hint.local.mat_algo    = gamax::join::GFUR; // GFUR variants only
                    hint.local.unique_left = true; // PK (left) build side is unique
                    hint.local.max_out     = 0;
                    hint.left_key_idx      = 0;
                    hint.right_key_idx     = 0;
                    hint.strategy          = strat;
                    // broadcast_side autosel -> gathers PK (smaller); shuffle_align
                    // autosel -> size-gated aligned all-to-all.

                    // Collective probe: run once, agree across ranks whether it
                    // succeeded (phj throws build-side bucket overflow at large N;
                    // broadcast can OOM on the replicated side); skip if any rank fails.
                    int local_fail = 0;
                    std::string err;
                    // Value-correctness counters from the probe output (untimed).
                    unsigned long long v_bad_local = 0, v_keysum_local = 0;
                    try {
                        auto out = alps::join::inner_eq_join(pk->view(), fk->view(), hint, ctx, ctx.stream, nullptr);
                        auto ov = out->view();
                        if (ov.num_columns() == 4) {
                            v_bad_local = bad_rows_u64(ov.get_column(0).data<int32_t>(),
                                                       ov.get_column(1).data<int32_t>(),
                                                       ov.get_column(2).data<int32_t>(),
                                                       ov.get_column(3).data<int32_t>(),
                                                       out->num_rows(), ctx.stream);
                        }
                        v_keysum_local = colsum_u64(ov.get_column(0).data<int32_t>(),
                                                    out->num_rows(), ctx.stream);
                        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
                    } catch (const std::exception& ex) {
                        local_fail = 1; err = ex.what();
                        cudaStreamSynchronize(ctx.stream);
                        cudaGetLastError();
                    }
                    int global_fail = 0;
                    ALPS_MPICHECK(MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
                    if (global_fail) {
                        if (ctx.rank == 0)
                            std::fprintf(stderr, "[scaling_bench] skip %s/%s PK:FK=1/%d fk/rk=%lld: %s\n",
                                         sname, aname.c_str(), ratio, (long long)fk_per_rank,
                                         err.empty() ? "failed on another rank" : err.c_str());
                        continue;
                    }

                    // Reduce the value-correctness counters across ranks and check
                    // against the absolute expected key checksum. bad_rows must be 0
                    // (every output row is a correctly-materialized match) and the
                    // global output key sum must equal the whole FK input's.
                    unsigned long long v_bad_g = 0, v_keysum_g = 0;
                    ALPS_MPICHECK(MPI_Reduce(&v_bad_local, &v_bad_g, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, ctx.mpi_comm));
                    ALPS_MPICHECK(MPI_Reduce(&v_keysum_local, &v_keysum_g, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, ctx.mpi_comm));
                    int verify_ok = 1;
                    if (ctx.rank == 0) {
                        verify_ok = (v_bad_g == 0 && v_keysum_g == fk_keysum_global) ? 1 : 0;
                        if (!verify_ok) {
                            std::fprintf(stderr,
                                "[scaling_bench] VERIFY FAIL %s/%s P=%d PK:FK=1/%d fk/rk=%lld: "
                                "bad_rows=%llu out_keysum=%llu expected=%llu\n",
                                sname, aname.c_str(), P, ratio, (long long)fk_per_rank,
                                v_bad_g, v_keysum_g, fk_keysum_global);
                            exit_code = 1;
                        }
                    }

                    for (int w = 0; w < args.warmup; ++w) {
                        auto out = alps::join::inner_eq_join(pk->view(), fk->view(), hint, ctx, ctx.stream, nullptr);
                        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
                    }

                    std::vector<double> part, comm, join; // rank 0
                    size_type my_out_rows = 0;
                    for (int it = 0; it < args.iters; ++it) {
                        ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
                        alps::join::DistJoinTiming t{};
                        auto out = alps::join::inner_eq_join(pk->view(), fk->view(), hint, ctx, ctx.stream, &t);
                        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
                        my_out_rows = out->num_rows();
                        float pm=0, cm=0, jm=0;
                        ALPS_MPICHECK(MPI_Reduce(&t.partition_ms, &pm, 1, MPI_FLOAT, MPI_MAX, 0, ctx.mpi_comm));
                        ALPS_MPICHECK(MPI_Reduce(&t.comm_ms,      &cm, 1, MPI_FLOAT, MPI_MAX, 0, ctx.mpi_comm));
                        ALPS_MPICHECK(MPI_Reduce(&t.join_ms,      &jm, 1, MPI_FLOAT, MPI_MAX, 0, ctx.mpi_comm));
                        if (ctx.rank == 0) { part.push_back(pm); comm.push_back(cm); join.push_back(jm); }
                    }

                    long long my_rows = (long long)my_out_rows, global_out_rows = 0;
                    ALPS_MPICHECK(MPI_Reduce(&my_rows, &global_out_rows, 1, MPI_LONG_LONG, MPI_SUM, 0, ctx.mpi_comm));

                    if (ctx.rank == 0) {
                        const long long total_tuples = total_pk + total_fk;
                        std::vector<double> total(args.iters), thr(args.iters);
                        for (int i = 0; i < args.iters; ++i) {
                            total[i] = part[i] + comm[i] + join[i];
                            thr[i]   = total[i] > 0.0 ? (double)total_tuples / (total[i] / 1000.0) / 1e9 : 0.0;
                        }
                        std::fprintf(csv,
                            "%d,%s,%s,%s,%d,%lld,%lld,%lld,%lld,%d,%lld,%d,"
                            "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
                            P, args.scaling.c_str(), sname, aname.c_str(), ratio,
                            (long long)fk_per_rank, (long long)pk_per_rank, total_fk, total_pk,
                            args.iters, global_out_rows, verify_ok,
                            mean_of(part), median_of(part), mean_of(comm), median_of(comm),
                            mean_of(join), median_of(join), mean_of(total), median_of(total),
                            mean_of(thr));
                        std::fflush(csv);
                        const bool out_bad = (global_out_rows != total_fk) || !verify_ok;
                        std::printf("  %-9s/%-4s P=%2d %s PK:FK=1/%-3d fk/rk=%-10lld pk/rk=%-10lld "
                                    "part %6.3f  comm %7.3f  join %7.3f  total %7.3f ms%s\n",
                                    sname, aname.c_str(), P, args.scaling.c_str(), ratio,
                                    (long long)fk_per_rank, (long long)pk_per_rank,
                                    mean_of(part), mean_of(comm), mean_of(join),
                                    mean_of(total), out_bad ? "  [!VERIFY!]" : "  [ok]");
                    }
                  } // algo
                }
            }
        }
        if (csv) std::fclose(csv);
    } catch (const std::exception& e) {
        if (ctx.rank == 0) std::fprintf(stderr, "[alps_dist_join_scaling_bench] %s\n", e.what());
        exit_code = 1;
    }

    alps::destroy_context(ctx);
    MPI_Finalize();
    return exit_code;
}
