// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_allgather_bench — all-gather bandwidth microbenchmark (sibling of
// alps_shuffle_bench). Each GPU contributes a chunk of M int32 (the swept
// per-peer message size); after the all-gather every GPU holds the full
// concatenation of all g chunks (recv = g*M). Times ONLY the gather.
//
// Backends (--backend, with --copy/--direction for p2p):
//   nccl   — alps::collectives::allgather_column (ncclBroadcast per root)
//   p2p    — alps::collectives::allgather_p2p, direct fabric transfers; --copy
//            {sm,ce} and --direction {push,pull} (pull is sm-only; ce is push).
//
// Byte accounting matches the all2all bench (no NVLS multicast, so a broadcast
// costs (g-1)*M egress per rank):
//   throughput_gbps     = g*g*M*4 / time      (all data each rank ends with, incl self)
//   cross_gpu_gbps       = (g-1)*g*M*4 / time  (aggregate bytes on the wire)
//   per_gpu_cross_gbps   = cross_gpu_gbps / g  (one GPU's link bandwidth)
//
//   mpirun -np 8 ./alps_allgather_bench --msg-exp-min 16 --msg-exp-max 30 \
//       --iters 20 --warmup 5 --out allgather.csv [--backend p2p --copy sm --direction pull]
//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvtx3/nvToolsExt.h>

#include "memory/column.hpp"

#include "alps/collectives/allgather.hpp"
#include "alps/collectives/allgather_p2p.hpp"
#include "alps/collectives/alltoall.hpp" // CommPattern
#include "alps/context.hpp"
#include "alps/error.hpp"

namespace {

using gamax::size_type;
constexpr int kElemBytes = 4; // int32

struct Args {
    int         msg_exp_min{16};   // per-rank chunk size = 2^exp BYTES (64 KiB)
    int         msg_exp_max{30};   // .. up to 2^30 = 1 GiB
    int         iters{20};
    int         warmup{5};
    std::string backend{"nccl"};   // nccl | p2p | ce | ce_nocopy
    std::string copy{"sm"};        // sm | ce   (p2p only)
    std::string direction{"push"}; // push | pull (p2p; pull is sm-only)
    std::string counts;            // explicit per-rank chunk-count sweep (comma list);
                                   // overrides the --msg-exp power-of-two sweep when set
    bool        no_sort{false};    // keep --counts in given order (don't sort ascending)
    std::string tag{"na"};         // free-form label echoed into the CSV `tag` column
    std::string out;
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
        if      (k == "--msg-exp-min")  a.msg_exp_min  = (int)arg_ll(next());
        else if (k == "--msg-exp-max")  a.msg_exp_max  = (int)arg_ll(next());
        else if (k == "--iters")        a.iters        = (int)arg_ll(next());
        else if (k == "--warmup")       a.warmup       = (int)arg_ll(next());
        else if (k == "--backend")      a.backend      = next();
        else if (k == "--copy")         a.copy         = next();
        else if (k == "--direction")    a.direction    = next();
        else if (k == "--counts")       a.counts       = next();
        else if (k == "--no-sort")      a.no_sort      = true;
        else if (k == "--tag")          a.tag          = next();
        else if (k == "--out")          a.out          = next();
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.out.empty()) throw std::runtime_error("--out <csv file> is required");
    if (a.msg_exp_min > a.msg_exp_max) throw std::runtime_error("msg-exp-min > msg-exp-max");
    // Require >= 16 B (one uint4): the SM path uses m_u4 = M/4, so M must be a
    // multiple of 4 (and >= 4) to copy the whole chunk. For power-of-two messages
    // M = 2^(exp-2), so exp >= 4 (>=16 B) guarantees it. size_type shift avoids
    // 32-bit shift-width UB at large exponents.
    if ((size_type{1} << a.msg_exp_min) < 16)
        throw std::runtime_error("--msg-exp-min too small: per-peer chunk must be >= 16 bytes (one uint4)");
    if (a.backend != "nccl" && a.backend != "p2p" && a.backend != "ce" &&
        a.backend != "ce_nocopy")
        throw std::runtime_error("--backend must be nccl, p2p, ce, or ce_nocopy");
    if (a.copy != "sm" && a.copy != "ce")          throw std::runtime_error("--copy must be sm or ce");
    if (a.direction != "push" && a.direction != "pull") throw std::runtime_error("--direction must be push or pull");
    if (a.direction == "pull" && a.copy != "sm")   throw std::runtime_error("--direction pull is supported for --copy sm only");
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
double std_of(const std::vector<double>& v, double mean) {
    if (v.size() < 2) return 0.0;
    double s = 0; for (double x : v) s += (x - mean) * (x - mean);
    return std::sqrt(s / (v.size() - 1));
}

bool file_nonempty(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fclose(f);
    return sz > 0;
}

// Uniform all-gather pattern: every rank contributes M; chunk i lands at i*M.
alps::collectives::CommPattern uniform_pattern(int g, size_type M) {
    alps::collectives::CommPattern pat;
    pat.send_counts.assign(g, M);
    pat.recv_counts.assign(g, M);
    pat.send_offsets.assign(g, 0);          // local chunk at offset 0
    pat.recv_offsets.resize(g);
    for (int p = 0; p < g; ++p) pat.recv_offsets[p] = (size_type)p * M;
    return pat;
}

// recv layout after all-gather: chunk i (from rank i) at recv[i*Mc]. Tag rank's
// send with rank*Nc + k (Nc = g*Mc) so every rank's recv[i*Mc+k] == i*Nc + k.
void verify_nccl(const alps::Context& ctx, cudaStream_t stream) {
    const int g = ctx.size;
    const size_type Mc = 1024;
    const size_type Nc = (size_type)g * Mc;
    auto send = gamax::make_sequence<int32_t>(
        Mc, stream, static_cast<int32_t>(ctx.rank * Nc), 1); // [rank*Nc .. +Mc)
    auto recv = gamax::make_column<int32_t>(Nc, stream);
    auto pat  = uniform_pattern(g, Mc);
    alps::collectives::allgather_column(send->view(), recv->mutable_view(), pat, ctx, stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> h(Nc);
    ALPS_CUDACHECK(cudaMemcpy(h.data(), recv->data<int32_t>(), Nc * sizeof(int32_t), cudaMemcpyDeviceToHost));
    int local_fail = 0;
    for (int s = 0; s < g && !local_fail; ++s)
        for (size_type k = 0; k < Mc; ++k)
            if (h[(size_type)s * Mc + k] != (int32_t)((size_type)s * Nc + k)) { local_fail = 1; break; }
    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    if (global_fail) throw std::runtime_error("allgather correctness check FAILED");
    if (ctx.rank == 0) std::fprintf(stderr, "[alps_allgather_bench] correctness check OK (g=%d)\n", g);
}

void verify_p2p(const alps::collectives::P2pBuffers& pbuf, alps::collectives::P2pCopy copy,
                alps::collectives::P2pDir dir, const alps::Context& ctx, cudaStream_t stream) {
    const int g = ctx.size;
    // Clamp to the allocated send capacity (M_max) so the check never overruns
    // the fabric buffers when a small --msg-exp sweep makes M_max < 1024.
    const size_type Mc = std::min<size_type>(1024, pbuf.send_elems);
    const size_type Nc = (size_type)g * Mc;

    std::vector<int32_t> h_send(Mc);
    for (size_type k = 0; k < Mc; ++k) h_send[k] = (int32_t)((size_type)ctx.rank * Nc + k);
    ALPS_CUDACHECK(cudaMemcpy(reinterpret_cast<void*>(pbuf.send.own()), h_send.data(),
                              Mc * sizeof(int32_t), cudaMemcpyHostToDevice));

    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));            // all sends staged (pull reads peers')
    alps::collectives::allgather_p2p(pbuf, Mc, copy, dir, ctx, stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));            // all transfers landed

    std::vector<int32_t> h(Nc);
    ALPS_CUDACHECK(cudaMemcpy(h.data(), reinterpret_cast<void*>(pbuf.recv.own()),
                              Nc * sizeof(int32_t), cudaMemcpyDeviceToHost));
    int local_fail = 0;
    for (int s = 0; s < g && !local_fail; ++s)
        for (size_type k = 0; k < Mc; ++k)
            if (h[(size_type)s * Mc + k] != (int32_t)((size_type)s * Nc + k)) { local_fail = 1; break; }
    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    if (global_fail) throw std::runtime_error("p2p allgather correctness check FAILED");
    if (ctx.rank == 0)
        std::fprintf(stderr, "[alps_allgather_bench] P2P correctness check OK (g=%d, copy=%s, dir=%s)\n",
                     g, copy == alps::collectives::P2pCopy::CE ? "ce" : "sm",
                     dir == alps::collectives::P2pDir::Pull ? "pull" : "push");
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process();
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    int exit_code = 0;
    try {
        Args args = parse_args(argc, argv);
        const int  g       = ctx.size;
        const bool is_p2p       = (args.backend == "p2p");
        const bool is_ce_nocopy = (args.backend == "ce_nocopy"); // CE transfer core only
        // is_ce covers both CE variants: same setup, differ only in run() + verify.
        const bool is_ce        = (args.backend == "ce") || is_ce_nocopy;
        const alps::collectives::P2pCopy copy_kind =
            (args.copy == "ce") ? alps::collectives::P2pCopy::CE : alps::collectives::P2pCopy::SM;
        const alps::collectives::P2pDir dir_kind =
            (args.direction == "pull") ? alps::collectives::P2pDir::Pull : alps::collectives::P2pDir::Push;
        const std::string label = is_p2p ? ("p2p_" + args.copy)
                                  : is_ce_nocopy ? std::string("ce_nocopy")
                                  : is_ce ? std::string("ce") : std::string("nccl");
        const std::string dir_s = is_p2p ? args.direction : std::string("na");

        // ce_nocopy never fills recv (stops after the broadcast-PUSH + completion
        // barrier); skip the output check (redundant with the full `ce` backend).
        if (!is_p2p && !is_ce_nocopy) verify_nccl(ctx, ctx.stream);

        FILE* csv = nullptr;
        if (ctx.rank == 0) {
            const bool need_header = !file_nonempty(args.out);
            csv = std::fopen(args.out.c_str(), "a");
            if (!csv) throw std::runtime_error("cannot open --out file: " + args.out);
            if (need_header) {
                std::fprintf(csv,
                    "tag,backend,direction,g,elem_bytes,recv_per_gpu,per_peer_count,per_peer_bytes,total_bytes,iters,"
                    "comm_ms_mean,comm_ms_median,comm_ms_std,"
                    "throughput_gbps_mean,throughput_gbps_median,throughput_gbps_std,"
                    "cross_gpu_gbps_mean,cross_gpu_gbps_median,cross_gpu_gbps_std,"
                    "per_gpu_cross_gbps_mean,per_gpu_cross_gbps_median,per_gpu_cross_gbps_std\n");
                std::fflush(csv);
            }
        }

        // Build the ascending per-rank chunk-count sweep. --counts c1,c2,... gives
        // explicit per-rank chunk element counts (the driver maps a per-GPU TOTAL
        // gather size to c = round(total/(g*elem_bytes)), so any g works); else the
        // per-rank chunk = 2^e bytes, e in [min,max]. Memory per GPU: send(M) +
        // recv(g*M) = (g+1)*M; CE adds a fabric recv staging region (g*M) -> (2g+1)*M.
        // A candidate is kept only if it fits on EVERY rank.
        size_t freeB = 0, freeTotB = 0;
        ALPS_CUDACHECK(cudaMemGetInfo(&freeB, &freeTotB));

        std::vector<size_type> sweep_counts;
        {
            std::vector<size_type> cand;
            if (!args.counts.empty()) {
                std::stringstream ss(args.counts);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    if (tok.empty()) continue;
                    long long c = std::strtoll(tok.c_str(), nullptr, 10);
                    if (c >= 1) cand.push_back((size_type)c);
                }
            } else {
                for (int e = args.msg_exp_min; e <= args.msg_exp_max; ++e) {
                    if ((size_type{1} << e) % kElemBytes != 0) continue;
                    cand.push_back((size_type{1} << e) / kElemBytes);
                }
            }
            if (!args.no_sort) std::sort(cand.begin(), cand.end());
            cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
            const double mult = is_ce ? (2.0 * g + 1.0) : (double)(g + 1);
            for (size_type M : cand) {
                const double need = mult * (double)M * kElemBytes;
                int fits_local = (need <= 0.8 * (double)freeB) ? 1 : 0, fits = 0;
                ALPS_MPICHECK(MPI_Allreduce(&fits_local, &fits, 1, MPI_INT, MPI_MIN, ctx.mpi_comm));
                if (fits) sweep_counts.push_back(M);
                else if (ctx.rank == 0)
                    std::fprintf(stderr, "[alps_allgather_bench] dropping chunk count %lld "
                                 "(per-GPU ~%.1f GiB > 80%% free)\n",
                                 (long long)M, need / (1024.0 * 1024 * 1024));
            }
            if (sweep_counts.empty())
                throw std::runtime_error("no sweep point fits in 80% of free GPU memory");
        }

        // p2p SM copies in 16B uint4 units (m_u4 = M/4): a chunk count not divisible
        // by 4 silently drops its tail. Power-of-two --msg-exp guarantees it; an
        // explicit --counts list may not.
        if (is_p2p && copy_kind == alps::collectives::P2pCopy::SM)
            for (size_type M : sweep_counts)
                if (M % 4 != 0)
                    throw std::runtime_error("--backend p2p --copy sm requires every chunk count "
                                             "to be a multiple of 4 (uint4-vectorized copy)");

        const size_type M_max     = *std::max_element(sweep_counts.begin(), sweep_counts.end());
        const size_type recv_max  = (size_type)g * M_max;

        // Allocate ONCE; reuse the leading M (send) / g*M (recv) each size.
        std::unique_ptr<gamax::column> send, recv;          // nccl
        alps::collectives::P2pBuffers  pbuf;                 // p2p
        if (is_p2p) {
            pbuf = alps::collectives::p2p_alloc(/*send=*/M_max, /*recv=*/recv_max, ctx);
            verify_p2p(pbuf, copy_kind, dir_kind, ctx, ctx.stream);
        } else {
            send = gamax::make_column<int32_t>(M_max, ctx.stream);
            recv = gamax::make_column<int32_t>(recv_max, ctx.stream);
        }

        auto run = [&](const alps::collectives::CommPattern& pat, size_type M) {
            if (is_p2p)            alps::collectives::allgather_p2p(pbuf, M, copy_kind, dir_kind, ctx, ctx.stream);
            else if (is_ce_nocopy) alps::collectives::allgather_column_ce_push(send->view(), pat, ctx, ctx.stream);
            else if (is_ce)        alps::collectives::allgather_column_ce(send->view(), recv->mutable_view(), pat, ctx, ctx.stream);
            else                   alps::collectives::allgather_column(send->view(), recv->mutable_view(), pat, ctx, ctx.stream);
        };

        // CE keeps a persistent fabric recv staging region that grows on demand;
        // pre-grow it to the largest size once (untimed) so the ascending sweep
        // never regrows mid-measurement (see the shuffle bench for the rationale).
        if (is_ce) {
            auto pat0 = uniform_pattern(g, M_max);
            alps::collectives::finalize_dst_offsets(pat0, ctx);
            run(pat0, M_max);
            ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
        }

        for (size_type M : sweep_counts) {
            auto pat = uniform_pattern(g, M);
            if (is_ce) alps::collectives::finalize_dst_offsets(pat, ctx); // PUSH dest offsets (once/size)

            char nvname[96];
            std::snprintf(nvname, sizeof nvname, "%s g=%d total=%lldB",
                          args.tag.c_str(), g, (long long)((size_type)g * M * kElemBytes));
            nvtxRangePushA(nvname);

            for (int w = 0; w < args.warmup; ++w) {
                run(pat, M);
                ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
            }

            std::vector<double> comm; // rank 0 only
            if (ctx.rank == 0) comm.reserve(args.iters);
            for (int it = 0; it < args.iters; ++it) {
                cudaEvent_t e0, e1;
                ALPS_CUDACHECK(cudaEventCreate(&e0));
                ALPS_CUDACHECK(cudaEventCreate(&e1));
                ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
                ALPS_CUDACHECK(cudaEventRecord(e0, ctx.stream));
                run(pat, M);
                ALPS_CUDACHECK(cudaEventRecord(e1, ctx.stream));
                ALPS_CUDACHECK(cudaEventSynchronize(e1));
                float ms = 0; ALPS_CUDACHECK(cudaEventElapsedTime(&ms, e0, e1));
                cudaEventDestroy(e0); cudaEventDestroy(e1);
                float mx = 0;
                ALPS_MPICHECK(MPI_Reduce(&ms, &mx, 1, MPI_FLOAT, MPI_MAX, 0, ctx.mpi_comm));
                if (ctx.rank == 0) comm.push_back(mx);
            }
            nvtxRangePop(); // per-size range

            if (ctx.rank == 0) {
                const double total_bytes    = (double)g * g * M * kElemBytes;   // each rank gathers g*M
                const double cross_bytes    = (double)(g - 1) * g * M * kElemBytes;
                const double per_gpu_xbytes = (double)(g - 1) * M * kElemBytes;  // cross / g
                const long long per_peer_b  = (long long)M * kElemBytes;
                const long long recv_per_gpu = (long long)g * M;
                std::vector<double> thr(args.iters), xthr(args.iters), pgthr(args.iters);
                for (int i = 0; i < args.iters; ++i) {
                    const double s = comm[i] / 1000.0;
                    thr[i]   = s > 0 ? total_bytes / s / 1e9 : 0.0;
                    xthr[i]  = s > 0 ? cross_bytes / s / 1e9 : 0.0;
                    pgthr[i] = s > 0 ? per_gpu_xbytes / s / 1e9 : 0.0;
                }
                const double cm = mean_of(comm), tm = mean_of(thr), xm = mean_of(xthr), pm = mean_of(pgthr);
                std::fprintf(csv,
                    "%s,%s,%s,%d,%d,%lld,%lld,%lld,%.0f,%d,"
                    "%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    args.tag.c_str(), label.c_str(), dir_s.c_str(), g, kElemBytes, recv_per_gpu, (long long)M, per_peer_b,
                    total_bytes, args.iters,
                    cm, median_of(comm), std_of(comm, cm),
                    tm, median_of(thr), std_of(thr, tm),
                    xm, median_of(xthr), std_of(xthr, xm),
                    pm, median_of(pgthr), std_of(pgthr, pm));
                std::fflush(csv);
            }
        }
        if (is_p2p) alps::collectives::p2p_free(pbuf, ctx);
        if (csv) std::fclose(csv);
    } catch (const std::exception& e) {
        if (ctx.rank == 0) std::fprintf(stderr, "[alps_allgather_bench] %s\n", e.what());
        exit_code = 1;
    }

    alps::destroy_context(ctx);
    MPI_Finalize();
    return exit_code;
}
