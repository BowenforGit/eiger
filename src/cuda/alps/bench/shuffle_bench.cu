// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_shuffle_bench — shuffle-only (all-to-all) bandwidth microbenchmark.
//
// The sweep knob is the PER-PEER MESSAGE SIZE m: the bytes each GPU sends to
// each of the g peers. Each GPU holds N = g*M elements (M = m/sizeof(T) per
// peer); chunk c is destined for rank c (positional split -> zero partition
// compute, send buffer already in destination order). Times ONLY the NCCL
// all-to-all via alps::collectives::alltoall_column on a pre-allocated recv buf.
//
// Specifying the message size directly (rather than a per-GPU total N split g
// ways) means: (a) every g is sampled at the SAME per-peer message sizes, so the
// bandwidth-vs-message-size curves overlap and are directly comparable; and
// (b) a power-of-two message is inherently 16B-aligned for ANY g — no alignment
// artifact from N/g being an odd element count. Works for any rank count g.
//
//   throughput_gbps     = g*N*sizeof(T)/time   = g^2 * m / time     (all data, incl self-chunk)
//   cross_gpu_gbps       = (g-1)*N*sizeof(T)/time = (g-1)*g*m / time (aggregate bytes on the wire)
//   per_gpu_cross_gbps   = cross_gpu_gbps / g   = (g-1)*m / time     (one GPU's link egress bw)
//
// Two backends, selected by --backend (and --copy/--direction for p2p), share
// the harness:
//   nccl   — alps::collectives::alltoall_column (grouped NCCL Send/Recv)
//   p2p    — alps::collectives::alltoall_p2p, direct fabric transfers; --copy
//            {sm,ce} picks the SM kernel or CE cuMemcpyAsync, and --direction
//            {push,pull} picks remote-store (push) vs remote-load (pull, sm only).
// CSV columns `backend` (nccl/p2p_sm/p2p_ce) + `direction` (na/push/pull) label
// each variant, so one CSV overlays them all.
//
//   mpirun -np 8 ./alps_shuffle_bench --msg-exp-min 16 --msg-exp-max 30 \
//       --iters 20 --warmup 5 --out shuffle.csv [--backend p2p --copy sm --direction pull]
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

#include "alps/collectives/alltoall.hpp"
#include "alps/collectives/alltoall_p2p.hpp"
#include "alps/context.hpp"
#include "alps/error.hpp"

namespace {

using gamax::size_type;
constexpr int kElemBytes = 4; // int32

struct Args {
    int         msg_exp_min{16};  // per-peer message size = 2^exp BYTES (64 KiB)
    int         msg_exp_max{30};  // .. up to 2^30 = 1 GiB per peer
    int         iters{20};
    int         warmup{5};
    int         seed{42};
    bool        odd_count{false}; // per-peer count = M-1 (odd) -> mis-aligned chunk starts
    std::string backend{"nccl"};  // nccl | p2p | ce | ce_nocopy
    std::string copy{"sm"};       // sm | ce   (p2p only)
    std::string direction{"push"};// push | pull (p2p; pull is sm-only)
    std::string counts;           // explicit per-peer element-count sweep (comma list);
                                  // overrides the --msg-exp power-of-two sweep when set
    bool        no_sort{false};   // keep --counts in given order (don't sort ascending)
    std::string tag{"na"};        // free-form label echoed into the CSV `tag` column
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
        else if (k == "--seed")         a.seed         = (int)arg_ll(next());
        else if (k == "--odd-count")    a.odd_count    = true;
        else if (k == "--backend")      a.backend      = next();
        else if (k == "--copy")         a.copy         = next();
        else if (k == "--direction")    a.direction    = next();
        else if (k == "--counts")       a.counts       = next();
        else if (k == "--no-sort")      a.no_sort      = true;
        else if (k == "--tag")          a.tag          = next();
        else if (k == "--out")          a.out          = next();
        else if (k == "--size-exp-min" || k == "--size-exp-max")
            throw std::runtime_error(k + " is gone; the bench now sweeps per-peer message "
                                     "size — use --msg-exp-min/--msg-exp-max (exponent of bytes)");
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.out.empty()) throw std::runtime_error("--out <csv file> is required");
    if (a.msg_exp_min > a.msg_exp_max) throw std::runtime_error("msg-exp-min > msg-exp-max");
    // Require >= 16 B (one uint4): the SM path uses m_u4 = M/4, so M must be a
    // multiple of 4 (and >= 4). For power-of-two messages M = 2^(exp-2), so
    // exp >= 4 guarantees it. size_type shift avoids 32-bit shift-width UB.
    if ((size_type{1} << a.msg_exp_min) < 16)
        throw std::runtime_error("--msg-exp-min too small: per-peer message must be >= 16 bytes (one uint4)");
    if (a.backend != "nccl" && a.backend != "p2p" && a.backend != "ce" &&
        a.backend != "ce_nocopy")
        throw std::runtime_error("--backend must be nccl, p2p, ce, or ce_nocopy");
    if (!a.counts.empty() && a.odd_count)
        throw std::runtime_error("--odd-count cannot be combined with --counts "
                                 "(counts are explicit; --odd-count only shapes the --msg-exp sweep)");
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

// Uniform positional pattern: M = N/g elements to/from each peer at offset p*M.
alps::collectives::CommPattern uniform_pattern(int g, size_type M) {
    alps::collectives::CommPattern pat;
    pat.send_counts.assign(g, M);
    pat.recv_counts.assign(g, M);
    pat.send_offsets.resize(g);
    pat.recv_offsets.resize(g);
    for (int p = 0; p < g; ++p) { pat.send_offsets[p] = (size_type)p * M; pat.recv_offsets[p] = (size_type)p * M; }
    return pat;
}

// One-time correctness check at a small size: send[j]=rank*Nc+j; after the
// all-to-all, recv[p*Mc + k] must equal peer p's chunk for us = p*Nc + rank*Mc + k.
void verify(const alps::Context& ctx, cudaStream_t stream) {
    const int g = ctx.size;
    const size_type Nc = (size_type)g * 1024; // divisible by g, tags fit int32
    const size_type Mc = Nc / g;
    auto send = gamax::make_sequence<int32_t>(
        Nc, stream, static_cast<int32_t>(ctx.rank * Nc), 1);
    auto recv = gamax::make_column<int32_t>(Nc, stream);
    auto pat  = uniform_pattern(g, Mc);
    alps::collectives::alltoall_column(send->view(), recv->mutable_view(), pat, ctx, stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> h(Nc);
    ALPS_CUDACHECK(cudaMemcpy(h.data(), recv->data<int32_t>(), Nc * sizeof(int32_t), cudaMemcpyDeviceToHost));
    int local_fail = 0;
    for (int p = 0; p < g && !local_fail; ++p) {
        for (size_type k = 0; k < Mc; ++k) {
            const int32_t expect = (int32_t)((size_type)p * Nc + (size_type)ctx.rank * Mc + k);
            if (h[(size_type)p * Mc + k] != expect) { local_fail = 1; break; }
        }
    }
    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    if (global_fail) throw std::runtime_error("alltoall correctness check FAILED");
    if (ctx.rank == 0) std::fprintf(stderr, "[alps_shuffle_bench] correctness check OK (g=%d)\n", g);
}

// P2P correctness check on the already-allocated fabric buffers, at a small
// per-peer count. Same layout/expectation as verify(): after the push, this
// rank's recv[s*Mc + k] must equal sender s's chunk for us = s*Nc + rank*Mc + k.
void verify_p2p(const alps::collectives::P2pBuffers& pbuf, alps::collectives::P2pCopy copy,
                alps::collectives::P2pDir dir, bool odd_count,
                const alps::Context& ctx, cudaStream_t stream) {
    const int       g  = ctx.size;
    // Clamp to the per-rank send capacity (send_elems = g*M_max) so the check
    // never overruns the fabric buffers when a small --msg-exp sweep shrinks M_max.
    size_type Mc = std::min<size_type>(odd_count ? 1023 : 1024, pbuf.send_elems / g);
    // Exercise the same alignment regime as the measured sweep. The CE path
    // supports arbitrary byte counts; the SM path's divisibility check below
    // rejects odd-count sweeps before they can be measured.
    if (odd_count && Mc % 2 == 0 && Mc > 1) --Mc;
    const size_type Nc = (size_type)g * Mc;

    // Each rank tags its OWN send buffer; the expected recv layout is the same
    // for push and pull (recv[s*Mc+k] holds sender s's chunk for us).
    std::vector<int32_t> h_send(Nc);
    for (size_type j = 0; j < Nc; ++j) h_send[j] = (int32_t)((size_type)ctx.rank * Nc + j);
    ALPS_CUDACHECK(cudaMemcpy(reinterpret_cast<void*>(pbuf.send.own()), h_send.data(),
                              Nc * sizeof(int32_t), cudaMemcpyHostToDevice));

    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));            // all sends staged (pull reads peers')
    alps::collectives::alltoall_p2p(pbuf, Mc, copy, dir, ctx, stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));            // all transfers landed

    std::vector<int32_t> h(Nc);
    ALPS_CUDACHECK(cudaMemcpy(h.data(), reinterpret_cast<void*>(pbuf.recv.own()),
                              Nc * sizeof(int32_t), cudaMemcpyDeviceToHost));
    int local_fail = 0;
    for (int s = 0; s < g && !local_fail; ++s) {
        for (size_type k = 0; k < Mc; ++k) {
            const int32_t expect = (int32_t)((size_type)s * Nc + (size_type)ctx.rank * Mc + k);
            if (h[(size_type)s * Mc + k] != expect) { local_fail = 1; break; }
        }
    }
    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    if (global_fail) throw std::runtime_error("p2p alltoall correctness check FAILED");
    if (ctx.rank == 0)
        std::fprintf(stderr, "[alps_shuffle_bench] P2P correctness check OK (g=%d, copy=%s, dir=%s)\n",
                     g, copy == alps::collectives::P2pCopy::CE ? "ce" : "sm",
                     dir == alps::collectives::P2pDir::Pull ? "pull" : "push");
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process(); // allocator + MPI env defaults; before MPI_Init
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    int exit_code = 0;
    try {
        Args args = parse_args(argc, argv);
        const int  g       = ctx.size; // any rank count supported (positional all-to-all)
        const bool is_p2p       = (args.backend == "p2p");
        const bool is_ce_nocopy = (args.backend == "ce_nocopy"); // CE transfer core only
        // is_ce covers both CE variants: they share all setup (dst_offsets, the 3x
        // mem clamp, fabric staging) and differ only in the run() call + verify.
        const bool is_ce        = (args.backend == "ce") || is_ce_nocopy;
        const alps::collectives::P2pCopy copy_kind =
            (args.copy == "ce") ? alps::collectives::P2pCopy::CE : alps::collectives::P2pCopy::SM;
        const alps::collectives::P2pDir dir_kind =
            (args.direction == "pull") ? alps::collectives::P2pDir::Pull : alps::collectives::P2pDir::Push;
        // CSV: backend label nccl | ce | ce_nocopy | p2p_sm | p2p_ce; direction na | push | pull.
        const std::string label = is_p2p ? ("p2p_" + args.copy)
                                  : is_ce_nocopy ? std::string("ce_nocopy")
                                  : is_ce ? std::string("ce") : std::string("nccl");
        const std::string dir_s = is_p2p ? args.direction : std::string("na");

        // ce_nocopy never fills recv (it stops after the transfer + completion
        // barrier), so the output check would fail — and is redundant anyway, since
        // the full `ce` backend already validates the whole staging->recv path.
        if (!is_p2p && !is_ce_nocopy) verify(ctx, ctx.stream); // p2p check runs after its buffers exist

        FILE* csv = nullptr;
        if (ctx.rank == 0) {
            const bool need_header = !file_nonempty(args.out);
            csv = std::fopen(args.out.c_str(), "a");
            if (!csv) throw std::runtime_error("cannot open --out file: " + args.out);
            if (need_header) {
                std::fprintf(csv,
                    "tag,backend,direction,g,odd_count,elem_bytes,N_per_gpu,per_peer_count,per_peer_bytes,total_bytes,iters,"
                    "comm_ms_mean,comm_ms_median,comm_ms_std,"
                    "throughput_gbps_mean,throughput_gbps_median,throughput_gbps_std,"
                    "cross_gpu_gbps_mean,cross_gpu_gbps_median,cross_gpu_gbps_std,"
                    "per_gpu_cross_gbps_mean,per_gpu_cross_gbps_median,per_gpu_cross_gbps_std\n");
                std::fflush(csv);
            }
        }

        // Build the ascending per-peer count sweep. Two modes:
        //   --counts c1,c2,...  : explicit per-peer element counts (the driver maps
        //       a per-GPU TOTAL transfer to c = round(total/(g*elem_bytes)), so any
        //       g works — not just powers of two where total/g is exact).
        //   else                : per-peer message = 2^e bytes, e in [min, max].
        // A candidate is kept only if it fits memory on EVERY rank. CE keeps an
        // extra fabric recv staging region (= recv size) beyond the send+recv
        // columns, so its peak is ~3x g*c*elem_bytes vs 2x for nccl/p2p.
        const double mem_mult = is_ce ? 3.0 : 2.0;
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
                    const size_type M = (size_type{1} << e) / kElemBytes;
                    const size_type c = args.odd_count ? (M - 1) : M;
                    if (c >= 1) cand.push_back(c);
                }
            }
            if (!args.no_sort) std::sort(cand.begin(), cand.end());
            // de-dup adjacent equals while preserving order
            cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
            for (size_type c : cand) {
                const double need = mem_mult * (double)g * (double)c * kElemBytes;
                int fits_local = (need <= 0.8 * (double)freeB) ? 1 : 0, fits = 0;
                ALPS_MPICHECK(MPI_Allreduce(&fits_local, &fits, 1, MPI_INT, MPI_MIN, ctx.mpi_comm));
                if (fits) sweep_counts.push_back(c);
                else if (ctx.rank == 0)
                    std::fprintf(stderr, "[alps_shuffle_bench] dropping per-peer count %lld "
                                 "(per-GPU x%.0f ~%.1f GiB > 80%% free)\n",
                                 (long long)c, mem_mult, need / (1024.0 * 1024 * 1024));
            }
            if (sweep_counts.empty())
                throw std::runtime_error("no sweep point fits in 80% of free GPU memory");
        }

        // The p2p SM kernels vectorize the copy in 16B uint4 units (m_u4 = M/4), so
        // a per-peer count not divisible by 4 silently drops its tail. The power-of-
        // two --msg-exp sweep guarantees divisibility; an explicit --counts may not.
        if (is_p2p && copy_kind == alps::collectives::P2pCopy::SM)
            for (size_type c : sweep_counts)
                if (c % 4 != 0)
                    throw std::runtime_error("--backend p2p --copy sm requires every per-peer count "
                                             "to be a multiple of 4 (uint4-vectorized copy)");

        // Allocate send/recv ONCE at the largest size and reuse the leading g*c
        // elements for every message size (avoids per-size cudaMalloc churn).
        const size_type max_count = *std::max_element(sweep_counts.begin(), sweep_counts.end());
        const size_type buf_max   = (size_type)g * max_count;

        // Backend-specific buffers (allocated once). NCCL: gamax columns driven
        // by alltoall_column. P2P: fabric VMM buffers driven by alltoall_p2p.
        std::unique_ptr<gamax::column> send, recv;            // nccl
        alps::collectives::P2pBuffers  pbuf;                  // p2p
        if (is_p2p) {
            pbuf = alps::collectives::p2p_alloc(buf_max, buf_max, ctx); // all2all: send == recv == g*M_max
            verify_p2p(pbuf, copy_kind, dir_kind, args.odd_count, ctx, ctx.stream);
        } else {
            send = gamax::make_column<int32_t>(buf_max, ctx.stream); // contents irrelevant for bw
            recv = gamax::make_column<int32_t>(buf_max, ctx.stream);
        }

        // One all-to-all of the current per-peer count `c`, on ctx.stream.
        auto run = [&](const alps::collectives::CommPattern& pat, size_type c) {
            if (is_p2p)           alps::collectives::alltoall_p2p(pbuf, c, copy_kind, dir_kind, ctx, ctx.stream);
            else if (is_ce_nocopy) alps::collectives::alltoall_column_ce_push(send->view(), pat, ctx, ctx.stream);
            else if (is_ce)       alps::collectives::alltoall_column_ce(send->view(), recv->mutable_view(), pat, ctx, ctx.stream);
            else                  alps::collectives::alltoall_column(send->view(), recv->mutable_view(), pat, ctx, ctx.stream);
        };

        // CE keeps a persistent fabric recv staging region that grows on demand.
        // Pre-grow it to the LARGEST size once (untimed) so the ascending sweep
        // never regrows during a measurement — repeated regrow (teardown + remap
        // of all peers per grow) was observed to degrade the largest-size CE
        // bandwidth. Mirrors the up-front send/recv allocation at buf_max.
        if (is_ce) {
            auto pat0 = uniform_pattern(g, max_count);
            alps::collectives::finalize_dst_offsets(pat0, ctx);
            run(pat0, max_count);
            ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
        }

        for (size_type c : sweep_counts) {
            const size_type N = (size_type)g * c;   // reported N_per_gpu == g * per_peer_count
            auto pat  = uniform_pattern(g, c);
            if (is_ce) alps::collectives::finalize_dst_offsets(pat, ctx); // PUSH dest offsets (once/size)

            // NVTX: one range per swept size (covers warmup + timed iters) so the
            // small- vs large-message phases are locatable in the nsys timeline.
            char nvname[96];
            std::snprintf(nvname, sizeof nvname, "%s g=%d total=%lldB",
                          args.tag.c_str(), g, (long long)((size_type)N * kElemBytes));
            nvtxRangePushA(nvname);

            for (int w = 0; w < args.warmup; ++w) {
                run(pat, c);
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
                run(pat, c);
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
                // total / cross bytes based on the ACTUAL per-peer count c so
                // aligned vs odd are compared on real bytes moved.
                const double total_bytes     = (double)g * g * c * kElemBytes; // g GPUs x (g*c) each
                const double cross_bytes     = (double)(g - 1) * g * c * kElemBytes;
                const double per_gpu_xbytes  = (double)(g - 1) * c * kElemBytes; // cross_bytes / g:
                                              // bytes one GPU pushes onto its links (the effective
                                              // per-GPU NVLink/fabric egress bandwidth)
                const long long per_peer_b   = (long long)c * kElemBytes;
                std::vector<double> thr(args.iters), xthr(args.iters), pgthr(args.iters);
                for (int i = 0; i < args.iters; ++i) {
                    const double s = comm[i] / 1000.0;
                    thr[i]   = s > 0 ? total_bytes / s / 1e9 : 0.0; // GB/s
                    xthr[i]  = s > 0 ? cross_bytes / s / 1e9 : 0.0;
                    pgthr[i] = s > 0 ? per_gpu_xbytes / s / 1e9 : 0.0;
                }
                const double cm = mean_of(comm), tm = mean_of(thr), xm = mean_of(xthr), pm = mean_of(pgthr);
                std::fprintf(csv,
                    "%s,%s,%s,%d,%d,%d,%lld,%lld,%lld,%.0f,%d,"
                    "%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    args.tag.c_str(), label.c_str(), dir_s.c_str(), g, args.odd_count ? 1 : 0, kElemBytes, (long long)N, (long long)c, per_peer_b,
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
        if (ctx.rank == 0) std::fprintf(stderr, "[alps_shuffle_bench] %s\n", e.what());
        exit_code = 1;
    }

    alps::destroy_context(ctx);
    MPI_Finalize();
    return exit_code;
}
