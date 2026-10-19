// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_dist_join_compare_bench -- compare the fabric-striped shared-static-map
// join (alps::join::shared_map_join) against the NCCL-based dist_join
// (shuffle / broadcast) on a PK-FK inner equi-join, with a per-phase time
// breakdown. One invocation = one dtype, one (left_per_gpu, right_per_gpu)
// point, one GPU count; it runs the selected strategies over the SAME generated
// tables and appends one CSV row per strategy.
//
// Data model (uniform PK-FK, tunable match rate m via --match-rate, default 1.0):
//   - PK (left, build): globally-unique keys, rank r owns [r*L, (r+1)*L);
//     payload = key. (Same scheme as shared_map_bench.)
//   - FK (right, probe): R per-rank keys, each an independent uniform draw from
//     the EXPANDED range [0, ceil(world*L / m)); PK occupies [0, world*L) of that,
//     so a fraction m of FK rows match a PK and the rest miss (keys in
//     [world*L, range)). Output ~ m*world*R rows. m=1 => the classic 100% match.
//     Misses are uniform => no shuffle-partition skew. payload = key.
// Specify per-GPU sizes in #elements (--left-per-gpu / --right-per-gpu); the
// driver varies them for strong/weak scaling.
//
// Timing: per iteration MPI_Barrier -> the join times its own phases (CUDA
// events, internal sync) -> MPI_Allreduce(MAX) each phase (critical path) ->
// mean/median over iters. Wide-superset CSV: shared_map fills construct/build/
// probe/compact; NCCL fills partition/comm/join; total = the strategy's headline
// sum (shared_map: build+probe+compact, construct EXCLUDED; NCCL: all three).
//
// Defaults: RMM arena and NCCL_MIN_CTAS=NCCL_MAX_CTAS=64 are setenv'd with
// overwrite=0 (a launcher -x still wins). A strategy whose
// estimated peak exceeds the GPU is skipped-and-logged (OOM-skip).
//
// Example (4 GPUs / 1 node, strong-scaling point):
//   GAMAX_RMM_RESOURCE=arena GAMAX_MEM_RESERVE_BYTES=100000000000 \
//   mpirun -np 4 alps_dist_join_compare_bench --left-per-gpu 16777216 \
//       --right-per-gpu 805306368 --dtype int32 --strategy all \
//       --iters 10 --warmup 3 --out compare.csv

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>            // driver API: CUDA green contexts (SM partitioning)
#include <mpi.h>
#include <nvml.h>
#include <nvtx3/nvToolsExt.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/join/join.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

#include "alps/context.hpp"
#include "alps/error.hpp"
#include "alps/containers/fabric_region.hpp"
#include "alps/operators/dist_join.hpp"
#include "alps/operators/shared_map_join.hpp"

namespace {

using gamax::size_type;

struct Args {
    long long   left_per_gpu  = 0;       // PK rows per rank  (--left-per-gpu)
    long long   right_per_gpu = 0;       // FK rows per rank  (--right-per-gpu)
    std::string dtype         = "int32"; // int32 | int64 (key+payload width)
    std::string strategy      = "all";   // all | shared_map | shuffle | broadcast
    int         cg            = 4;       // shared_map cooperative-group size
    double      lf            = 0.5;     // shared_map load factor
    int         iters         = 10;
    int         warmup        = 3;
    int         seed          = 42;
    bool        nvlink        = false;   // --nvlink-bytes: measure per-join NVLink traffic via NVML
    std::string local_join    = "static_map"; // static_map | gamax (NCCL paths' per-rank local join)
    std::string mode          = "narrow"; // narrow (1 key + 1 value) | wide (int64, P_L/P_R payloads)
    int         left_payloads = 1;       // wide: P_L payload cols on the PK side (--left-payloads)
    int         right_payloads= 1;       // wide: P_R payload cols on the FK side (--right-payloads)
    double      match_rate    = 1.0;     // fraction of FK rows that match a PK (--match-rate, (0,1])
    int         green_sms     = 0;       // >0: run the whole join (compute+NCCL) in a green ctx of ~N SMs
    std::string out;                     // required CSV
};

long long arg_ll(const char* v) { return std::strtoll(v, nullptr, 10); }

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* f) -> const char* {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + f);
            return argv[++i];
        };
        std::string k = argv[i];
        if      (k == "--left-per-gpu")  a.left_per_gpu  = arg_ll(next("--left-per-gpu"));
        else if (k == "--right-per-gpu") a.right_per_gpu = arg_ll(next("--right-per-gpu"));
        else if (k == "--dtype")         a.dtype         = next("--dtype");
        else if (k == "--strategy")      a.strategy      = next("--strategy");
        else if (k == "--cg")            a.cg            = (int)arg_ll(next("--cg"));
        else if (k == "--lf")            a.lf            = std::atof(next("--lf"));
        else if (k == "--iters")         a.iters         = (int)arg_ll(next("--iters"));
        else if (k == "--warmup")        a.warmup        = (int)arg_ll(next("--warmup"));
        else if (k == "--seed")          a.seed          = (int)arg_ll(next("--seed"));
        else if (k == "--nvlink-bytes")  a.nvlink        = true;
        else if (k == "--local-join")    a.local_join    = next("--local-join");
        else if (k == "--mode")          a.mode          = next("--mode");
        else if (k == "--left-payloads") a.left_payloads = (int)arg_ll(next("--left-payloads"));
        else if (k == "--right-payloads")a.right_payloads= (int)arg_ll(next("--right-payloads"));
        else if (k == "--match-rate")    a.match_rate    = std::atof(next("--match-rate"));
        else if (k == "--green-sms")     a.green_sms     = (int)arg_ll(next("--green-sms"));
        else if (k == "--out")           a.out           = next("--out");
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.out.empty())          throw std::runtime_error("--out <csv> is required");
    if (a.left_per_gpu <= 0)    throw std::runtime_error("--left-per-gpu must be > 0");
    if (a.right_per_gpu <= 0)   throw std::runtime_error("--right-per-gpu must be > 0");
    if (a.dtype != "int32" && a.dtype != "int64")
        throw std::runtime_error("--dtype must be int32 or int64");
    if (a.strategy != "all" && a.strategy != "shared_map" && a.strategy != "shuffle" &&
        a.strategy != "broadcast")
        throw std::runtime_error("--strategy must be all|shared_map|shuffle|broadcast");
    if (a.cg != 1 && a.cg != 2 && a.cg != 4) throw std::runtime_error("--cg must be 1, 2, or 4");
    if (a.local_join != "static_map" && a.local_join != "gamax")
        throw std::runtime_error("--local-join must be static_map or gamax");
    if (a.mode != "narrow" && a.mode != "wide")
        throw std::runtime_error("--mode must be narrow or wide");
    if (a.match_rate <= 0.0 || a.match_rate > 1.0)
        throw std::runtime_error("--match-rate must be in (0, 1]");
    if (a.green_sms < 0) throw std::runtime_error("--green-sms must be >= 0 (0 = disabled)");
    if (a.mode == "wide") {
        // Wide is int64-only and uses the static_map local join (multiset is out of
        // scope for the wide comparison); P_L/P_R are the per-side payload counts.
        if (a.dtype != "int64")        throw std::runtime_error("--mode wide requires --dtype int64");
        if (a.local_join != "static_map") throw std::runtime_error("--mode wide requires --local-join static_map");
        if (a.left_payloads  < 0)      throw std::runtime_error("--mode wide requires --left-payloads >= 0");
        if (a.right_payloads < 0)      throw std::runtime_error("--mode wide requires --right-payloads >= 0");
    } else {
        a.left_payloads = 1; a.right_payloads = 1;   // narrow is exactly [key, value]
    }
    // >= 1 warmup so the one-time fabric-region creation / first-collective setup
    // lands in an untimed iteration, not in the first measured one.
    if (a.warmup < 1) throw std::runtime_error("--warmup must be >= 1");
    if (a.iters  < 1) throw std::runtime_error("--iters must be >= 1");
    return a;
}

// ---- CUDA green context (benchmark-only SM partitioning) ---------------------
// Create a green context holding ~n_sms SMs on the bound device + a stream in it,
// then SWAP ctx.stream to that stream: since the join issues ALL device work
// (compute kernels AND NCCL collectives) on ctx.stream, the entire join runs on
// the SM partition. Driver API (CUDA 13 has no runtime green-ctx API). Isolated
// to this bench -- gamax_alps / alps::make_context are untouched. Actual partition
// size rounds UP to the device SM alignment (returned; e.g. want 66 -> 72 on GB200).
#define ALPS_DRVCHECK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = nullptr; \
    cuGetErrorString(r_, &s_); throw std::runtime_error(std::string("CUDA driver error: ") + #x + " -> " + \
    (s_ ? s_ : "?")); } } while (0)

struct GreenCtx { CUgreenCtx gctx{}; CUstream gstream{}; cudaStream_t orig{}; bool active{false}; unsigned sms{0}; };

unsigned green_setup(alps::Context& ctx, int n_sms, GreenCtx& g) {
    ALPS_DRVCHECK(cuInit(0));
    CUdevice dev; ALPS_DRVCHECK(cuDeviceGet(&dev, ctx.local_device_id));
    CUdevResource total{}; ALPS_DRVCHECK(cuDeviceGetDevResource(dev, &total, CU_DEV_RESOURCE_TYPE_SM));
    if ((unsigned)n_sms > total.sm.smCount) n_sms = (int)total.sm.smCount;   // clamp to device
    CUdevResource res[64] = {}; CUdevResource remain{}; unsigned nb = 64;
    ALPS_DRVCHECK(cuDevSmResourceSplitByCount(res, &nb, &total, &remain, 0u, (unsigned)n_sms));
    if (nb == 0) throw std::runtime_error("green-ctx split produced 0 groups");
    CUdevResourceDesc desc; ALPS_DRVCHECK(cuDevResourceGenerateDesc(&desc, &res[0], 1));
    ALPS_DRVCHECK(cuGreenCtxCreate(&g.gctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUdevResource gres{}; ALPS_DRVCHECK(cuGreenCtxGetDevResource(g.gctx, &gres, CU_DEV_RESOURCE_TYPE_SM));
    ALPS_DRVCHECK(cuGreenCtxStreamCreate(&g.gstream, g.gctx, CU_STREAM_NON_BLOCKING, 0));
    g.orig = ctx.stream; g.sms = gres.sm.smCount; g.active = true;
    ctx.stream = reinterpret_cast<cudaStream_t>(g.gstream);   // route the whole join onto the partition
    return g.sms;
}
void green_teardown(alps::Context& ctx, GreenCtx& g) {
    if (!g.active) return;
    cudaStreamSynchronize(ctx.stream);
    ctx.stream = g.orig;                       // restore so destroy_context tears the real stream down
    cuStreamDestroy(g.gstream);
    cuGreenCtxDestroy(g.gctx);
    g.active = false;
}

// ---- statistics --------------------------------------------------------------
struct Stat { float mean{0}, median{0}; };
Stat summarize(std::vector<float> v) {
    Stat s;
    if (v.empty()) return s;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    std::sort(v.begin(), v.end());
    s.median = v[v.size() / 2];
    return s;
}
double median_d(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---- NVML: NVLink traffic over a whole join op (--nvlink-bytes) --------------
// Same counters/idiom as bench/shared_map_bench.cu. DATA = user payload, RAW =
// wire (payload + protocol); scopeId=UINT_MAX aggregates over all 18 links (the
// default scopeId=0 reads link 0 only ~ aggregate/18). Read before/after a
// strategy's join with a trailing MPI_Barrier so the window also covers peers
// serving this rank's remote requests; summed across ranks = total job GiB.
nvmlDevice_t g_nvmldev{}; bool g_nvml_ok = false;
void nvml_setup(int cuda_dev) {
    if (nvmlInit_v2() != NVML_SUCCESS) { std::fprintf(stderr, "[nvml] init failed\n"); return; }
    char bus[64] = {0};
    if (cudaDeviceGetPCIBusId(bus, sizeof bus, cuda_dev) != cudaSuccess) {
        std::fprintf(stderr, "[nvml] PciBusId failed\n"); return; }
    if (nvmlDeviceGetHandleByPciBusId_v2(bus, &g_nvmldev) != NVML_SUCCESS) {
        std::fprintf(stderr, "[nvml] handle failed\n"); return; }
    g_nvml_ok = true;
}
struct NvlBytes { unsigned long long tx{0}, rx{0}, tx_raw{0}, rx_raw{0}; }; // KiB, aggregate links
NvlBytes nvml_read() {
    NvlBytes b; if (!g_nvml_ok) return b;
    nvmlFieldValue_t fv[4] = {};
    fv[0].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX; fv[1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
    fv[2].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX;  fv[3].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX;
    for (auto& f : fv) f.scopeId = UINT_MAX;
    if (nvmlDeviceGetFieldValues(g_nvmldev, 4, fv) == NVML_SUCCESS) {
        if (fv[0].nvmlReturn == NVML_SUCCESS) b.tx     = fv[0].value.ullVal;
        if (fv[1].nvmlReturn == NVML_SUCCESS) b.rx     = fv[1].value.ullVal;
        if (fv[2].nvmlReturn == NVML_SUCCESS) b.tx_raw = fv[2].value.ullVal;
        if (fv[3].nvmlReturn == NVML_SUCCESS) b.rx_raw = fv[3].value.ullVal;
    }
    return b;
}
double reduce_gib(unsigned long long kib, MPI_Comm comm) {
    unsigned long long g = 0;
    MPI_Allreduce(&kib, &g, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    return static_cast<double>(g) / (1024.0 * 1024.0);
}

// ---- data generation (device-side, uniform) ---------------------------------
__device__ inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

template <typename T>
__global__ void fill_pk_key(T* key, long long n, long long base) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) key[i] = static_cast<T>(base + i);  // globally-unique PK id
}

// Match-rate control (selectivity): draw uniformly from [0, draw_range) where
// draw_range = ceil(total_pk / match_rate). PK occupies [0, total_pk) inside that
// range, so P(key < total_pk) = match_rate in expectation -> a fraction match_rate
// of FK rows hit a PK; the rest land in [total_pk, draw_range) and MISS (the map
// only holds [0,total_pk)). Misses are spread uniformly, so they add no shuffle-
// partition skew. At match_rate=1, draw_range==total_pk -> every key matches (the
// original scheme, byte-identical). Keys stay >= 0 and < draw_range (the caller's
// range guard ensures draw_range-1 fits T; else no negative wrap into the sentinel).
template <typename T>
__global__ void fill_fk_key(T* key, long long n, unsigned long long draw_range,
                            unsigned long long seed, int rank) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    // Per-rank stream base (rank folded in through a full mix), then mix the row
    // index -> independent uniform draws across ranks and rows.
    const uint64_t base = splitmix64(seed + static_cast<uint64_t>(rank) * 0x9e3779b97f4a7c15ull);
    for (; i < n; i += stride) {
        uint64_t h = splitmix64(base + static_cast<uint64_t>(i));
        key[i] = static_cast<T>(h % draw_range);   // uniform over [0,draw_range); < total_pk matches
    }
}

// Expanded FK draw range for a target match rate: ceil(total_pk / match_rate).
// match_rate=1 (or >=1) collapses to total_pk (the exact-match scheme). Kept as a
// free function so the FK gen and the deterministic range guard agree exactly.
inline unsigned long long fk_draw_range(unsigned long long total_pk, double match_rate) {
    if (match_rate >= 1.0) return total_pk;
    return static_cast<unsigned long long>(std::ceil(static_cast<double>(total_pk) / match_rate));
}

// Payload column j: pay_j[i] = key[i] + j. The +j offset makes columns
// distinguishable, so the output validator catches wrong-index / wrong-gather AND
// column-permutation bugs. Narrow (1 payload, j=0) => pay_0 = key, the original scheme.
template <typename T>
__global__ void fill_payload(T* pay, const T* key, long long n, long long j) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) pay[i] = static_cast<T>(key[i] + static_cast<T>(j));
}

// Output validator: with pay_j = key + j and 100% match, every output column k must
// equal right_key + expected_offset(k). `ref` = the right_key output column; counts
// col[i] != ref[i] + offset. Catches right-count-but-wrong-payload joins, a bad
// fabric gather index, or a column permutation.
template <typename T>
__global__ void count_neq_offset(const T* col, const T* ref, long long n, long long offset,
                                 unsigned long long* bad) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    unsigned long long local = 0;
    for (; i < n; i += stride)
        if (col[i] != static_cast<T>(ref[i] + static_cast<T>(offset))) ++local;
    if (local) atomicAdd(bad, local);
}

// Build a side = [key, pay_0, ..., pay_{n_payloads-1}] (key at column 0). Narrow
// passes n_payloads=1 -> [key, key] (== the original 2-col table).
template <typename T>
std::unique_ptr<gamax::table> build_table(long long n, bool is_pk, long long base_or_total,
                                          int n_payloads, const Args& a, const alps::Context& ctx) {
    const int threads = 256;
    const int blocks  = static_cast<int>(std::min<long long>((n + threads - 1) / threads, 4096));
    auto kc = gamax::make_column<T>(static_cast<size_type>(n), ctx.stream);
    if (is_pk)
        fill_pk_key<T><<<blocks, threads, 0, ctx.stream>>>(kc->template data<T>(), n, base_or_total);
    else {
        // base_or_total == total_pk here; widen the draw range to hit match_rate.
        const unsigned long long draw_range =
            fk_draw_range(static_cast<unsigned long long>(base_or_total), a.match_rate);
        fill_fk_key<T><<<blocks, threads, 0, ctx.stream>>>(
            kc->template data<T>(), n, draw_range,
            static_cast<unsigned long long>(a.seed) + 0x1234ull, ctx.rank);
    }
    std::vector<std::unique_ptr<gamax::column>> cols;
    cols.push_back(std::move(kc));
    const T* key_ptr = cols[0]->template data<T>();
    for (int j = 0; j < n_payloads; ++j) {
        auto pc = gamax::make_column<T>(static_cast<size_type>(n), ctx.stream);
        fill_payload<T><<<blocks, threads, 0, ctx.stream>>>(pc->template data<T>(), key_ptr, n, j);
        cols.push_back(std::move(pc));
    }
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    return gamax::make_table(std::move(cols));
}

// ---- OOM pre-check -----------------------------------------------------------
// Coarse per-GPU peak-footprint estimate (BYTES) per strategy, computed ONLY from
// the args (identical on every rank) and a rank-invariant device-capacity budget.
// That makes the skip decision DETERMINISTIC and identical across ranks, so all
// ranks skip-or-run a (collective) strategy together -- a per-rank cudaMemGetInfo
// (free differs across ranks/fragmentation) would let ranks diverge and deadlock
// in the join's internal MPI/NCCL. Footprints sum pool + out-of-pool (the
// shared-map fabric slab is raw cuMem outside the pool, slab/GPU = 2*L*esz/lf;
// broadcast replicates the whole PK to every rank).
unsigned long long est_bytes(const char* strat, const Args& a, int world, std::size_t esz) {
    const unsigned long long L = (unsigned long long)a.left_per_gpu;
    const unsigned long long R = (unsigned long long)a.right_per_gpu;
    const unsigned long long total_pk = L * (unsigned long long)world;
    const unsigned long long tables   = (L + R) * 2ull * esz;            // PK + FK, 2 cols each
    if (!std::strcmp(strat, "shared_map")) {
        const unsigned long long fabric = (unsigned long long)(2.0 * (double)L * (double)esz / a.lf);
        return tables + fabric + 6ull * R * esz;                         // out(3R) + found(R) + cub(2R)
    }
    if (!std::strcmp(strat, "shuffle"))
        return tables + 4ull * (L + R) * esz + 4ull * R * esz + 2ull * L * esz;
    return tables + 4ull * total_pk * esz + 4ull * R * esz;              // broadcast: replicate full PK
}

// Wide-mode POOL-RESIDENT footprint (int64). Returns only what lives in the
// RMM arena; the shared-map slot array + P_L payload slabs live outside the arena
// (in the reserve), so they are NOT counted here -- the caller checks this against
// (budget - reserve) (see run_all). Columns per side = 1 + payloads. The local map
// (cuco slots via the pool) is built on L for shuffle (local shard) but on the FULL
// gathered PK (total_pk) for broadcast -- the latter dominates and is why broadcast
// OOMs first at large PK. Same rank-invariant intent as est_bytes.
unsigned long long est_bytes_wide(const char* strat, const Args& a, int world) {
    const unsigned long long esz = 8;   // int64
    const unsigned long long L = (unsigned long long)a.left_per_gpu;
    const unsigned long long R = (unsigned long long)a.right_per_gpu;
    const unsigned long long total_pk = L * (unsigned long long)world;
    const unsigned long long PL = (unsigned long long)a.left_payloads;
    const unsigned long long PR = (unsigned long long)a.right_payloads;
    const unsigned long long nL = 1 + PL, nR = 1 + PR;               // columns per side
    const unsigned long long tables = (nL * L + nR * R) * esz;        // PK + FK shards (pool)
    const unsigned long long out    = (PL + 1 + PR) * R * esz;        // output (100% match => R rows)
    const unsigned long long idx    = 4ull * R * esz;                // found + left_idx + right_pos + slack
    // local cuco map = capacity/lf * sizeof(pair<i64,i64>) = 4*keys*esz.
    if (!std::strcmp(strat, "shared_map"))
        return tables + out + idx + L * esz /*ids*/;                 // slots+payloads are in the reserve
    if (!std::strcmp(strat, "shuffle"))
        return tables + out + idx + 2ull * (nL * L + nR * R) * esz /*send+recv*/ + 4ull * L * esz /*local map on L*/;
    // broadcast: gather the smaller side (PK here) to every rank + local map on the FULL PK.
    return tables + out + idx + nL * total_pk * esz /*gathered PK*/ + 4ull * total_pk * esz /*local map on total_pk*/;
}

bool should_skip(const char* strat, const Args& a, int world, std::size_t esz,
                 unsigned long long budget_bytes) {
    const unsigned long long e = (a.mode == "wide") ? est_bytes_wide(strat, a, world)
                                                    : est_bytes(strat, a, world, esz);
    return e >= budget_bytes;
}

// ---- CSV ---------------------------------------------------------------------
const char* kHeader =
    "gpus,dtype,strategy,left_per_gpu,right_per_gpu,total_left,total_right,cg,lf,iters,"
    "construct_ms_mean,construct_ms_median,build_ms_mean,build_ms_median,"
    "probe_ms_mean,probe_ms_median,compact_ms_mean,compact_ms_median,"
    "partition_ms_mean,partition_ms_median,comm_ms_mean,comm_ms_median,"
    "join_ms_mean,join_ms_median,total_ms_mean,total_ms_median,out_rows,mrows_s";
// Always appended (both modes): the requested match rate (selectivity). Expected
// out_rows = match_rate*total_right and the measured rate = out_rows/total_right are
// both derived downstream; only the requested m is recorded here. m=1 => 100% match.
const char* kHeaderSel = ",match_rate,green_sms";   // green_sms = SM-partition size (0 = full device, no green ctx)
// Always appended (both modes share ONE schema): mode + per-side payload counts, and
// the wide-only shared_map timers (0 in narrow, and 0 for NCCL rows -- their payload
// movement is in comm_ms, their local gather in join_ms).
const char* kHeaderWide = ",mode,left_payloads,right_payloads,"
    "payload_copy_ms_mean,payload_copy_ms_median,gather_ms_mean,gather_ms_median";
// Appended only with --nvlink-bytes (DATA=payload, RAW=wire; aggregate job GiB):
const char* kHeaderNvl = ",nvl_tx_gib,nvl_rx_gib,nvl_tx_raw_gib,nvl_rx_raw_gib";

struct Phases {
    std::vector<float>  construct, payload_copy, build, probe, compact, gather,
                        partition, comm, join, total;
    std::vector<double> nvl_tx, nvl_rx, nvl_tx_raw, nvl_rx_raw; // per-iter job GiB (--nvlink-bytes)
};

void write_row(FILE* csv, const alps::Context& ctx, const Args& a, const char* strat,
               const Phases& ph, long long out_rows) {
    if (ctx.rank != 0 || !csv) return;
    const Stat c = summarize(ph.construct), b = summarize(ph.build), p = summarize(ph.probe),
               cm = summarize(ph.compact), pa = summarize(ph.partition), co = summarize(ph.comm),
               j = summarize(ph.join), t = summarize(ph.total);
    const long long total_left  = a.left_per_gpu  * ctx.size;
    const long long total_right = a.right_per_gpu * ctx.size;
    const double mrows = t.median > 0 ? (double)out_rows / (t.median / 1e3) / 1e6 : 0.0;
    std::fprintf(csv,
        "%d,%s,%s,%lld,%lld,%lld,%lld,%d,%.3f,%d,"
        "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
        "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%lld,%.1f",
        ctx.size, a.dtype.c_str(), strat, a.left_per_gpu, a.right_per_gpu, total_left, total_right,
        a.cg, a.lf, a.iters,
        c.mean, c.median, b.mean, b.median, p.mean, p.median, cm.mean, cm.median,
        pa.mean, pa.median, co.mean, co.median, j.mean, j.median, t.mean, t.median, out_rows, mrows);
    std::fprintf(csv, ",%.4f,%d", a.match_rate, a.green_sms);   // selectivity + green-ctx SM partition
    // wide columns (always present; 0 for narrow and for NCCL rows).
    const Stat pcm = summarize(ph.payload_copy), g = summarize(ph.gather);
    std::fprintf(csv, ",%s,%d,%d,%.4f,%.4f,%.4f,%.4f",
                 a.mode.c_str(), a.left_payloads, a.right_payloads,
                 pcm.mean, pcm.median, g.mean, g.median);
    double nx = median_d(ph.nvl_tx), nr = median_d(ph.nvl_rx),
           nxr = median_d(ph.nvl_tx_raw), nrr = median_d(ph.nvl_rx_raw);
    if (a.nvlink) std::fprintf(csv, ",%.3f,%.3f,%.3f,%.3f", nx, nr, nxr, nrr);
    std::fprintf(csv, "\n");
    std::fflush(csv);
    std::printf("  np=%d %-10s %-5s L=%lld R=%lld m=%.2f  total=%.3f ms (out=%lld, %.0f Mrows/s)",
                ctx.size, strat, a.dtype.c_str(), a.left_per_gpu, a.right_per_gpu, a.match_rate,
                t.median, out_rows, mrows);
    if (a.nvlink) std::printf("  NVLink GiB: DATA tx=%.2f rx=%.2f | RAW(wire) tx=%.2f rx=%.2f", nx, nr, nxr, nrr);
    std::printf("\n");
}

// Record the BOTTLENECK rank's phase breakdown for one iteration: MAXLOC on the
// headline total (v[headline]), then broadcast THAT rank's whole phase vector so
// every rank logs a consistent breakdown -- the per-phase values sum to the total
// because they come from one rank. (Independently MAX-reducing each phase would
// make the reported phases not sum to the reported total.)
void reduce_bottleneck(float* v, int n, int headline, int rank, MPI_Comm comm) {
    struct { float val; int rank; } in{v[headline], rank}, out{};
    ALPS_MPICHECK(MPI_Allreduce(&in, &out, 1, MPI_FLOAT_INT, MPI_MAXLOC, comm));
    ALPS_MPICHECK(MPI_Bcast(v, n, MPI_FLOAT, out.rank, comm));
}

long long sum_ll(long long local, MPI_Comm comm) {
    long long g = 0;
    ALPS_MPICHECK(MPI_Allreduce(&local, &g, 1, MPI_LONG_LONG, MPI_SUM, comm));
    return g;
}

// Output = [left payloads (n_left_pay), right_key, right payloads]. Every column k
// must equal right_key + offset(k): left payloads -> +k; right_key -> +0; right
// payloads -> +(k-n_left_pay-1). Sums violations across ALL columns. (Narrow:
// n_left_pay=1, 3 cols, all offsets 0 -> the original col0==col1 check, plus col2==col1.)
template <typename T>
unsigned long long value_mismatches(const gamax::table& out, int n_left_pay, cudaStream_t stream) {
    const long long n = out.num_rows();
    const int ncols = static_cast<int>(out.num_columns());
    if (n <= 0 || ncols < 2 || n_left_pay >= ncols) return 0;
    const T* ref = out.get_column(n_left_pay).data<T>();   // the right_key column
    gamax::device_buffer bad_storage(
        sizeof(unsigned long long), stream, gamax::buffer_init::zero);
    auto* d_bad = static_cast<unsigned long long*>(bad_storage.data());
    unsigned long long h_bad = 0;
    const int threads = 256;
    const int blocks  = (int)std::min<long long>((n + threads - 1) / threads, 4096);
    for (int k = 0; k < ncols; ++k) {
        if (k == n_left_pay) continue;   // this column IS `ref` (right_key) -> ref==ref, skip
        const long long offset = (k < n_left_pay) ? k : (k - n_left_pay - 1);
        count_neq_offset<T><<<blocks, threads, 0, stream>>>(
            out.get_column(k).data<T>(), ref, n, offset, d_bad);
    }
    ALPS_CUDACHECK(cudaMemcpyAsync(&h_bad, d_bad, sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost, stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return h_bad;
}

// Selectivity guard. With statistical misses (match_rate<1) out_rows won't exactly
// equal match_rate*total_right, so WARN only OUTSIDE a 2% relative band (absorbs the
// Binomial(R,m) noise + the ceil()-in-fk_draw_range bias). At match_rate=1 there is NO
// statistical noise -- every FK matches -- so we keep the ORIGINAL exact check
// (out_rows==total_right); the band would silently tolerate a dropped-rows bug there,
// and value_mismatches only inspects EMITTED rows (it cannot see missing matches).
bool out_rows_offband(long long out_rows, long long bad_rows, const Args& a, int world) {
    if (bad_rows != 0) return true;
    const long long total_right = a.right_per_gpu * (long long)world;
    if (a.match_rate >= 1.0) return out_rows != total_right;   // exact at 100% match
    const double expected = a.match_rate * static_cast<double>(total_right);
    return std::fabs(static_cast<double>(out_rows) - expected) > 0.02 * expected;
}

// ---- strategy runners --------------------------------------------------------
template <typename T>
void run_shared_map(const gamax::table& pk, const gamax::table& fk, const Args& a,
                    const alps::Context& ctx, unsigned long long budget, FILE* csv) {
    if (should_skip("shared_map", a, ctx.size, sizeof(T), budget)) {
        if (ctx.rank == 0) std::printf("  np=%d shared_map  L=%lld R=%lld  SKIP (est OOM)\n",
                                       ctx.size, a.left_per_gpu, a.right_per_gpu);
        return;
    }
    const bool wide = (a.mode == "wide");
    alps::containers::FabricStripedRegion region(ctx);
    // WIDE: P_L caller-owned reusable fabric regions for the left payload columns
    // (non-movable, so hold them by unique_ptr and pass a vector of raw pointers).
    std::vector<std::unique_ptr<alps::containers::FabricStripedRegion>> pay_regions;
    std::vector<alps::containers::FabricStripedRegion*> pay_ptrs;
    if (wide)
        for (int c = 0; c < a.left_payloads; ++c) {
            pay_regions.push_back(std::make_unique<alps::containers::FabricStripedRegion>(ctx));
            pay_ptrs.push_back(pay_regions.back().get());
        }
    alps::join::SharedJoinHint hint; hint.left_key_idx = 0; hint.right_key_idx = 0;
    hint.cg = a.cg; hint.lf = a.lf; hint.wide = wide;
    Phases ph;
    long long out_local = 0, bad_local = 0;
    for (int it = 0; it < a.warmup + a.iters; ++it) {
        ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
        NvlBytes n0 = a.nvlink ? nvml_read() : NvlBytes{};
        nvtxRangePushA("shared_map");          // NVTX: the whole join, for nsys
        alps::join::SharedJoinTiming tm{};
        auto out = alps::join::shared_map_join(pk.view(), fk.view(), region, ctx, ctx.stream,
                                               hint, &tm, pay_ptrs);
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
        nvtxRangePop();
        NvlBytes n1{}; if (a.nvlink) { ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm)); n1 = nvml_read(); }
        if (it == a.warmup + a.iters - 1) {
            out_local = out->num_rows();
            bad_local = (long long)value_mismatches<T>(*out, a.left_payloads, ctx.stream);
        }
        if (it >= a.warmup) {
            // headline total = payload_copy + build + probe + compact + gather (construct
            // excluded; payload_copy/gather are 0 in narrow -> same as before).
            float v[7] = {tm.construct_ms, tm.payload_copy_ms, tm.build_ms, tm.probe_ms,
                          tm.compact_ms, tm.gather_ms,
                          tm.payload_copy_ms + tm.build_ms + tm.probe_ms + tm.compact_ms + tm.gather_ms};
            reduce_bottleneck(v, 7, /*headline=*/6, ctx.rank, ctx.mpi_comm);
            ph.construct.push_back(v[0]); ph.payload_copy.push_back(v[1]); ph.build.push_back(v[2]);
            ph.probe.push_back(v[3]); ph.compact.push_back(v[4]); ph.gather.push_back(v[5]);
            ph.total.push_back(v[6]);
            if (a.nvlink) {
                ph.nvl_tx.push_back(reduce_gib(n1.tx - n0.tx, ctx.mpi_comm));
                ph.nvl_rx.push_back(reduce_gib(n1.rx - n0.rx, ctx.mpi_comm));
                ph.nvl_tx_raw.push_back(reduce_gib(n1.tx_raw - n0.tx_raw, ctx.mpi_comm));
                ph.nvl_rx_raw.push_back(reduce_gib(n1.rx_raw - n0.rx_raw, ctx.mpi_comm));
            }
        }
    }
    region.free();
    for (auto& r : pay_regions) r->free();   // collective teardown (all ranks, same order)
    const long long out_rows = sum_ll(out_local, ctx.mpi_comm);
    const long long bad_rows = sum_ll(bad_local, ctx.mpi_comm);
    const long long total_right = a.right_per_gpu * ctx.size;
    if (ctx.rank == 0 && out_rows_offband(out_rows, bad_rows, a, ctx.size))
        std::fprintf(stderr, "  [WARN] shared_map out_rows=%lld (expect ~%.0f +/-2%%, m=%.3f) value_mismatches=%lld\n",
                     out_rows, a.match_rate * (double)total_right, a.match_rate, bad_rows);
    write_row(csv, ctx, a, "shared_map", ph, out_rows);
}

template <typename T>
void run_nccl(const gamax::table& pk, const gamax::table& fk, const Args& a,
              alps::join::DistJoinHint::Strategy strat, const char* label,
              const alps::Context& ctx, unsigned long long budget, FILE* csv) {
    if (should_skip(label, a, ctx.size, sizeof(T), budget)) {
        if (ctx.rank == 0) std::printf("  np=%d %-10s L=%lld R=%lld  SKIP (est OOM)\n",
                                       ctx.size, label, a.left_per_gpu, a.right_per_gpu);
        return;
    }
    alps::join::DistJoinHint hint;
    hint.local.join_algo   = gamax::join::HASH;
    hint.local.mat_algo    = gamax::join::GFUR;
    hint.local.unique_left = true;          // PK build side
    hint.local.max_out     = 0;
    hint.left_key_idx = 0; hint.right_key_idx = 0;
    hint.strategy = strat;
    // Opt-in static_map local join (fair vs shared_map). "_mm" suffix in the CSV
    // label marks the old gamax multiset path so the two are distinguishable.
    hint.local_static_map = (a.local_join == "static_map");
    hint.local_map_cg = a.cg; hint.local_map_lf = a.lf;
    hint.wide = (a.mode == "wide");   // collectives already move all columns; wide routes
                                      // the local join to local_map_join's index+gather path.
    const std::string csv_label = std::string(label) + (a.local_join == "gamax" ? "_mm" : "");
    Phases ph;
    long long out_local = 0, bad_local = 0;
    for (int it = 0; it < a.warmup + a.iters; ++it) {
        ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
        NvlBytes n0 = a.nvlink ? nvml_read() : NvlBytes{};
        nvtxRangePushA(label);                 // NVTX: "shuffle" / "broadcast", for nsys
        alps::join::DistJoinTiming tm{};
        auto out = alps::join::inner_eq_join(pk.view(), fk.view(), hint, ctx, ctx.stream, &tm);
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
        nvtxRangePop();
        NvlBytes n1{}; if (a.nvlink) { ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm)); n1 = nvml_read(); }
        if (it == a.warmup + a.iters - 1) {
            out_local = out->num_rows();
            bad_local = (long long)value_mismatches<T>(*out, a.left_payloads, ctx.stream);
        }
        if (it >= a.warmup) {
            float v[4] = {tm.partition_ms, tm.comm_ms, tm.join_ms,
                          tm.partition_ms + tm.comm_ms + tm.join_ms};
            reduce_bottleneck(v, 4, /*headline=*/3, ctx.rank, ctx.mpi_comm);
            ph.partition.push_back(v[0]); ph.comm.push_back(v[1]); ph.join.push_back(v[2]);
            ph.total.push_back(v[3]);
            if (a.nvlink) {
                ph.nvl_tx.push_back(reduce_gib(n1.tx - n0.tx, ctx.mpi_comm));
                ph.nvl_rx.push_back(reduce_gib(n1.rx - n0.rx, ctx.mpi_comm));
                ph.nvl_tx_raw.push_back(reduce_gib(n1.tx_raw - n0.tx_raw, ctx.mpi_comm));
                ph.nvl_rx_raw.push_back(reduce_gib(n1.rx_raw - n0.rx_raw, ctx.mpi_comm));
            }
        }
    }
    const long long out_rows = sum_ll(out_local, ctx.mpi_comm);
    const long long bad_rows = sum_ll(bad_local, ctx.mpi_comm);
    const long long total_right = a.right_per_gpu * ctx.size;
    if (ctx.rank == 0 && out_rows_offband(out_rows, bad_rows, a, ctx.size))
        std::fprintf(stderr, "  [WARN] %s out_rows=%lld (expect ~%.0f +/-2%%, m=%.3f) value_mismatches=%lld\n",
                     label, out_rows, a.match_rate * (double)total_right, a.match_rate, bad_rows);
    write_row(csv, ctx, a, csv_label.c_str(), ph, out_rows);
}

template <typename T>
void run_all(const Args& a, const alps::Context& ctx, FILE* csv) {
    const long long total_pk = a.left_per_gpu * ctx.size;
    // Key-range guard: PK ids span [0, total_pk), but FK draws span the EXPANDED
    // range [0, ceil(total_pk/match_rate)) (the miss keys live in [total_pk, range)).
    // The largest FK id must fit T, else fill_fk wraps to NEGATIVE keys, violating the
    // shared_map precondition (keys >= 0; -1 is the empty sentinel) -> silently wrong
    // joins. Deterministic on every rank (args identical), so all skip the config
    // together. At match_rate=1 the bound is total_pk-1 (the original guard).
    const unsigned long long draw_range = fk_draw_range((unsigned long long)total_pk, a.match_rate);
    if (draw_range - 1ull > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
        if (ctx.rank == 0)
            std::printf("  np=%d L=%lld R=%lld m=%.3f  SKIP-CONFIG (expanded FK range %llu exceeds %s; use int64/higher m)\n",
                        ctx.size, a.left_per_gpu, a.right_per_gpu, a.match_rate, draw_range, a.dtype.c_str());
        return;
    }

    // Rank-invariant memory budget for the OOM pre-check: MIN device capacity
    // across ranks * 0.85 (so the should_skip decision is identical everywhere).
    size_t freeb = 0, totalb = 0; ALPS_CUDACHECK(cudaMemGetInfo(&freeb, &totalb));
    unsigned long long mn = totalb, budget = 0;
    ALPS_MPICHECK(MPI_Allreduce(&mn, &budget, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, ctx.mpi_comm));
    budget = (unsigned long long)(0.85 * (double)budget);

    // Wide mode: est_bytes_wide is POOL-resident only, and the pool = device - reserve
    // (the fabric slots/payload slabs + NCCL transport live in the reserve, out of pool).
    // So check the pool-resident estimate against (budget - reserve). Narrow's est already
    // includes the fabric and is checked against the full budget (small reserve).
    unsigned long long eff_budget = budget;
    if (a.mode == "wide") {
        unsigned long long reserve = 0;
        if (const char* r = std::getenv("GAMAX_MEM_RESERVE_BYTES")) reserve = (unsigned long long)arg_ll(r);
        eff_budget = budget > reserve ? budget - reserve : 0;
    }

    // Config-level pre-check (rank-invariant): if NO strategy fits, skip the WHOLE config
    // BEFORE generating tables -- in wide mode the input tables themselves (big FK payloads)
    // can exceed the pool, so generating them first would OOM at gen rather than skip
    // cleanly. If at least one strategy fits, its est includes the tables, so gen is safe.
    const bool any_fits = !should_skip("shared_map", a, ctx.size, sizeof(T), eff_budget)
                       || !should_skip("shuffle",    a, ctx.size, sizeof(T), eff_budget)
                       || !should_skip("broadcast",  a, ctx.size, sizeof(T), eff_budget);
    if (!any_fits) {
        if (ctx.rank == 0)
            std::printf("  np=%d L=%lld R=%lld  SKIP-CONFIG (no strategy fits est budget; tables not generated)\n",
                        ctx.size, a.left_per_gpu, a.right_per_gpu);
        return;
    }

    const long long base = (long long)ctx.rank * a.left_per_gpu;     // PK id base for this rank
    auto pk = build_table<T>(a.left_per_gpu,  /*is_pk=*/true,  base,     a.left_payloads,  a, ctx);
    auto fk = build_table<T>(a.right_per_gpu, /*is_pk=*/false, total_pk, a.right_payloads, a, ctx);

    const bool all = (a.strategy == "all");
    if (all || a.strategy == "shared_map") run_shared_map<T>(*pk, *fk, a, ctx, eff_budget, csv);
    if (all || a.strategy == "shuffle")
        run_nccl<T>(*pk, *fk, a, alps::join::DistJoinHint::Strategy::shuffle, "shuffle", ctx, eff_budget, csv);
    if (all || a.strategy == "broadcast")
        run_nccl<T>(*pk, *fk, a, alps::join::DistJoinHint::Strategy::broadcast, "broadcast", ctx, eff_budget, csv);
}

} // namespace

int main(int argc, char** argv) {
    // Defaults (overridable by a launcher -x): RMM arena + 64 NCCL CTAs.
    setenv("GAMAX_RMM_RESOURCE", "arena", /*overwrite=*/0);
    setenv("NCCL_MIN_CTAS", "64", /*overwrite=*/0);
    setenv("NCCL_MAX_CTAS", "64", /*overwrite=*/0);

    alps::configure_process();
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    try {
        Args args = parse(argc, argv);
        if (args.nvlink) nvml_setup(ctx.local_device_id); // each rank inits its local GPU handle

        int setup_fail = 0;

        FILE* csv = nullptr;
        if (ctx.rank == 0) {
            csv = std::fopen(args.out.c_str(), "a");
            if (!csv) { std::fprintf(stderr, "[compare_bench] cannot open --out: %s\n",
                                     args.out.c_str()); setup_fail = 1; }
            else { std::fseek(csv, 0, SEEK_END);
                   if (std::ftell(csv) == 0) {
                       std::fputs(kHeader, csv);
                       std::fputs(kHeaderSel, csv);           // match_rate (selectivity)
                       std::fputs(kHeaderWide, csv);          // mode + payload cols + wide timers
                       if (args.nvlink) std::fputs(kHeaderNvl, csv);
                       std::fputs("\n", csv);
                   } }
        }
        int global_fail = 0;
        ALPS_MPICHECK(MPI_Allreduce(&setup_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
        if (global_fail) { if (csv) std::fclose(csv); throw std::runtime_error("setup failed"); }

        // Optional: partition the device with a CUDA green context so the WHOLE join
        // (compute + NCCL) runs on ~green_sms SMs. Each rank partitions its own GPU
        // (no collective). Record the ACTUAL partition size (rounds up to alignment).
        GreenCtx green;
        if (args.green_sms > 0) {
            unsigned actual = green_setup(ctx, args.green_sms, green);
            args.green_sms = (int)actual;
            if (ctx.rank == 0)
                std::printf("[green-ctx] join runs on a %u-SM partition per GPU (all %d ranks)\n",
                            actual, ctx.size);
        }

        if (args.dtype == "int64") run_all<int64_t>(args, ctx, csv);
        else                       run_all<int32_t>(args, ctx, csv);

        green_teardown(ctx, green);
        if (csv) std::fclose(csv);
    } catch (const std::exception& ex) {
        if (ctx.rank == 0) std::fprintf(stderr, "[compare_bench] %s\n", ex.what());
        if (g_nvml_ok) nvmlShutdown();
        alps::destroy_context(ctx);
        MPI_Finalize();
        return 1;
    }
    if (g_nvml_ok) nvmlShutdown();
    alps::destroy_context(ctx);
    MPI_Finalize();
    return 0;
}
