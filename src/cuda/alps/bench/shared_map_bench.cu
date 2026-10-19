// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_shared_map_bench -- milestone-1 microbenchmark for the fabric-backed shared
// hash table. Measures aggregate insert and find throughput (Mops/s) of a
// cuco::static_map whose slots are block-striped across every GPU's memory
// (mode=shared) against the same work done in per-GPU private tables with no
// fabric (mode=local_device, the linear-scaling reference ceiling).
//
// Weak scaling: each rank contributes --keys-per-gpu globally-unique int32 keys,
// so the shared table grows with the GPU count and -- under the uniform hash --
// each rank's accesses are 1-1/world remote. The GPU-count sweep therefore
// doubles as a remote-fraction sweep. find runs at match_ratio 1.0 (each rank
// probes the keys it inserted), so every lookup is a full successful probe + a
// remote value read.
//
// Cold-table: each timed iteration constructs a fresh map (so insert always hits
// an empty table) over a VMM region that is created ONCE and reused. Timing is
// per phase: MPI_Barrier -> CUDA events -> MPI_Allreduce(MAX); aggregate
// throughput = world * keys_per_gpu / max_phase_time.
//
// Example:
//   mpirun -np 4 alps_shared_map_bench --keys-per-gpu 67108864 --lf 0.5 --cg 2 \
//          --iters 10 --warmup 3 --mode both --out shared_map.csv

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvml.h>
#include <nvtx3/nvToolsExt.h>   // header-only NVTX ranges: mark insert/find for ncu --lockstep-nvtx

#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda/std/tuple>

#include "types/data_type.hpp"
#include "memory/column.hpp"
#include "alps/context.hpp"
#include "alps/error.hpp"
#include "alps/containers/fabric_region.hpp"
#include "alps/containers/shared_static_map.cuh"

namespace {

// Empty sentinel: -1 works for both int32 and int64 keys/values (all real keys >= 0).
constexpr long long kEmptySentinel = -1;

struct Args {
    gamax::size_type keys_per_gpu = gamax::size_type{1} << 26; // 2^26
    double           lf           = 0.5;
    int              cg           = 2;
    int              iters        = 10;
    int              warmup       = 3;
    std::string      mode         = "both";  // shared | local_device | both
    std::string      dtype        = "int32"; // int32 | int64 (key AND value width)
    bool             nvlink       = false;   // measure per-op NVLink bytes via NVML
    std::string      out;                    // required CSV
};

// --- NVML: measured per-op NVLink traffic (unprivileged, per local GPU) --------------------
// NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_{TX,RX} are cumulative aggregate (all links) user-data KiB
// counters (same counters as `nvidia-smi nvlink -gt d`); read before/after a synced op and diff.
nvmlDevice_t g_nvmldev{};
bool         g_nvml_ok = false;

void nvml_setup(int cuda_dev) {
    if (nvmlInit_v2() != NVML_SUCCESS) { std::fprintf(stderr, "[nvml] init failed\n"); return; }
    char busid[64] = {0};
    if (cudaDeviceGetPCIBusId(busid, sizeof(busid), cuda_dev) != cudaSuccess) {
        std::fprintf(stderr, "[nvml] cudaDeviceGetPCIBusId failed\n"); return; }
    if (nvmlDeviceGetHandleByPciBusId_v2(busid, &g_nvmldev) != NVML_SUCCESS) {
        std::fprintf(stderr, "[nvml] handle for %s failed\n", busid); return; }
    g_nvml_ok = true;
}

// DATA_* = user-payload bytes; RAW_* = data + protocol overhead (the actual wire bytes). We read
// both so the analysis can separate payload from protocol and report true link (wire) utilization.
struct NvlBytes { unsigned long long tx_kib{0}, rx_kib{0}, tx_raw_kib{0}, rx_raw_kib{0}; };
NvlBytes nvml_read() {
    NvlBytes b;
    if (!g_nvml_ok) return b;
    nvmlFieldValue_t fv[4] = {};
    fv[0].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;  // user payload
    fv[1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
    fv[2].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX;   // payload + protocol = wire
    fv[3].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX;
    // scopeId = UINT_MAX => aggregate over ALL NVLinks. Default (0) returns only LINK 0, which is
    // ~aggregate/link_count (18 on this GB200, "NV18") when traffic is link-balanced -- verified by
    // exp_results/shared_map_ncu_20260626/nvml_perlink_probe.cu vs an ncu ground-truth calibration.
    for (auto& f : fv) f.scopeId = UINT_MAX;
    if (nvmlDeviceGetFieldValues(g_nvmldev, 4, fv) == NVML_SUCCESS) {
        if (fv[0].nvmlReturn == NVML_SUCCESS) b.tx_kib     = fv[0].value.ullVal;
        if (fv[1].nvmlReturn == NVML_SUCCESS) b.rx_kib     = fv[1].value.ullVal;
        if (fv[2].nvmlReturn == NVML_SUCCESS) b.tx_raw_kib = fv[2].value.ullVal;
        if (fv[3].nvmlReturn == NVML_SUCCESS) b.rx_raw_kib = fv[3].value.ullVal;
    }
    return b;
}
// Sum a per-rank KiB delta across all ranks; return aggregate job traffic in GiB.
double reduce_gib(unsigned long long local_kib, MPI_Comm comm) {
    unsigned long long g = 0;
    MPI_Allreduce(&local_kib, &g, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    return static_cast<double>(g) / (1024.0 * 1024.0);
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* f) -> const char* {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + f);
            return argv[++i];
        };
        if      (!std::strcmp(argv[i], "--keys-per-gpu")) a.keys_per_gpu = std::atoll(next(argv[i]));
        else if (!std::strcmp(argv[i], "--lf"))           a.lf           = std::atof(next(argv[i]));
        else if (!std::strcmp(argv[i], "--cg"))           a.cg           = std::atoi(next(argv[i]));
        else if (!std::strcmp(argv[i], "--iters"))        a.iters        = std::atoi(next(argv[i]));
        else if (!std::strcmp(argv[i], "--warmup"))       a.warmup       = std::atoi(next(argv[i]));
        else if (!std::strcmp(argv[i], "--mode"))         a.mode         = next(argv[i]);
        else if (!std::strcmp(argv[i], "--dtype"))        a.dtype        = next(argv[i]);
        else if (!std::strcmp(argv[i], "--nvlink-bytes"))  a.nvlink      = true;
        else if (!std::strcmp(argv[i], "--out"))          a.out          = next(argv[i]);
        else throw std::runtime_error(std::string("unknown arg: ") + argv[i]);
    }
    if (a.out.empty()) throw std::runtime_error("--out <csv file> is required");
    if (a.dtype != "int32" && a.dtype != "int64")
        throw std::runtime_error("--dtype must be int32 or int64");
    return a;
}

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

struct PhaseTimes {
    std::vector<float>  insert_ms, find_ms;
    // measured per-op NVLink traffic, aggregate across all ranks, GiB (when --nvlink-bytes).
    // *_tx/*_rx = DATA (user payload); *_tx_raw/*_rx_raw = RAW (payload + protocol = wire).
    std::vector<double> ins_tx, ins_rx, fnd_tx, fnd_rx;
    std::vector<double> ins_tx_raw, ins_rx_raw, fnd_tx_raw, fnd_rx_raw;
};

// Construct a fresh map each iteration (cold table), time insert then find,
// MAX-reduce each phase across ranks. `make_map` returns a freshly built map.
template <typename MakeMap, typename KeyT>
PhaseTimes run_loop(MakeMap&& make_map, const KeyT* keys_ptr, KeyT* found_ptr,
                    gamax::size_type kpg, const Args& a, const alps::Context& ctx) {
    PhaseTimes t;
    cudaEvent_t s, e;
    ALPS_CUDACHECK(cudaEventCreate(&s));
    ALPS_CUDACHECK(cudaEventCreate(&e));
    auto pairs = thrust::make_zip_iterator(cuda::std::make_tuple(keys_ptr, keys_ptr));

    for (int it = 0; it < a.warmup + a.iters; ++it) {
        // Cold construct (not timed). NOTE: cuco's ctor clears the WHOLE slot array
        // to the empty sentinel, so in shared mode every rank redundantly zeroes the
        // entire striped table (incl. peers' segments over the fabric) -- world x the
        // necessary init traffic. Correct (idempotent: all write the same sentinel)
        // and untimed, but inflates per-construct overhead at large world; cuco
        // exposes no partial-clear hook to avoid it.
        auto map = make_map();
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream)); // finish slot init

        // ---- insert ---- (stream is idle here from the slot-init sync above, so the
        // NVML counter delta brackets exactly this rank's insert kernel)
        ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
        NvlBytes ni0 = a.nvlink ? nvml_read() : NvlBytes{};
        nvtxRangePushA("insert");  // wraps the launch so ncu --lockstep-nvtx-include "insert/" matches it
        ALPS_CUDACHECK(cudaEventRecord(s, ctx.stream));
        map.insert_async(pairs, pairs + kpg, ctx.stream);
        ALPS_CUDACHECK(cudaEventRecord(e, ctx.stream));
        ALPS_CUDACHECK(cudaEventSynchronize(e));
        nvtxRangePop();
        float ins_ms = 0.0f;
        ALPS_CUDACHECK(cudaEventElapsedTime(&ins_ms, s, e));
        // barrier so the NVML window covers the whole phase (incl. serving peers'
        // requests that continue after this rank's own kernel ends) -> tx≈rx.
        NvlBytes ni1{};
        if (a.nvlink) { ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm)); ni1 = nvml_read(); }

        // ---- find (match_ratio 1.0: probe the keys we inserted) ----
        ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
        NvlBytes nf0 = a.nvlink ? nvml_read() : NvlBytes{};
        nvtxRangePushA("find");  // wraps the launch so ncu --lockstep-nvtx-include "find/" matches it
        ALPS_CUDACHECK(cudaEventRecord(s, ctx.stream));
        map.find_async(keys_ptr, keys_ptr + kpg, found_ptr, ctx.stream);
        ALPS_CUDACHECK(cudaEventRecord(e, ctx.stream));
        ALPS_CUDACHECK(cudaEventSynchronize(e));
        nvtxRangePop();
        float fnd_ms = 0.0f;
        ALPS_CUDACHECK(cudaEventElapsedTime(&fnd_ms, s, e));
        NvlBytes nf1{};
        if (a.nvlink) { ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm)); nf1 = nvml_read(); }

        // Untimed correctness spot-check on the final iteration: with match_ratio
        // 1.0 and value==key, every probe must return found[i]==key[i]. Without this
        // a silently-broken find (e.g. all-misses) would still report healthy
        // throughput. (found_ptr still holds this iteration's results.)
        if (it == a.warmup + a.iters - 1) {
            gamax::rmm_sync_exec_policy_handle policy{ctx.stream};
            const bool ok = thrust::equal(policy.get(),
                                          found_ptr, found_ptr + kpg, keys_ptr);
            if (!ok)
                std::fprintf(stderr,
                    "[rank %d] WARNING: find validation FAILED -- lookups did not return "
                    "value==key; throughput is not on a correctly-probed table\n", ctx.rank);
        }

        if (it >= a.warmup) {
            float gi = 0.0f, gf = 0.0f;
            ALPS_MPICHECK(MPI_Allreduce(&ins_ms, &gi, 1, MPI_FLOAT, MPI_MAX, ctx.mpi_comm));
            ALPS_MPICHECK(MPI_Allreduce(&fnd_ms, &gf, 1, MPI_FLOAT, MPI_MAX, ctx.mpi_comm));
            t.insert_ms.push_back(gi);
            t.find_ms.push_back(gf);
            if (a.nvlink) {  // sum each rank's NVLink delta -> total job traffic (GiB) for this op
                t.ins_tx.push_back(reduce_gib(ni1.tx_kib - ni0.tx_kib, ctx.mpi_comm));
                t.ins_rx.push_back(reduce_gib(ni1.rx_kib - ni0.rx_kib, ctx.mpi_comm));
                t.fnd_tx.push_back(reduce_gib(nf1.tx_kib - nf0.tx_kib, ctx.mpi_comm));
                t.fnd_rx.push_back(reduce_gib(nf1.rx_kib - nf0.rx_kib, ctx.mpi_comm));
                t.ins_tx_raw.push_back(reduce_gib(ni1.tx_raw_kib - ni0.tx_raw_kib, ctx.mpi_comm));
                t.ins_rx_raw.push_back(reduce_gib(ni1.rx_raw_kib - ni0.rx_raw_kib, ctx.mpi_comm));
                t.fnd_tx_raw.push_back(reduce_gib(nf1.tx_raw_kib - nf0.tx_raw_kib, ctx.mpi_comm));
                t.fnd_rx_raw.push_back(reduce_gib(nf1.rx_raw_kib - nf0.rx_raw_kib, ctx.mpi_comm));
            }
        }

        // Sync all ranks before the next iteration's clear so a slow rank's find
        // cannot race a fast rank's table clear over the fabric (warmup iterations
        // have no MPI_Allreduce above to serve as that barrier).
        ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
    }
    ALPS_CUDACHECK(cudaEventDestroy(s));
    ALPS_CUDACHECK(cudaEventDestroy(e));
    return t;
}

void write_row(FILE* csv, const alps::Context& ctx, const Args& a, const char* mode,
               const PhaseTimes& t) {
    if (ctx.rank != 0) return;
    const Stat is = summarize(t.insert_ms), fs = summarize(t.find_ms);
    const double total_keys = static_cast<double>(ctx.size) * static_cast<double>(a.keys_per_gpu);
    const double ins_mops = is.median > 0 ? total_keys / (is.median / 1e3) / 1e6 : 0.0;
    const double fnd_mops = fs.median > 0 ? total_keys / (fs.median / 1e3) / 1e6 : 0.0;
    std::fprintf(csv, "%d,%s,%lld,%.3f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f",
                 ctx.size, mode, static_cast<long long>(a.keys_per_gpu), a.lf, a.cg, a.iters,
                 is.mean, is.median, fs.mean, fs.median, ins_mops, fnd_mops);
    if (a.nvlink)
        std::fprintf(csv, ",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
                     median_d(t.ins_tx), median_d(t.ins_rx), median_d(t.fnd_tx), median_d(t.fnd_rx),
                     median_d(t.ins_tx_raw), median_d(t.ins_rx_raw),
                     median_d(t.fnd_tx_raw), median_d(t.fnd_rx_raw));
    std::fprintf(csv, "\n");
    std::fflush(csv);
    std::printf("  np=%d %-13s %s cg=%d ins=%.3f ms (%.0f Mops) find=%.3f ms (%.0f Mops)\n",
                ctx.size, mode, a.dtype.c_str(), a.cg, is.median, ins_mops, fs.median, fnd_mops);
    if (a.nvlink)
        std::printf("       NVLink GiB (job total): insert tx=%.2f rx=%.2f | find tx=%.2f rx=%.2f\n",
                    median_d(t.ins_tx), median_d(t.ins_rx), median_d(t.fnd_tx), median_d(t.fnd_rx));
}

template <int CG, typename KeyT>
void run_cg(const Args& a, const alps::Context& ctx, alps::containers::FabricStripedRegion& region,
            FILE* csv) {
    const gamax::size_type kpg   = a.keys_per_gpu;
    const long long        total = static_cast<long long>(ctx.size) * kpg;
    // Globally-unique keys span [0, total); the max key (total-1) must fit KeyT.
    if (static_cast<unsigned long long>(total) >
        static_cast<unsigned long long>(std::numeric_limits<KeyT>::max()))
        throw std::runtime_error("world*keys_per_gpu exceeds the key type's range; use --dtype int64");

    const KeyT empty = static_cast<KeyT>(kEmptySentinel);

    // This rank's globally-unique build keys: [rank*kpg, (rank+1)*kpg). The hash
    // scatters them across all segments regardless of their sequential values.
    auto keys = gamax::make_column<KeyT>(kpg, ctx.stream);
    {
        gamax::rmm_exec_policy_handle policy{ctx.stream};
        thrust::sequence(
            policy.get(), keys->template data<KeyT>(), keys->template data<KeyT>() + kpg,
            static_cast<KeyT>(ctx.rank * kpg));
    }
    const KeyT* keys_ptr = keys->template data<KeyT>();

    auto found = gamax::make_column<KeyT>(kpg, ctx.stream);
    KeyT* found_ptr = found->template data<KeyT>();
    {
        gamax::rmm_exec_policy_handle policy{ctx.stream};
        thrust::fill(policy.get(), found_ptr, found_ptr + kpg, empty);
    }

    const bool do_shared = (a.mode == "shared" || a.mode == "both");
    const bool do_local  = (a.mode == "local_device" || a.mode == "both");

    if (do_shared) {
        alps::containers::FabricAllocator<cuco::pair<KeyT, KeyT>> alloc(&region);
        auto make_shared = [&] {
            return alps::containers::SharedStaticMap<CG, KeyT, KeyT>(
                static_cast<std::size_t>(total), a.lf,                 // GLOBAL capacity
                cuco::empty_key{empty}, cuco::empty_value{empty},
                {}, {}, {}, {}, alloc, ctx.stream);
        };
        auto t = run_loop(make_shared, keys_ptr, found_ptr, kpg, a, ctx);
        write_row(csv, ctx, a, "shared", t);
    }

    if (do_local) {
        using Pair = cuco::pair<KeyT, KeyT>;
        using Allocator = gamax::rmm_cuco_allocator<Pair>;
        using LocalMap = cuco::static_map<
            KeyT, KeyT, cuco::extent<std::size_t>, cuda::thread_scope_device,
            cuda::std::equal_to<KeyT>,
            cuco::linear_probing<CG, cuco::default_hash_function<KeyT>>,
            Allocator, cuco::storage<1>>;
        gamax::rmm_allocator_handle<Pair> allocator{ctx.stream};
        auto make_local = [&] {
            return LocalMap(
                static_cast<std::size_t>(kpg), a.lf,                   // PRIVATE per-GPU capacity
                cuco::empty_key{empty}, cuco::empty_value{empty},
                {}, {}, {}, {}, allocator.get(), ctx.stream);
        };
        auto t = run_loop(make_local, keys_ptr, found_ptr, kpg, a, ctx);
        write_row(csv, ctx, a, "local_device", t);
    }
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process();
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    try {
        Args args = parse(argc, argv);

        FILE* csv = nullptr;
        int   setup_fail = 0;
        if (ctx.rank == 0) {
            csv = std::fopen(args.out.c_str(), "a");
            if (!csv) {
                std::fprintf(stderr, "[alps_shared_map_bench] cannot open --out file: %s\n",
                             args.out.c_str());
                setup_fail = 1;
            } else {
                std::fseek(csv, 0, SEEK_END);
                if (std::ftell(csv) == 0) {
                    std::fprintf(csv,
                        "np,mode,keys_per_gpu,load_factor,cg,iters,"
                        "insert_ms_mean,insert_ms_median,find_ms_mean,find_ms_median,"
                        "insert_mops,find_mops");
                    if (args.nvlink)
                        std::fprintf(csv, ",insert_nvl_tx_gib,insert_nvl_rx_gib,"
                                          "find_nvl_tx_gib,find_nvl_rx_gib,"
                                          "insert_nvl_tx_raw_gib,insert_nvl_rx_raw_gib,"
                                          "find_nvl_tx_raw_gib,find_nvl_rx_raw_gib");
                    std::fprintf(csv, "\n");
                }
            }
        }
        if (args.nvlink) nvml_setup(ctx.local_device_id);  // each rank inits its local GPU handle
        // Collective gate: a rank-0-only fopen failure must NOT let the other ranks
        // proceed into the collective region build (they would deadlock waiting on
        // rank 0). All ranks agree on the verdict and bail together.
        int global_fail = 0;
        ALPS_MPICHECK(MPI_Allreduce(&setup_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
        if (global_fail) {
            if (csv) std::fclose(csv);
            throw std::runtime_error("setup failed (could not open --out on rank 0)");
        }

        alps::containers::FabricStripedRegion region(ctx);
        const bool i64 = (args.dtype == "int64");
        switch (args.cg) {
            case 1: i64 ? run_cg<1, int64_t>(args, ctx, region, csv)
                        : run_cg<1, int32_t>(args, ctx, region, csv); break;
            case 2: i64 ? run_cg<2, int64_t>(args, ctx, region, csv)
                        : run_cg<2, int32_t>(args, ctx, region, csv); break;
            case 4: i64 ? run_cg<4, int64_t>(args, ctx, region, csv)
                        : run_cg<4, int32_t>(args, ctx, region, csv); break;
            default: throw std::runtime_error("--cg must be 1, 2, or 4");
        }
        region.free();

        if (csv) std::fclose(csv);
    } catch (const std::exception& ex) {
        if (ctx.rank == 0) std::fprintf(stderr, "[alps_shared_map_bench] %s\n", ex.what());
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
