// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once
// -----------------------------------------------------------------------------
// green_ctx.hpp -- reusable CUDA green-context SM-partition helper for NVBench.
//
// Creates a green context spanning ~N SMs and a stream on it; kernels launched on
// that stream run on exactly the N-SM partition (rounded UP to the device SM
// alignment, e.g. want 66 -> get 72 on GB200's align=8). Driver API only: CUDA 13
// has no runtime green-ctx API. Adapted from the reusable green_setup/green_teardown
// in src/cuda/alps/bench/dist_join_compare_bench.cu (which swaps alps::Context.stream);
// here it is Context-free so plain NVBench benchmarks can route their timed stream
// onto a partition.
//
// Usage in a benchmark:
//   GreenCtx g; cudaStream_t s = green_bench_stream(state, g);   // reads "green_sms" axis
//   cudaDeviceSynchronize();                                     // finish untimed setup first
//   state.set_cuda_stream(nvbench::make_cuda_stream_view(s));
//   ... state.exec(...) ...
//   green_teardown(g);
//
// green_sms <= 0  -> native full device (stream 0), no green ctx created.
//
// The core (GreenCtx / green_setup / green_teardown) is NVBench-free so it can
// also be reused by non-benchmark callers (e.g. run_tpch_ir's --green-sms
// correctness gate). Only the NVBench glue `green_bench_stream` needs NVBench;
// it is compiled solely where <nvbench/nvbench.cuh> is on the include path.
// -----------------------------------------------------------------------------
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define GCTX_DRVCHECK(x)                                                                    \
    do {                                                                                    \
        CUresult r_ = (x);                                                                  \
        if (r_ != CUDA_SUCCESS) {                                                           \
            const char* s_ = nullptr;                                                       \
            cuGetErrorString(r_, &s_);                                                      \
            throw std::runtime_error(std::string("CUDA driver error: ") + #x + " -> " +     \
                                     (s_ ? s_ : "?"));                                       \
        }                                                                                   \
    } while (0)

struct GreenCtx {
    CUgreenCtx gctx{};
    CUstream gstream{};
    bool active{false};
    unsigned sms{0};

    GreenCtx() = default;
    // Owns a green ctx + its stream. Non-copyable so the RAII destructor
    // can't double-free; the destructor tears down on ALL exit paths
    // (incl. exceptions) — an explicit green_teardown() stays valid and
    // idempotent (it clears `active`).
    GreenCtx(const GreenCtx&) = delete;
    GreenCtx& operator=(const GreenCtx&) = delete;
    ~GreenCtx();
};

// Build a green ctx + non-blocking stream from one SM resource group and record
// the actual SM count into g. Shared by green_setup (single partition) and
// green_setup_pair (disjoint pair) so the ctx-creation sequence lives in one place.
inline void green_make_ctx(CUdevice dev, CUdevResource& res, GreenCtx& g) {
    CUdevResourceDesc desc;
    GCTX_DRVCHECK(cuDevResourceGenerateDesc(&desc, &res, 1));
    GCTX_DRVCHECK(cuGreenCtxCreate(&g.gctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUdevResource gres{};
    GCTX_DRVCHECK(cuGreenCtxGetDevResource(g.gctx, &gres, CU_DEV_RESOURCE_TYPE_SM));
    GCTX_DRVCHECK(cuGreenCtxStreamCreate(&g.gstream, g.gctx, CU_STREAM_NON_BLOCKING, 0));
    g.sms = gres.sm.smCount;
    g.active = true;
}

// Create a green ctx with ~n_sms SMs on `device`; fills g and returns the ACTUAL
// (alignment-rounded) SM count of the partition. n_sms is clamped to the device total.
inline unsigned green_setup(int device, int n_sms, GreenCtx& g) {
    GCTX_DRVCHECK(cuInit(0));
    CUdevice dev;
    GCTX_DRVCHECK(cuDeviceGet(&dev, device));
    CUdevResource total{};
    GCTX_DRVCHECK(cuDeviceGetDevResource(dev, &total, CU_DEV_RESOURCE_TYPE_SM));
    if ((unsigned)n_sms > total.sm.smCount) n_sms = (int)total.sm.smCount;  // clamp to device
    CUdevResource res[64] = {};
    CUdevResource remain{};
    unsigned nb = 64;
    GCTX_DRVCHECK(cuDevSmResourceSplitByCount(res, &nb, &total, &remain, 0u, (unsigned)n_sms));
    if (nb == 0) throw std::runtime_error("green-ctx split produced 0 groups");
    green_make_ctx(dev, res[0], g);
    return g.sms;
}

// Carve TWO DISJOINT SM partitions from one device in a single split so they
// truly isolate (calling green_setup twice would carve each from the FULL
// device -> the two partitions OVERLAP). Partition A gets ~n_a SMs (rounded UP
// to the device SM alignment); partition B gets the ENTIRE remainder, so the
// two tile the whole device with zero idle SMs (full-device tiling). n_a is
// clamped to [1, total-1] so both sides are non-empty; the actual
// (alignment-rounded) sizes land in ga.sms / gb.sms. Used for the two-query
// partitioned-concurrency benchmark; for a single partition use green_setup.
//
// NOTE: `remaining` from cuDevSmResourceSplitByCount is used directly to build
// B's descriptor (no re-split), which is what tiles all SMs. If a given device
// rejects a green ctx built from an unaligned remainder, request an aligned n_a
// whose complement is also aligned (e.g. multiples of the alignment) so the
// remainder is aligned too.
inline void green_setup_pair(int device, int n_a, GreenCtx& ga, GreenCtx& gb) {
    GCTX_DRVCHECK(cuInit(0));
    CUdevice dev;
    GCTX_DRVCHECK(cuDeviceGet(&dev, device));
    CUdevResource total{};
    GCTX_DRVCHECK(cuDeviceGetDevResource(dev, &total, CU_DEV_RESOURCE_TYPE_SM));
    unsigned tot = total.sm.smCount;
    if (n_a < 1) n_a = 1;
    if ((unsigned)n_a >= tot) n_a = (int)tot - 1;  // leave >=1 SM for B

    // Capacity 1 => exactly ONE group of ~n_a SMs for A; `remain` holds
    // everything else (NOT the greedy multi-group leftover green_setup gets).
    CUdevResource gA{};
    CUdevResource remain{};
    unsigned nb = 1;
    GCTX_DRVCHECK(cuDevSmResourceSplitByCount(&gA, &nb, &total, &remain, 0u, (unsigned)n_a));
    if (nb == 0) throw std::runtime_error("green_setup_pair: split produced 0 groups for A");
    if (remain.sm.smCount == 0)
        throw std::runtime_error("green_setup_pair: no SMs left for B (n_a too large)");

    green_make_ctx(dev, gA, ga);      // partition A: the ~n_a group
    green_make_ctx(dev, remain, gb);  // partition B: the entire remainder
}

inline void green_teardown(GreenCtx& g) {
    if (!g.active) return;
    cudaStreamSynchronize((cudaStream_t)g.gstream);
    cuStreamDestroy(g.gstream);
    cuGreenCtxDestroy(g.gctx);
    g.active = false;
}

// RAII teardown (defined after green_teardown so it can call it). Fires on every
// scope exit — normal return AND exception unwind — so a green ctx created before
// a throwing op (e.g. execute()/verify in run_tpch_ir) is never leaked.
inline GreenCtx::~GreenCtx() { green_teardown(*this); }

// ----------------------------------------------------------------------------
// NVBench glue. Only defined when NVBench headers are reachable (the
// benchmark_*_green targets). Non-benchmark callers include green_ctx.hpp for
// green_setup/green_teardown above and never pull in NVBench.
// ----------------------------------------------------------------------------
#if defined(__has_include)
#  if __has_include(<nvbench/nvbench.cuh>)
#    define GCTX_HAVE_NVBENCH 1
#  endif
#endif

#ifdef GCTX_HAVE_NVBENCH
#include <nvbench/nvbench.cuh>

// Reads the int64 "green_sms" axis from `state`, sets up a green ctx if > 0, and
// records the actual partition size as an "actual_sms" summary column (0 = native).
// Returns the cudaStream_t the benchmark should hand to NVBench.
inline cudaStream_t green_bench_stream(nvbench::state& state, GreenCtx& g) {
    int64_t green_sms = state.get_int64_or_default("green_sms", 0);
    int64_t actual = 0;
    cudaStream_t stream = 0;  // native: default stream (full device)
    if (green_sms > 0) {
        // Use the device NVBench is currently running this benchmark on, NOT a
        // hardcoded 0 -- otherwise the green ctx (and its stream) live on the wrong
        // device and every op, incl. NVBench's own L2 flush, fails cudaErrorInvalidValue.
        int dev = 0;
        cudaGetDevice(&dev);
        actual = static_cast<int64_t>(green_setup(dev, static_cast<int>(green_sms), g));
        stream = reinterpret_cast<cudaStream_t>(g.gstream);
    }
    nvbench::summary& s = state.add_summary("actual_sms");
    s.set_string("name", "actual_sms");
    s.set_string("description", "actual green-ctx SM partition size (0 = native full device)");
    s.set_int64("value", actual);
    return stream;
}

#endif  // GCTX_HAVE_NVBENCH
