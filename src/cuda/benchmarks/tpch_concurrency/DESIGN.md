<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# TPC-H Two-Query Concurrency Benchmark — Design

**Status:** design agreed 2026-07-14 (grill-me interview). **Platform:** single NVIDIA GH200 480GB.
**Question:** when two TPC-H queries run concurrently on one GPU, what is the speedup/slowdown vs
running them sequentially, and how much does each individual query slow down?

This is the TPC-H analog of the operation-concurrency study in
`src/cuda/benchmarks/concurrency/` (archived at `exp_results/eiger_concurrency_20260713/`). It
deliberately reuses that study's metric definitions so results are cross-comparable.

## 1. Why a new harness (not `benchmark_concurrency.cu`)

The op-study's correctness rests on the `PreparedOperation` invariant: an operation enqueues its
whole GPU program **without any host-side sync**, so a mapped-memory gate kernel can hold both
streams and release them for a truly simultaneous GPU start. TPC-H query execution reads
data-dependent sizes back to the host mid-flight (`exec::execute` syncs at `executor.cpp:439,506`
plus per-op sizing syncs), so a whole query **cannot** be pre-enqueued behind a gate. Therefore:

- **One host thread per query, each driving its own stream.** Start alignment happens on the host.
- This is a standalone binary; `PreparedOperation`/gate machinery is not reused.

## 2. Execution path

Drive the **IR/executor path** exactly like `run_tpch_ir.cpp`:
`build_qN(qb)` → `apply_rules` → `apply_query_hints` → `lower_to_exec_plan` →
`exec::execute(ep, src_fn, ctx)` with `ctx.stream` per thread. The hand-written `get_query_runner`
path only wires Q1/Q6/Q14, so it is not usable for a broad pair study. All 22 queries are
verify-correct and stream-clean on regular streams (the green-ctx failures are CCCL/CUB library
issues under a *partitioned* context, **not** stream-0 leaks) — so we use full-device shared
regular non-blocking streams and **never** touch green contexts here.

Each worker thread owns its own `catalog`, `QueryBuilder`, `TpchTableReader`, `IrSourceResolver`,
and stream (the resolver cache is a plain non-thread-safe `unordered_map`). Only the process-wide,
thread-aware RMM memory resource is shared.

## 3. Timing model

- **Primary instrument: host wall-clock** (`steady_clock`). A TPC-H query is GPU work interleaved
  with host syncs + launch latency; end-to-end latency (including host stalls that are themselves a
  contention channel) is the quantity of interest. CUDA events would under-count exactly the
  interference we study.
- **Secondary: CUDA-event bracket** per stream (start/stop around the whole query) recorded
  alongside, to later decompose GPU-time vs wall-clock slowdown.
- **Start alignment:** two workers rendezvous on a host barrier immediately before `execute()`; one
  worker stamps a common `t0` on release. `C_A = end_A − t0`, `C_B = end_B − t0`,
  `M = max(end_A,end_B) − t0`. `start_skew` = how far apart the two workers cleared the barrier.
- **NVTX everywhere:** per-invocation top range (`iso/conc/batch|qa|qb`), per-query range (`Q<n>`),
  per-iteration sub-ranges in batch mode. So nsys timelines read cleanly and overlap is obvious.

## 4. Timed-region boundary — pre-warm, execution-only

Sources are **pre-warmed out of band**; timing wraps only `execute()` + final
`cudaStreamSynchronize`. The `IrSourceResolver` caches loaded `gamax::table`s resident on GPU for
its lifetime, so the untimed warmup runs (§7) leave all source columns resident; measured runs then
capture **GPU execution + intermediate-allocation contention**, not Parquet/H2D I/O contention.
(Cold-cache / load-included is a legitimate *different* study, deferred.)

## 5. Modes (one mode per binary invocation, never mixed)

| Invocation | Threads | Emits |
|---|---|---|
| `--isolated <Q>` | 1 | `T_Q` (single-shot latency) |
| `--concurrent <Qa> <Qb>` | 2 | `C_a`, `C_b`, `M`, `start_skew` |
| `--batch-isolated <Q> --iters K` | 1 | `B_Q` (K-iter batch total) + per-iter times |
| `--batch-concurrent <Qa> <Qb> --iters K --submission-policy dedicated\|alternating` | 2 | `BC_a`, `BC_b`, `BM` + per-iter times |
| `--profile` (modifier) | — | small fixed reps (1 warmup + 2 measured) for nsys, NVTX-rich |

- **Isolated latencies measured once per query (22 runs) and reused** across every pair.
- **Batch mode** = sustained-throughput contention. Each thread loops its query K times
  (default 10) back-to-back: retire result each iter, **no inter-iter sync or L2 purge** (realistic
  pressure), purge once before the batch. Measurement unit = batch-total wall-time per thread.
- **Two submission policies** do identical total work (`K`·A + `K`·B executions), so they share one
  sequential baseline and are directly comparable:
  - `dedicated`: `thread0: A×K`, `thread1: B×K`. Pairwise A-vs-B contention; lopsided pairs
    desync (long query's slowdown diluted).
  - `alternating`: `thread0: ABAB…`, `thread1: BABA…` anti-phased. Both threads process the same
    A/B mix → finish together regardless of length ratio → sustained realistic mixed contention.
  - **Per-iteration timestamping** (CUDA event + host stamp per iter, tagged by query) recovers
    per-query slowdown in *both* policies; the isolated per-iter A time (from `--batch-isolated A`)
    is the shared denominator.

## 6. Metrics (derived in Python post-processing, not in the binary)

Single-shot (reusing op-study definitions):
- `H_ideal = M / max(T_A,T_B)` — 1 = perfect overlap; →2 = full serialization.
- `H_sequential = M / (T_A + T_B)`; **speedup vs sequential = (T_A+T_B)/M = 1/H_sequential**.
- `S_A = C_A/T_A`, `S_B = C_B/T_B` — per-query slowdown.

Batch analogs: `speedup^batch = (B_A+B_B)/BM`; `S_A^batch = mean(conc A-iter)/mean(iso A-iter)`.

No single concurrent process knows `T_Q`, so the C++ binary emits only what it measured; Python
joins isolated↔concurrent CSVs (nearest-in-time `T_Q`, see §9) to compute all derived metrics.

## 7. Sampling & hygiene

- **Warmups: 3, untimed** — double as source pre-warm + arena growth; the first result of each
  mode is validated here (§8).
- **Adaptive counts:** single-shot **min-10/max-30** reps, stop when bootstrap 95% CIs on the
  headline ratios reach ≤2–3% relative half-width. Batch **min-5/max-15** batches (each already
  averages K iters). Report per-mode `Distribution` (median/mean/stddev/p5/p95) + `stable` flag.
- **Per timed sample:** drop previous result + retire intermediates (frees arena to the
  resident-sources baseline), then **L2 purge** (reuse `L2Purger`). Re-execution is just another
  `execute()` — sources hit the resolver cache; intermediates+result alloc fresh.
- **Concurrent single-shot is single-shot per query** (each thread runs its query once); for a
  lopsided pair the long query runs partly alone → its `S` is naturally diluted (correct
  "launch both once" semantics; batch/alternating addresses sustained contention).
- **Alternate query→stream assignment across reps** in concurrent mode to cancel stream-arbitration
  bias (the op-study found stream B deterministically won post-gate).

## 8. Validation

Once per query per mode, in warmup, **outside timing**, via
`gamax::tpch::verify_result(view, /storage/tpch/sf10/golden/qNN.parquet)`. Concurrent path
validates **both** results — this is the run that could expose a concurrency-induced race. On
mismatch: abort the cell, emit `failed_verify`, no timing rows. Not validated per sample (adds
host syncs that perturb the measured contention).

## 9. Clock-drift mitigation (no clock pinning available)

`bowwu` is non-root; `nvidia-smi -lgc/-rgc` are **denied**, and nvbench's programmatic path
(`device_info::lock_gpu_clocks` → `nvmlDeviceSetGpuLockedClocks`) needs the same root privilege, so
pinning is impossible without escalation. Because isolated (`T_Q`) and concurrent (`C_Q`) are now
measured in separate processes minutes apart, GPU clock drift would bias `S = C/T` directly.
Mitigation:
- **Record live SM clock + throttle reasons per run** via NVML
  (`nvmlDeviceGetClockInfo(NVML_CLOCK_SM)`, `nvmlDeviceGetCurrentClocksThrottleReasons`) — not the
  static `cudaDevAttrClockRate` max. Post-processing flags/derates cells measured under throttle.
- **Periodic isolated refresh:** driver re-runs the 22 isolated baselines every ~50 concurrent
  cells (~5 rounds, tagged `refresh_round` + `wall_timestamp`); Python matches each concurrent cell
  to the **nearest-in-time** `T_Q`. Converts drift from silent bias into measured/corrected signal.

## 10. Pair enumeration & scale

- **Full symmetric 22×22 matrix = 253 unordered pairs incl. self-pairs (diagonal)**, at **SF=10**
  (`/storage/tpch/sf10`, all 7 parquet + all 22 goldens present). NOTE: the GH200 has ~96GB **HBM**
  (the "480GB" is coherent system memory), so the RMM arena is ~75% of ~96GB ≈ **72GB**, shared by
  both queries — a worst-case memory probe (heaviest pair) is required before the full sweep, and a
  few heavy SF10 pairs may OOM (handled per-cell by subprocess isolation, recorded as `failed_oom`).
  SF10 is still the regime where a big GH200 underutilized by one query shows real overlap benefit
  from a co-runner. Secondary SF sweep on a subset is deferred.
- Each unordered pair measured once; both `S_A`,`S_B` captured, filling the full asymmetric
  slowdown matrix `S(i,j)`.

## 11. Process isolation & driver

**Subprocess-per-run** (single-invocation binary + resumable driver), mirroring
`script/run_tpch_axis_probe.sh` — because TPC-H queries can hard-corrupt the CUDA context and a
two-query pair doubles the crash/OOM surface. Fault isolation (a bad cell loses only itself), clean
per-cell context+arena (no cross-pair fragmentation bias), simpler binary.

Driver order: (0) write one metadata JSON via `collect_metadata`; (1) 22 isolated single-shot;
(2) 253 concurrent single-shot over `i≤j` with isolated refresh every ~50 cells; (3) batch phase on
selected pairs (§13). Resumable: skip any (run_type,policy,qa,qb) whose per-run file exists. A
poisoned/crashed cell leaves no file → re-runs.

## 12. Data model

Per-subprocess files aggregated by the driver (robust vs mid-write crash):
- **raw** (measured-only, no warmup rows): `run_type, policy, qa, qb, sf, rep, iter, a_wall_ms,
  b_wall_ms, a_gpu_ms, b_gpu_ms, makespan_ms, start_skew_ms, sm_clock_mhz, throttle_reasons,
  wall_timestamp`.
- **summary** (one row per cell/mode): keys + `n, stable` + `Distribution` for each measured
  quantity only (concurrent: `C_a,C_b,M,skew`; isolated: `T_Q` with `refresh_round`+timestamp).
  **No derived metrics** (Python computes them).
- **one metadata JSON per sweep**: git commit/dirty, GPU/driver/runtime, nvidia-smi snapshot, SF,
  K, sample params, seed, and the no-pin/record-clock caveat.

Reuse the *pure* helpers from the concurrency `reporting` TU — `collect_metadata`, `Distribution`
percentiles, bootstrap CI — factored into a small shared header.

## 13. Deliverables (auto-experiment archive)

`exp_results/tpch_concurrency_<date>/` in the auto-experiment convention: README (platform, date,
exact reproduce commands, methodology, no-pin caveat), aggregated raw+summary CSVs, metadata JSON,
**standalone** summarize + plot Python (re-derive from raw, no repo deps), figures PNG+PDF, platform
capture, and a few nsys `.nsys-rep` captures of the most interesting cells.

Figures:
- **Symmetric heatmaps** (Goal 1): `H_ideal(i,j)`, `speedup(i,j)=(T_i+T_j)/M`. Diagonal = self-pairs.
- **Asymmetric 22×22 heatmap** (Goal 2): `S(i,j) = C_i(with j)/T_i` — a row shows how badly `Qi`
  suffers across all co-runners.
- **Supporting:** `T_Q` isolated-latency bars; `start_skew` + clock-stability diagnostic panel
  (flag throttled / high-skew cells).
- **Batch (selected pairs):** dedicated-vs-alternating `speedup^batch` grouped bars + per-query
  `S^batch`.

**Interesting-pair selection for batch** (chosen against single-shot data): (a) strongest overlap
(lowest `H_ideal`), (b) worst mutual slowdown, (c) lopsided pairs (dedicated-vs-alternating
contrast), (d) self-pairs, (e) scan/join/agg shape spread — ~8–10 queries → 36–55 pairs.

## 14. Code layout

- `src/cuda/benchmarks/tpch_concurrency/benchmark_tpch_concurrency.cu` + `CMakeLists.txt`
  (link `exec`, `ir`, `tpch` reader/`ir_source_resolver`/`verify`, `memory`, `phys`, `CUDA::nvml`,
  NVTX; + shared reporting helpers).
- `script/run_tpch_concurrency.sh` — resumable driver.
- Standalone `summarize`/`plot` Python authored into the exp_results archive.

## 15. Accepted caveats

- **Shared arena:** allocator-lock interference is part of the measured slowdown — realistic for two
  queries in one process; NVTX + recorded clocks allow attribution if a cell looks anomalous.
- **No clock pinning:** mitigated by record-clock + periodic refresh (§9), not eliminated.
- **Temporal split** of isolated vs concurrent is the price of the clean one-mode-per-invocation
  model; the refresh scheme bounds it.
