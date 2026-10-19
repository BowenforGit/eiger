<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# TPC-H Two-Query Concurrency — Green-Context SM-Partitioning Extension — Design

**Status:** design agreed 2026-07-15 (grill-me interview). **Platform:** single NVIDIA GH200 480GB
(**132 SMs**, ~96 GB HBM → ~72 GB usable RMM arena; SM alignment queried at runtime — Blackwell is 8,
Hopper TBD). Extends the base study in [`DESIGN.md`](DESIGN.md); reuses its harness, timing model,
sampling, and hygiene verbatim except where noted.

**Question (Goal 3):** the base study showed that at SF10 most TPC-H queries nearly saturate the 132
SMs, so *time-shared* concurrency slows each ~1.5–2×, and on the **aggregate-slowdown** metric
(Σ response/isolated) **serial-shortest-first beat time-sharing for 98% of pairs**. Green-context SM
partitioning is the natural next lever: give each query a *disjoint* SM slice so they contend
spatially instead of time-sharing. This extension asks:

> **Can spatial (green-ctx) SM partitioning make concurrency a win where time-sharing couldn't — i.e.
> for lopsided pairs, does a dedicated SM slice per query beat (a) full-device time-shared concurrency
> and (b) serial-shortest-first, on makespan AND aggregate slowdown?**

with an **enabling sub-study**: the per-query **SM-scaling curve `T_Q(N)`** (isolated latency vs SM
budget), which (i) supplies the fair partition baseline and (ii) locates each query's saturation
knee — the thing that *predicts* whether partitioning can help a given pair.

The tension that defines the interesting regime: a small under-utilizing query gains dedicated SMs and
stops waiting (slowdown → 1), but the co-runner loses those SMs. For pairs where **both** saturate,
partitioning strictly loses (each gets fewer SMs, both slow, nothing gained); the payoff is expected
only for **lopsided / under-utilizing** pairs.

---

## 1. Baselines & slowdown definitions (three isolated latencies)

For `S = C/T`, three distinct isolated latencies are in play, each with a role:

1. **`T_Q^full,native`** — query alone on all 132 SMs, native stream-0, no green ctx. **Headline
   denominator.** It is the *only* baseline under which partitioned-concurrent, time-shared-concurrent,
   and serial-shortest-first are all comparable (all reference "query alone on the whole device"), so
   the vs-serial / vs-time-shared comparisons **must** use it. → **`S_full = C / T_Q^full,native`**.
2. **`T_Q(N)`** — the **SM-scaling curve**; query alone on an N-SM green partition. The value at a
   split's actual `N_a` is the fair partition baseline → **`S_part = C / T_Q(N_a)`**, the *mechanism
   diagnostic*: `S_part ≈ 1` ⇒ partitions truly isolated; `S_part > 1` ⇒ the slices still fight (shared
   HBM/L2/memory-controller — memory is *not* partitioned, only SM scheduling is).
3. **`T_Q^132,green`** — top of the scaling curve; vs `T_Q^full,native` it isolates *pure green-ctx
   mechanism overhead* (prior GB200 study: green@152/native = 0.987, ≈0 — re-confirm green@132/native
   on GH200). It is just the N=132 point of curve #2, so free.

We carry **both** `S_full` (headline) and `S_part` (diagnostic).

---

## 2. Curve source — in-harness, not the nvbench

`S_part = C / T_Q(N)` is a **ratio**, and `C` is measured by this harness as **host wall-clock**
(base DESIGN §3 — host stalls are a contention channel we count). Therefore the denominator must share
that instrument + session:

- **Baseline of record = a new in-harness `isolated --sms N` mode.** Reuses `run_isolated` verbatim but
  sets up a single green partition (existing `green_setup`) and routes `execute_once` onto it — same
  wall-clock primary + GPU-event secondary, same session, same refresh tagging, same pre-warm/arena.
  Measured at exactly the actual `N` values the splits produce, plus the coarse scaling grid.
- **`benchmark_tpch_ir_green` (the existing nvbench) is demoted to (a)** an independent **GPU-event
  cross-check** (does the wall-clock curve agree, or is it host-overhead-dominated?) **and (b)** the
  **operator-breakdown narrative** (`GAMAX_BENCH_STEP_CSV` shows which operator saturates — the *why*
  behind each curve). Its numbers are **not** used as ratio denominators. (Note: its "GB200 = 152 SMs"
  comment is stale for our 132-SM GH200.)

---

## 3. Split policy — full-device tiling, two intents per pair

**Full-device tiling only.** `N_a + N_b = 132` (zero idle SMs) so *both* mechanisms get the whole
device and the only difference is spatial-partition vs time-share. Sub-device / idle-SM splits answer
"what if you under-provision" — not our question — and are dropped.

**Mechanics** (new `green_setup_pair`, §7): split the full device **once** at `n_a` → `res[0]` is A
(rounded **up** to alignment); build B's ctx from **`remain` directly (the entire leftover, no
re-split)** → tiles all 132 with zero idle. Clamp `n_a ∈ [align, total−align]` (both sides non-empty).
Alignment makes exact ratios impossible, so cells are labeled by **intent** and the **actual**
`(N_a, N_b)` is recorded.

**Two configs per pair:**
- **`balanced`** — closest achievable to 50/50 (e.g. request 64 → 64/68).
- **`proportional`** — `N_a : N_b = T_a^full : T_b^full` (longer solo query gets proportionally more
  SMs); the driver computes it from the freshly-measured native baselines. Rationale: proportional is
  the makespan-optimal split *iff* scaling were linear (`T(N) ∝ 1/N`); queries *saturate*, so it will
  sometimes be beaten by a knee-aware split — which the showcase sweep exposes.
- **Self-pairs:** `T_a = T_b` ⇒ proportional ≡ balanced → emit **balanced only**.

**Showcase full sweep:** the most-lopsided green-safe pair **Q22+Q9** additionally gets a full
~9-point `N_a` grid (`{8,16,32,48,64,84,100,116,124}`, `N_b = 132−N_a`) → the "makespan &
aggregate-slowdown vs split ratio" money figure, validating whether proportional lands near the
achievable optimum and letting the SM-scaling knees explain where it sits.

---

## 4. Scale factor & green-safe subset

**SF10** (unchanged) — the under-utilization regime, cross-comparable with the archived time-shared +
serial baselines, identical shared-arena memory story.

**Green-safe subset — FINDING: all 22 are green-safe on GH200.** On GB200 (152 SMs, sm_100/Blackwell) a
subset failed under a *partitioned* context from **CUDA-13/CCCL device primitives that break on a
green-ctx stream** (`cudaMemcpyBatchAsync` SIGSEGV, `cub::DeviceTransform`/`DeviceScan`
`cudaErrorInvalidDevice`) — **not** stream-0 leaks — leaving only **17/22** even after the
`green_safe_elementwise.cuh` fixes (commit `4895fee`, confirmed on `alps_tpch`). **Re-tested on GH200
(132 SMs, sm_90/Hopper) during smoke, all 5 previously-excluded queries (Q1, Q4, Q13, Q20, Q21)
verify-correct and do NOT hang** — Q13's silent miscompute and Q21's hang were **Blackwell-specific**.
So the CUB-green blocker is an sm_100 issue; on GH200 the green-safe set is the full 22.

- The **step-0 green-verify gate** still runs all 22 systematically (smallest N=8 + a mid N + under
  pairs) to confirm; per-pair warmup validation (base DESIGN §8) + subprocess isolation catch any
  straggler non-fatally.
- **Consequence:** Q21 (native 44.8 ms, the largest query) is available → the base study's most dramatic
  co-runner is back, and **Q22+Q21 (25× lopsided, 15× time-shared slowdown in the base study) becomes
  the showcase** — the green result is now directly comparable to the base study's headline finding.
- The SM-scaling curve and native baselines cover **all 22** (not just 17).

---

## 5. Representative pairs (from GH200 native `T^full` + green-report efficiency)

Native SF10 `T^full` medians (ms) and scaling profile place each query; the 7 pairs span every
hypothesis-cell (wins *and* losses, not cherry-picked):

| # | Pair | T-ratio | Role | Prediction |
|---|---|---|---|---|
| 1 | **Q22+Q21** | 25× | **Extreme lopsided → SHOWCASE (full split sweep)**. Base study's dramatic pair (15× time-shared Q22 slowdown) | Proportional *starves* tiny Q22 (8 SMs → ~11× slower even concurrent); sweep finds the SM knee that cuts Q22 slowdown while Q21 (saturated) barely notices |
| 2 | Q22+Q9 | 10.5× | Lopsided (tiny launch-bound × big scaler) | Partition helps Q22 if given enough SMs |
| 3 | Q16+Q21 | 8.9× | Lopsided (saturate-early × biggest) | Partition helps Q16 slowdown |
| 4 | Q2+Q5 | 2.9× | Lopsided (launch-bound × scaler) | Aggregate-slowdown win |
| 5 | Q21+Q21 | 1.0× | Biggest saturating self-pair — **control** | Partition **loses most** (both need full device) |
| 6 | Q9+Q9 | 1.0× | Saturating self-pair — **control** | Partition loses |
| 7 | Q22+Q22 | 1.0× | Under-utilizing self-pair (time-share won prior) | Partition ties/wins |
| 8 | Q2+Q16 | 1.1× | Both under-utilizing (time-share won prior) | Partition matches time-share |

Reference `T^full` (ms): Q22 1.76, Q2 4.60, Q16 5.05, Q6 3.69, Q5 13.28, Q9 18.52, **Q21 44.83**.
Efficiency@152 (GB200): scale-well Q6 .94/Q5 .86/Q9 .85; saturate-early Q22 .49/Q16 .42/Q2 .22. Smoke on
GH200: Q22+Q21 proportional (8/124) → Ca=19.8 ms (Q22, 11× its 1.76 native), Cb=49.5 ms (Q21, 1.1×) —
proportional clearly starves Q22, motivating the sweep.

---

## 6. Metrics & figures

**Four strategies compared per pair:**
1. **serial-shortest-first** — `M = T_a^full + T_b^full`, `C_short = T_short`, `C_long = T_short+T_long`
   (derived from baselines, no new runs).
2. **time-shared concurrent** (`conc_single`, full device, regular streams).
3. **partitioned balanced** (`conc_green`).
4. **partitioned proportional** (`conc_green`).

**Metrics (all Python-derived):**
- **Makespan `M`** + speedup `(T_a^full+T_b^full)/M`.
- **Aggregate slowdown `Σ C_i/T_i^full`** — *the headline* (serial beat time-share 98% prior). Bet:
  small query's `C/T` stays ≈1 (dedicated SMs), big query's rises modestly → aggregate beats serial.
- **Per-query slowdown** `S_full = C_i/T_i^full` (headline) and `S_part = C_i/T_i(N_i)` (diagnostic;
  green cells only).

**Re-measure in-session** (not reuse the 2026-07-14 archive → one clock regime): native `T^full` for all
17 + time-shared `conc_single` for the 7 pairs. Archive becomes a cross-check.

**Figures:**
1. **Makespan & aggregate-slowdown vs split** (Q22+Q9) — money figure; balanced/proportional/
   time-shared/serial reference lines; SM-scaling-predicted optimum marked.
2. **Makespan/speedup — grouped bars**, 4 strategies × 7 pairs.
3. **Aggregate slowdown — grouped bars**, 4 strategies × 7 pairs (headline result).
4. **Per-query slowdown** — small query's `S_full` collapsing time-shared → partitioned; `S_part≈1`
   diagnostic panel.
5. **SM-scaling curves** `T_Q(N)/T_Q(132)` for the 17, knees marked (cross-checked vs GB200 report +
   nvbench GPU-event curve).

---

## 7. Implementation surface

- **Load stream:** construct `QueryRunner` **with the green stream** (`params.stream = green stream`) so
  load + execute both run on the partition — uniform (one context), the path the nvbench already
  validated green-safe (it resolves sources on the green stream), green overhead ≈0, loading untimed.
- **`green_setup_pair(device, n_a, GreenCtx& ga, GreenCtx& gb)`** — new reusable infra in
  `green_ctx.hpp`. Split once at `n_a`; A = `res[0]` (aligned up); **B = `remain` directly** (whole
  leftover, no re-split) → full-device tiling, zero idle. Clamp `n_a ∈ [align, total−align]`; record
  actual `ga.sms`/`gb.sms` via `cuGreenCtxGetDevResource`. Isolated `--sms N` reuses single
  `green_setup` unchanged.
- **Teardown order (RMM discipline):** green ctxs in an **outer** scope, `QueryRunner`s **inner** — runners
  (resident sources on the green stream) destroyed **before** `green_teardown` syncs+destroys the stream
  (same pattern as current lines 455/508/523/582).
- **CLI:** `--sms N` (isolated), `--sms-a N --sms-b M` or driver-computed split for concurrent; intent
  flag (`balanced|proportional|sweep`). `0`/omitted = native full device.
- **CSV schema:** add explicit **`sms_a`, `sms_b`** columns (actual sizes; native rows 132/132). New
  `run_type`: `iso_green`, `conc_green` (existing `iso_single`/`conc_single` = native baselines). New
  `policy` intents: `balanced|proportional|sweep`. (Explicit columns > parsing a `green_120_12` string.)
- **CMake:** add **`CUDA::cuda_driver`** to the `benchmark_tpch_concurrency` target; include
  `benchmarks/green-ctx/green_ctx.hpp` (include root `src/cuda`).

---

## 8. Driver, archive, scope

**Archive:** new sibling **`exp_results/tpch_concurrency_greenctx_20260715/`** (clean separation; reuses
2026-07-14 archive as cross-ref). New resumable subprocess-per-cell driver
**`script/run_tpch_concurrency_green.sh`**. Standalone plotter (numpy+csv+matplotlib, **no pandas**,
`.conda-build/bin/python`).

**Driver order (resumable, skip-if-file-exists):**
0. Metadata JSON — git commit/dirty, GH200/driver/CUDA, **actual device SM count + alignment**, N-grid,
   split intents, seed, no-pin caveat.
1. **Green-verify gate** — 17 candidates (+ retry 5 excluded) at a small N and a mid N; prune to
   GH200-green-safe.
2. **Native `T^full`** (`iso_single`) for all 17 — baselines + green@132/native ≈0-overhead control.
3. **SM-scaling** (`iso_green`) — 17 × {8,16,32,64,96,124,132} + 6 pair-queries at exact split-N.
4. **Time-shared** (`conc_single`) — 7 pairs.
5. **Partitioned** (`conc_green`) — balanced + proportional × 7 pairs (self-pairs balanced only);
   proportional split from step-2 `T^full`.
6. **Showcase sweep** (`conc_green sweep`) — Q22+Q9 × ~9-point `N_a` grid.
7. Periodic **native refresh** every ~50 cells (drift mechanism; prior was flat 1980 MHz, zero drift).

≈ **220–230 measured cells**, most fast single-query subprocesses (~1–3 h).

**Scope boundaries:**
- **Single-shot only — defer green-batch** to a follow-up (batch-partitioning = separate
  sustained-contention question; alternating with 4 runners × 2 partitions ≈ doubles the work).
- **One nsys capture** of a partitioned cell (Q22+Q9 proportional) as the **disjoint-partition
  correctness proof** — visually confirm the two queries occupy *disjoint* SMs (green ctx actually
  partitioned, not merely time-shared). Uses the gpu-profiling skill's GH200 recipe.

**Quality tail:** smoke one green cell before the sweep; after implementing run **code-review (max)**
and the **verify** skill. **No GPU work / heavy `-j` compiles during a live sweep** (perturbs host
wall-clock); launch with `nohup` + tracked completion waiter.

---

## 9. Accepted caveats (in addition to base DESIGN §15)

- **Memory is not partitioned** — only SM scheduling. HBM/L2/memory-controller stay shared, so `S_part`
  can exceed 1 even with disjoint SMs; that residual *is* a measured result (the isolation ceiling).
- **Alignment quantizes splits** — "50/50" and exact-proportional are the nearest aligned tiling; we
  label by intent and record actuals.
- **SF10 launch-bound regime** — small queries (Q2/Q16/Q22) can't fill even 8 SMs, so their "poor
  scaling" is partly launch-boundness, not a true operator ceiling (per the green-scaling report). This
  is *why* they're ideal carve-off candidates, but it means the win is partly a small-query artifact of
  SF10; a follow-up at SF30/100 would separate launch-bound from truly-saturated.
- **Green-safe subset is arch-dependent** — validated on GB200; the GH200 step-0 gate is authoritative.
- **No clock pinning** (non-root) — record NVML SM clock + throttle per sample; periodic refresh.
