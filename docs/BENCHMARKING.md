<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# Reproducing the GaMax TPC-H benchmarks on a new platform

End-to-end recipe for someone porting the TPC-H bench (correctness +
e2e + per-step breakdown + Nsight Systems traces + plots) to a fresh
machine.

The reference machine for the numbers in `docs/TPCH_JOURNEY.md` is one
NVIDIA H100 NVL with CUDA 12.x. Anything sm_70+ should work; the
build is portable across architectures, just retarget
`CMAKE_CUDA_ARCHITECTURES`.

> [!NOTE]
> Every command below is run from the worktree root unless stated
> otherwise. Replace `/scratch/<you>` with your data path; everything
> hangs off `TPCH_DATA_PATH`.
>
> Result files are organized per machine, dated, and tagged. Pick a
> short identifier for the platform (`h100`, `gh200`, etc.) and pass
> it as `--machine` to every script in this doc; pick a tag for each
> round of experiments (`baseline`, `after-count-all`, etc.) and
> pass it as `--tag`. All artefacts land under
> `tools/results/data/<machine>/tpch/<DATE>/<TAG>/`,
> `tools/results/pdf/tpch_bench/<machine>/<DATE>/<TAG>/`, and
> `/scratch/wubo/gamax_tpch_nsys/<machine>/<DATE>/<TAG>/sf<N>/`. The reference
> machine for the numbers in `docs/TPCH_JOURNEY.md` is `h100` (one
> NVIDIA H100 NVL); examples below use `<machine>` as a placeholder
> — substitute your identifier on the new platform.

## 1. Prerequisites

### Hardware
- NVIDIA GPU, sm_70+ (Volta/Turing/Ampere/Hopper/Blackwell). Memory: ≥24 GiB
  is comfortable for SF=10. The default RMM arena uses 75% of free
  memory; use `GAMAX_MEM_RESERVE_BYTES` when communication or a large
  external workspace needs a different reserve.
- Also exercised on **GB200 (Grace-Blackwell, sm_100, aarch64)** with
  CUDA 13.0 and `-DCMAKE_CUDA_ARCHITECTURES=100`. SF=100 fits in the
  ~186 GiB GPU; **SF=300 is the single-GPU ceiling** — 20/22 queries
  fit, but Q1 (~14 GiB project column over 1.8 B lineitem rows) and
  Q21 (~21 GiB) OOM at every config. See `docs/TPCH_JOURNEY.md`
  §"SF=300 readiness".

### Software
- CUDA Toolkit 12.x, including `nvcc` and `nsys`.
- CMake ≥ 3.30.4, Ninja (the build examples use the `Ninja`
  generator).
- GCC/Clang with C++20 support (the toolchain on the dev box is GCC
  11; anything ≥10 should be fine).
- Apache Arrow + Parquet (C++ libs). On the H100 dev box these come
  from `miniforge3`; on the GB200 box they're `libarrow`/`libparquet`
  **24.0** in the conda env `dev` (`conda install -n dev
  --override-channels -c conda-forge libarrow libparquet`), discovered
  via `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX`. Anything ≥17.0 known to
  work; the bench depends only on the `parquet::arrow::FileReader`
  projected-read API.
- Python 3.9+ with `pandas`, `matplotlib`, `numpy`, `pyarrow`,
  `duckdb` (only `duckdb` is needed for golden regen; the rest for
  plotting). On a fresh box `pip install duckdb` — the TPC-H extension
  auto-downloads on first `INSTALL tpch`.

### Dataset
TPC-H Parquet tables, scale-factor named subdirectories under one
root. Layout:

```
$TPCH_DATA_PATH/
  sf1/
    customer.parquet
    lineitem.parquet
    nation.parquet
    orders.parquet
    part.parquet
    partsupp.parquet
    region.parquet
    supplier.parquet
    golden/
      q01.parquet … q22.parquet      # produced by script/gen_tpch_golden.py
  sf10/ …
  sf30/ …
  sf100/ …
  sf300/ …
```

Large datasets may instead use the strict manifest-controlled multi-file
layout:

```text
$TPCH_DATA_PATH/sf1000/
  manifest.json
  _SUCCESS
  region.parquet
  nation.parquet
  lineitem/lineitem.1.parquet ... lineitem.144.parquet
  orders/orders.1.parquet ... orders.144.parquet
  ... one 144-part directory for every other sharded table
```

The distributed runners accept
`--file-shard-strategy round_robin|contiguous_range` and
`--parquet-decode-threads N`. Multi-file directories without a valid manifest
and completion marker are rejected by default. Generation, validation,
publication, and assignment details are in
`docs/TPCH_MULTIFILE_DATASET.md`.

If you don't have Parquet, generate from `dbgen` and convert with
DuckDB or `pyarrow.parquet`. The Arrow schema must match TPC-H
spec types — DATE columns become Arrow `DATE32` (the reader converts
to int32 `YYYYMMDD` on load).

## 2. Configure + build

```bash
cmake -G Ninja -B build \
  -DUSE_NVIDIA=ON \
  -DBUILD_TPCH=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90      # sm_90 = H100; use 80 for A100, 75 Turing, 70 V100
# If Arrow/Parquet aren't on the system path, point cmake at them:
#   -DArrow_DIR=$CONDA_PREFIX/lib/cmake/Arrow \
#   -DParquet_DIR=$CONDA_PREFIX/lib/cmake/Parquet

cmake --build build --target benchmark_tpch_ir tpch_ir_axis_probe run_tpch_ir -j
```

Targets you'll need:
- `benchmark_tpch_ir` — NVBench-driven sweep (e2e + per-step CSV).
- `tpch_ir_axis_probe` — fast correctness probe (no measurement loop).
- `run_tpch_ir` — single-query CLI driver, used for ad-hoc repros and
  golden verification.

`Release` is the default `CMAKE_BUILD_TYPE`. Don't switch to `Debug`
for performance numbers — the bench will still work but kernels will
be ~5× slower.

## 3. Generate Parquet goldens (once per scale)

The verifier needs reference results. DuckDB's TPC-H extension is
spec-compliant and fast:

```bash
TPCH_DATA_PATH=/scratch/<you>
for sf in 1 10 30 100 300; do
  python3 script/gen_tpch_golden.py --sf $sf --out $TPCH_DATA_PATH
done
```

DuckDB is CPU-only and fast even at scale (on the GB200 box: SF1 ~2 s,
SF10 ~33 s, SF100/SF300 a few minutes each; all 5 scales ≈ 9 min
total, run sequentially to avoid RAM oversubscription on the big
ones). Each writes `$TPCH_DATA_PATH/sfN/golden/q01.parquet … q22.parquet`.
The script registers the existing Parquet tables as DuckDB views, so
there's no data duplication. The DECIMAL→DOUBLE and DATE→INT32
coercions match what GaMax produces (see `tpch_reader.cpp`'s
DATE32→YYYYMMDD path), so the verifier compares like-for-like.

## 4. Sanity check: one query, end-to-end + verify

```bash
TPCH_DATA_PATH=/scratch/<you> ./build/src/cuda/tpch/run_tpch_ir 6 \
  --scale 10 --verify
```

You should see something like

```
[TpchTableReader] Opening Parquet file: /scratch/<you>/sf10/lineitem.parquet
…
VERIFY: comparing against /scratch/<you>/sf10/golden/q06.parquet
VERIFY: OK
```

Useful flags on `run_tpch_ir`:
- `--scale N` — pick the data scale (1, 10, 30, …).
- `--verify` — compare result columns to the parquet golden.
- `--print-plan` — dump the lowered ExecPlan before running.
- `--no-fuse-project`, `--eq-join-algo <name>` — exercise specific
  hint values matching the bench axes.
- `--rmm-resource cuda` — use direct RMM `cudaMalloc`/`cudaFree`
  with tracking (the diagnostic/Compute Sanitizer mode). Performance
  runs should use the default arena.

## 5. Multi-scale correctness sweep (axis probe)

Verifies that every (query × axis-config) cell that runs without
throwing also matches its golden. Runs the same one-hot sweep the
bench uses (one baseline + per-axis algo variations), so it's the
right correctness gate before measuring perf.

```bash
TPCH_DATA_PATH=/scratch/<you> \
  ./build/src/cuda/benchmarks/tpch_ir_axis_probe \
  --scale 1  --verify --out /tmp/sf1_verify.csv   --quiet
TPCH_DATA_PATH=/scratch/<you> \
  ./build/src/cuda/benchmarks/tpch_ir_axis_probe \
  --scale 10 --verify --out /tmp/sf10_verify.csv  --quiet
TPCH_DATA_PATH=/scratch/<you> \
  ./build/src/cuda/benchmarks/tpch_ir_axis_probe \
  --scale 30 --verify --out /tmp/sf30_verify.csv  --quiet
TPCH_DATA_PATH=/scratch/<you> \
  ./build/src/cuda/benchmarks/tpch_ir_axis_probe \
  --scale 100 --verify --out /tmp/sf100_verify.csv --quiet
```

Expected on the H100 reference (after the SF=100 readiness arc;
see `docs/TPCH_JOURNEY.md`):

| Scale  | pass | verify_ok | mismatch | fail | total | runtime |
|--:|--:|--:|--:|--:|--:|--:|
| SF=1   | 277  | 277       | 0        | 65   | 342   |  29 s   |
| SF=10  | 277  | 277       | 0        | 65   | 342   |  53 s   |
| SF=30  | 273  | 273       | 0        | 69   | 342   | 117 s   |
| SF=100 | 267  | 267       | 0        | 81   | 348   | 460 s   |

All `pass` cells verify byte-for-byte against the parquet golden;
any `mismatch` is a real regression — open a bug. The `fail` cells
are documented unsupported shapes:

- multi-key / STRING groupby on `sort_*` and `partition_*` algos
  (pattern A/B in `docs/PARTITION_KERNEL_BUGS.md`),
- `partition_hash_join` build-bucket overflows (4 cells at SF=30
  on Q9/Q10/Q21; 8 cells at SF=100 on Q8/Q11/Q12/Q20 — the
  cardinality-driven bucket-skew shifts which queries trip),
- `Invalid type_id: 12` under `project=fuse` on date columns,
- **SF=100 only:** all 13 cells of Q1 fail with `std::bad_alloc`
  from the slab allocator — fragmentation OOM (root-caused,
  blocked on AVG-without-explicit-ones-column; details in
  `docs/TPCH_JOURNEY.md` §"SF=100 readiness").

The SF=1/10/30 totals dropped from `348 → 342` after the SF=100
arc landed: queries whose Aggregate previously synthesized a
`Project(items=[..., lit(1)])` (which made the `project=fuse/nofuse`
axis applicable) now synthesize a `Reorder` instead, so those
project-axis variants are reclassified as `n/a` rather than
running. Same coverage, just labelled differently.

**Every expected SF=100 capacity failure is a host-side C++ exception**
(`rmm::out_of_memory`, or `std::runtime_error` from a guard) — the bench
fixture's pre-flight `try/catch` catches each one as
`state.skip(...)` and the run continues. No cell observed at
SF=100 should produce a CUDA-level error in the default arena path.
Confirmed RMM OOM is drained and recovered at the cell boundary;
generic or fatal CUDA errors are not treated as recoverable.

## 6. End-to-end + per-step benchmark (NVBench)

`script/run_tpch_perf.sh` wraps `benchmark_tpch_ir` with the
calibration this doc's numbers came from, organizes outputs into
a dated/tagged folder, and refuses to overwrite an existing CSV
(so the same `--tag` can't silently merge into an older run).

```bash
TAG=<your-tag>                              # e.g. baseline, after-count-all
for sf in 10 30; do
  ./script/run_tpch_perf.sh \
    --machine    <machine> \
    --scale      $sf \
    --tag        $TAG \
    --tpch-data  /scratch/<you> \
    --conda-env  <env> \
    --gpu        0
done
```

Wall on the H100 reference: ~95 min per scale.

Inputs / knobs:
- `--machine` — short identifier (`h100`, `gh200`, …). Picks the
  subdir under `tools/results/data/`.
- `--scale` — TPC-H scale factor (1 / 10 / 30 / 100). One run per
  invocation; the same `--tag` is meant to span multiple scales,
  so the SF goes in the filename, not the tag.
- `--tag` — your run identifier. Caller's choice; must NOT encode
  the scale factor (the script rejects `sfN` / `sfN_…` tags).
- `--tpch-data` — TPC-H parquet root (the dir containing
  `sf<N>/<table>.parquet`). The script forwards this as
  `TPCH_DATA_PATH`.
- `--conda-env` (optional) — conda env to activate before running.
- `--gpu` (optional, default `0`) — GPU index, passed to NVBench's
  `-d` flag.

Outputs:

```
tools/results/data/<machine>/tpch/<YYYYMMDD>/<TAG>/bench_e2e_sf<N>.csv
tools/results/data/<machine>/tpch/<YYYYMMDD>/<TAG>/bench_steps_sf<N>.csv
```

The bench is hardcoded to `--min-samples 3 --max-noise 5
--timeout 8` (the calibration that produced the numbers in
`docs/TPCH_JOURNEY.md`'s SF=10 table — trades ≤5% noise for a
3–4× speedup over NVBench's defaults). Per-step CSV schema:
`query, scale, filter, project, sort, group_by, eq_join,
step_index, step_kind, n_samples, mean_ms, min_ms, max_ms,
stddev_ms`.

CSVs are gitignored (these are your raw numbers — save somewhere
persistent if you want to share). The dated/tagged structure is
the convention every downstream tool (`run_tpch_nsys.sh`, the
plotters) reads from, so use the same `--machine` / `--tag` /
`--scale` triple consistently across one round of experiments.

## 7. Plot

```bash
# From the worktree root. DATE matches the dated folder produced
# by run_tpch_perf.sh (use `ls tools/results/data/<machine>/tpch/`
# to find it); TAG matches the --tag used at bench time.
DATE=20260429
TAG=<your-tag>
for sf in 10 30; do
  python3 tools/visualize/plot_tpch_bench.py \
    --machine <machine> \
    --date    $DATE \
    --tag     $TAG \
    --scale   $sf
done
```

CSV inputs are derived from
`tools/results/data/<machine>/tpch/<DATE>/<TAG>/`; overrides via
`--data-dir <path>` and `--out <path>`.

Produces four figures per scale into
`tools/results/pdf/tpch_bench/<machine>/<DATE>/<TAG>/`, each as
both PDF (paper-ready) and JPG (quick review):

- `tpch_e2e_sf<N>.pdf` — 22 subfigures (2 rows × 11 cols), one bar
  per cell coloured by the axis it varies. Three-bar version
  (`baseline / worst / best`) annotated with ms.
- `tpch_breakdown_sf<N>.pdf` — same layout, three stacked bars per
  query: baseline cell, worst-per-step composite, best-per-step
  composite. Per-step (not per-kind) minima/maxima — repeated kinds
  pick configs independently.
- `tpch_top_steps_sf<N>.pdf` — single-panel: top-8 most expensive
  individual steps across the workload under best-cfg-per-step.
  Each bar annotated with `pct% / ms / [picked cfg]`.
- `tpch_kind_share_sf<N>.pdf` — single-panel: per-StepKind time
  share of the best-per-step workload total, all kinds shown.

Style is muted-Tableau-10 with bigger fonts; the script's docstring
covers what each function does.

The kernel-attribution figure (`tpch_top_kernels_*`) is a separate
script — see §9 below. It depends on the nsys reports from §8, so
generate those first.

## 8. Per-cell Nsight Systems traces

For each non-skipped (query, scale, cfg) cell that has a meaningful
spread, we run `nsys profile` once with the bench fixture in
`--profile` mode (single execution, no convergence loop):

```bash
TAG=<your-tag>                              # same --tag as §6
for sf in 10 30; do
  ./script/run_tpch_nsys.sh \
    --machine    <machine> \
    --scale      $sf \
    --tag        $TAG \
    --tpch-data  /scratch/<you> \
    --execute
done
```

CLI mirrors `run_tpch_perf.sh`'s flag surface — same
`(machine, date, tag, scale)` identifies one logical run.
The bench's e2e CSV is read from
`tools/results/data/<machine>/tpch/<date>/<tag>/bench_e2e_sf<N>.csv`
and the path is derived, not passed in.

| flag                 | default                          | purpose                                                       |
|----------------------|----------------------------------|---------------------------------------------------------------|
| `--machine <name>`   | _required_                       | machine subdir; must match the bench run                      |
| `--scale <N>`        | _required_                       | TPC-H scale factor                                            |
| `--tag <name>`       | _required_                       | run tag — same as `--tag` passed to `run_tpch_perf.sh`        |
| `--tpch-data <path>` | _required_                       | TPC-H parquet root; forwarded as `TPCH_DATA_PATH`             |
| `--date <YYYYMMDD>`  | today                            | run date; override to profile a bench from a previous day     |
| `--gpu <id>`         | `0`                              | GPU index for `-d` (should match the bench run)               |
| `--csv-base <path>`  | `tools/results/data`             | base for the bench e2e CSV input                              |
| `--out-base <path>`  | `/scratch/wubo/gamax_tpch_nsys`  | base for nsys-rep output; `<machine>/<date>/<tag>/sf<N>/q<NN>_<label>.nsys-rep` underneath |
| `--gap <ratio>`      | `0.95`                           | profile all cells of queries with `best/baseline < gap` (set `1.0` to profile every passing cell) |
| `--execute`          | _absent (dry-run)_               | flag — actually run nsys. Default prints the manifest + planned commands. |
| `--conda-env <env>`  | _none_                           | conda env to activate before launching                        |

Per cell: `--trace=cuda,nvtx --force-overwrite=true --stats=false`.
The CUDA + NVTX trace combination captures kernel launches plus the
executor's per-step NVTX ranges (`step_profile.cpp`), giving an at-a-
glance per-step timeline in Nsight UI.

The driver also sets `GAMAX_BENCH_SKIP_PREFLIGHT=1` on every nsys
invocation. Without it, the bench fixture's pre-flight try/catch
re-runs the plan on `cudaStreamDefault` before `state.exec`, so each
`:step[N]:<Kind>` NVTX range is entered twice per cell and
`nvtx_kern_sum`'s "Total Time" is exactly 2× the actual per-execution
kernel cost. With the env var set, pre-flight is skipped (cells in
the manifest already passed it during §6) and the raw nsys numbers
match the bench's GPU Time within overhead.

Wall on the H100 reference: ~7 s/cell + ~3 s nsys overhead ≈ 25 min
per scale at default `--gap 0.95` (≈170 cells/scale), ~35 min per
scale at `--gap 1.0` (≈260 cells/scale). Reports range 5–30 MB each;
budget ~10 GiB of free disk under `--out-base`.

Open one in the GUI:

```bash
nsys-ui /scratch/wubo/gamax_tpch_nsys/<machine>/<DATE>/<TAG>/sf10/q11_ejn-sort_merge_gfur.nsys-rep
```

For batch post-processing (e.g. extract top-kernel tables to CSV):

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  /path/to/foo.nsys-rep > foo_kernel_sum.csv
```

## 9. Top-kernel attribution figure

Cross-references the per-step CSV from §6 with the nsys reports from
§8 to identify which CUDA kernels dominate the workload, with each
step optionally attributed to either the best-per-step cfg or the
default cfg.

```bash
# From the worktree root. DATE / TAG match the run from §6.
DATE=20260429
TAG=<your-tag>
for sf in 10 30; do
  for mode in best default; do
    python3 tools/visualize/plot_top_kernels.py \
      --machine  <machine> \
      --date     $DATE \
      --tag      $TAG \
      --scale    $sf \
      --cfg-mode $mode
  done
done
```

CSV inputs, nsys reports, and outputs are all keyed off
`(machine, date, tag)`: steps CSV from
`tools/results/data/<machine>/tpch/<DATE>/<TAG>/`, nsys reports
from `/scratch/wubo/gamax_tpch_nsys/<machine>/<DATE>/<TAG>/`, PDFs
to `tools/results/pdf/tpch_bench/<machine>/<DATE>/<TAG>/`,
top-kernels CSV alongside the steps CSV. Override base dirs with
`--data-dir`, `--nsys-base`, `--out-base`.

For each `(query, step_index)`:
- `--cfg-mode best` picks the cfg that minimized `mean_ms` for that
  step (matches the breakdown plot's "best-per-step" composite —
  oracle per-step picker; different steps can pick different cfgs).
- `--cfg-mode default` always picks the all-default cell (the bench
  baseline matching `bench_defaults`).

Then opens the matching `q<NN>_<label>.nsys-rep`, runs
`nsys stats --report nvtx_kern_sum --format csv` once per report
(with the SQLite cache reused across calls), filters rows by the
`:step[<si>]:` NVTX prefix, and aggregates `Total Time / NVTX Inst`
into per-kernel global totals. Two kernels are counted as the same
iff their fully-demangled names are bit-equal.

Outputs (PDF + JPG sidecar + CSV) per scale × mode:

```
tools/results/pdf/tpch_bench/<machine>/<DATE>/<TAG>/tpch_top_kernels_{best,default}_sf{10,30}.pdf
tools/results/data/<machine>/tpch/<DATE>/<TAG>/top_kernels_{best,default}_sf{10,30}.csv
```

The CSV has `kernel_full, kernel_short, total_ns, total_ms,
n_invocs, n_step_rows` for every distinct kernel name observed in
the chosen cfg's nsys reports — useful for ranking past the top-8
the figure shows.

Wall: ~10–15 s on first call per report (nsys generates the
`.sqlite` index), ~2 s on subsequent calls (reuses the index).

## 10. What to send back

For a perf comparison or a paper figure, the canonical artefacts are:

- **CSVs** (from `tools/results/data/<machine>/tpch/<DATE>/<TAG>/`):
  `bench_e2e_sf{10,30}.csv` + `bench_steps_sf{10,30}.csv` +
  `top_kernels_{best,default}_sf{10,30}.csv`.
- **Verify CSVs**: `sf{1,10,30}_verify.csv` from §5 (proves
  correctness on the new platform).
- **Plots** (from
  `tools/results/pdf/tpch_bench/<machine>/<DATE>/<TAG>/`):
  `tpch_*_sf{10,30}.pdf` (e2e, breakdown, top-steps, kind-share,
  top-kernels {best,default}).
- **A few representative `.nsys-rep`** files from
  `/scratch/wubo/gamax_tpch_nsys/<machine>/<DATE>/<TAG>/sf{10,30}/`
  (full-set is ~120 MiB on the reference machine but can grow on
  slower GPUs — cherry-pick the queries with the biggest
  baseline-vs-best gap, e.g. Q3, Q5, Q11, Q17 at SF=30).
- **Run metadata**: GPU model + driver, CUDA version, CMake config
  (`build/CMakeCache.txt` → grep `CMAKE_CUDA_ARCHITECTURES`,
  `CMAKE_BUILD_TYPE`, Arrow/Parquet versions), and the host's
  `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv`.

## 11. Common gotchas

- **Allocator initialization fails on a cold CUDA context.** Already fixed;
  if you see it on master after a clean
  build, your `cmake --build` didn't relink — `cmake --build build
  --target benchmark_tpch_ir tpch_ir_axis_probe -j`.
- **`offset overflow while concatenating arrays`** at SF=30 on Q13/
  Q19. Means your build still has the int32-offset Arrow concat
  path. Pull the chunked-string code in `tpch_reader.cpp`.
- **All 22 queries report `verify: skip`** — you didn't pass
  `--verify` to `tpch_ir_axis_probe`, or `TPCH_DATA_PATH` doesn't
  contain `golden/`.
- **`nsys` reports the trace but the timeline is mostly memcpy**.
  The bench's Phase-1 (parquet decode + H2D) is included in the
  trace; the actual query GPU work is the second region. Use the
  NVTX track in Nsight UI to filter to the steps' ranges.
- **The computed RMM arena is zero or too large.** Check the strict
  `GAMAX_MEM_RESERVE_BYTES` rules in `docs/MEMORY_ALLOCATOR.md` and
  make sure no other process has consumed device memory before prewarm.
- **Pre-flight catches a config that should work.** The bench's
  pre-flight runs the plan once on `cudaStreamDefault` before
  `state.exec`. If a cell throws during pre-flight, NVBench skips
  it. Inspect with `run_tpch_ir <q> --scale N --eq-join-algo …` to
  reproduce outside NVBench's measurement loop.
- **`nsys` per-kernel times are exactly 2× what e2e suggests.** The
  bench fixture re-runs the plan in pre-flight before `state.exec`,
  so each `:step[N]:<Kind>` NVTX range is entered twice and
  `nvtx_kern_sum`'s totals are doubled. The §8 driver sets
  `GAMAX_BENCH_SKIP_PREFLIGHT=1` to disable pre-flight under
  `nsys profile`. If you invoke `nsys` manually, set this env var.
  The §9 plotter normalizes by `NVTX Inst` so it works either way.
- **At SF=100 some cells exhaust the arena.** Only
  `rmm::out_of_memory` is eligible for cell recovery. A sticky CUDA
  execution error is fatal and must not be hidden by allocator reset.
- **All SF=100 axis-probe failures are host-side C++
  exceptions.** No CUDA-level errors with the default allocator.
  This means a partial run is recoverable (re-run with a fresh
  `--tag`) and SLURM can interrupt without leaving the GPU in
  an unusable state. **This property does NOT hold at SF=300** — see
  the next gotcha.
- **At SF=300 the single-process NVBench sweep dies early with
  `cudaErrorIllegalAddress`.** `partition_hash` join faults at
  `>2^30` rows (SF=300 lineitem ≈ 1.8 B crosses 2^30 = 1.07 B; SF=100
  at 600 M stayed under it). Unlike the SF=100 `bad_alloc` throws,
  this is a **sticky CUDA error that poisons the whole process** —
  every later cell's `cudaMemGetInfo`/`cudaMalloc` then returns 0
  (`Memory pool has initial size 0 bytes` cascade) and the run exits 1
  after only a handful of valid cells. Two fixes, used together:
    1. **Exclude the partition variants** so the faulting cell never
       runs (`default`=hash and the `sort_*` alternatives stay):
       `-a "eq_join=[sort_merge_gfur,sort_merge_gftr,default,n/a]"
       -a "group_by=[sort_gftr,sort_gftr_opt,sort_gfur,default,n/a]"`.
       (`-a "axis=[v1,v2,...]"` restricts a string axis to that subset;
       `-a scale=300` likewise overrides the scale axis.)
    2. **Run per-query** (`-a query=N` in a separate process each),
       distributed across nodes, so any residual crash is contained to
       one query. Merge the per-query e2e CSVs afterward — but a query
       with **0 measured cells** (all-OOM, e.g. Q1/Q21 at SF=300) gets
       a 12-column nvbench e2e header (no timing columns) vs 17 for
       measured queries; drop those files when merging or pandas chokes.
  Partition-based group_by (`partition_*`) is the same risk and is
  excluded for the same reason. phj also has a *separate, scale-
  independent* limit: a build-side **bucket overflow** on skewed FKs
  (`max_build_bucket_size=4096`), a clean refuse-to-run throw seen for
  Q8/Q10/Q12/Q18/Q20 even at SF=100 — HASH/SORT are correct there.

## Distributed TPC-H

This file covers the single-GPU NVBench and operator-axis workflow. For the
current distributed campaign runner, dependencies, build, datasets, local and
multi-node launch, resume/retry behavior, profiling, and output layout, use
[`DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md`](DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md).

Distributed timing and phase semantics are maintained in
[`DISTRIBUTED_TPCH_ARCHITECTURE.md`](DISTRIBUTED_TPCH_ARCHITECTURE.md), while
Nsight metric and overlap interpretation is in
[`DISTRIBUTED_TPCH_PROFILING.md`](DISTRIBUTED_TPCH_PROFILING.md).

## Reference pointers

- `docs/IR_ARCHITECTURE.md` — what each layer of the IR/lowering/exec
  stack does. Read first if you're going to modify the engine.
- `docs/TPCH_JOURNEY.md` — narrative of how the engine got here +
  the reference numbers, plus the corners cut and footguns. The
  perf+measurement arc lives at the end.
- `docs/PARTITION_KERNEL_BUGS.md` — the unsupported shapes the
  axis probe surfaces (multi-key groupby, partition_hash bucket
  overflow). Tactical guards turn them into clean errors.
- `script/run_tpch_perf.sh` — bench wrapper. One run per
  `(machine, scale, tag, gpu)`; outputs to a dated/tagged folder
  and refuses to overwrite. The convention every downstream tool
  reads from.
- `script/run_tpch_nsys.sh` — Nsight Systems driver. Same flag
  surface as `run_tpch_perf.sh` (`--machine --scale --tag
  --tpch-data` required, `--execute` to actually profile,
  default is dry-run); derives the bench's e2e CSV path from the
  `(machine, date, tag, scale)` triple.
- `tools/visualize/plot_tpch_bench.py` — the plot script for the
  e2e/breakdown/top-steps/kind-share figures. Takes
  `--machine --date --tag --scale`. Extending the plots usually
  means adding a function and wiring it into `main()`.
- `tools/visualize/plot_top_kernels.py` — the top-kernel attribution
  plot. Reads the per-step CSV + the nsys-rep tree under the same
  `(machine, date, tag)` keys; supports `--cfg-mode best` /
  `--cfg-mode default`.
- `tools/visualize/extract_kernel_times.py` — small utility that
  emits a per-query baseline / best-cfg pure-kernel-time CSV from
  the nsys reports. Same `--machine --date --tag --scale` triple.
