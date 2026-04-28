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
> Result files are organized per machine. Pick a short identifier for
> the platform (`h100`, `gh200`, etc.) and pass it as `--machine` to
> the plot scripts and `GAMAX_MACHINE=...` to `script/run_tpch_nsys.sh`.
> All artefacts land under `tools/results/data/<machine>/tpch/`,
> `tools/results/pdf/tpch_bench/<machine>/`, and
> `/scratch/wubo/gamax_tpch_nsys/<machine>/sf<N>/`. The reference
> machine for the numbers in `docs/TPCH_JOURNEY.md` is `h100` (one
> NVIDIA H100 NVL); examples below use `<machine>` as a placeholder
> — substitute your identifier on the new platform.

## 1. Prerequisites

### Hardware
- NVIDIA GPU, sm_70+ (Volta/Turing/Ampere/Hopper). Memory: ≥24 GiB
  is comfortable for SF=10; SF=30 needs ~90 GiB free for the
  UserSpaceMM slab. CudaMM (`use_cuda_mm()`) works on smaller
  cards but loses the slab-warmup amortization between cells.

### Software
- CUDA Toolkit 12.x, including `nvcc` and `nsys`.
- CMake ≥ 3.23.1, Ninja (the build defaults to the `Ninja`
  generator).
- GCC/Clang with C++20 support (the toolchain on the dev box is GCC
  11; anything ≥10 should be fine).
- Apache Arrow + Parquet (C++ libs). On the dev box these come from
  `miniforge3`. Anything ≥17.0 known to work; the bench depends only
  on the `parquet::arrow::FileReader` projected-read API.
- Python 3.9+ with `pandas`, `matplotlib`, `numpy`, `pyarrow`,
  `duckdb` (only `duckdb` is needed for golden regen; the rest for
  plotting).

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
```

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
python3 script/gen_tpch_golden.py --sf 1  --out $TPCH_DATA_PATH
python3 script/gen_tpch_golden.py --sf 10 --out $TPCH_DATA_PATH
python3 script/gen_tpch_golden.py --sf 30 --out $TPCH_DATA_PATH
```

Each writes `$TPCH_DATA_PATH/sfN/golden/q01.parquet … q22.parquet`.
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
- `--use-cuda-mm` — run under CudaMM
  (`cudaMallocAsync`/`cudaFreeAsync`) instead of the UserSpaceMM
  slab. Useful for `compute-sanitizer`-style debugging where each
  allocation needs its own range.

## 5. Multi-scale correctness sweep (axis probe)

Verifies that every (query × axis-config) cell that runs without
throwing also matches its golden. Runs the same one-hot sweep the
bench uses (one baseline + per-axis algo variations), so it's the
right correctness gate before measuring perf.

```bash
TPCH_DATA_PATH=/scratch/<you> \
  ./build/src/cuda/benchmarks/tpch_ir_axis_probe \
  --scale 1  --verify --out /tmp/sf1_verify.csv  --quiet
TPCH_DATA_PATH=/scratch/<you> \
  ./build/src/cuda/benchmarks/tpch_ir_axis_probe \
  --scale 10 --verify --out /tmp/sf10_verify.csv --quiet
TPCH_DATA_PATH=/scratch/<you> \
  ./build/src/cuda/benchmarks/tpch_ir_axis_probe \
  --scale 30 --verify --out /tmp/sf30_verify.csv --quiet
```

Expected on the H100 reference:
| Scale | pass | verify_ok | mismatch | fail | runtime |
|--:|--:|--:|--:|--:|--:|
| SF=1  | 283 | 283 | 0 | 65 | 28 s |
| SF=10 | 283 | 283 | 0 | 65 | 51 s |
| SF=30 | 279 | 279 | 0 | 69 | 113 s |

`fail` cells are the documented unsupported shapes: multi-key/STRING
groupby (pattern A/B in `docs/PARTITION_KERNEL_BUGS.md`) and 4 SF=30-
only `partition_hash_join` build-bucket overflows (Q9/Q10/Q21). All
pass cells must verify-OK; any `mismatch` is a real regression — open
a bug.

## 6. End-to-end + per-step benchmark (NVBench)

Two outputs from one run: NVBench's per-cell GPU-time CSV (e2e) and
a side-channel per-step CSV (breakdown).

```bash
MACHINE=<machine>                                     # e.g. h100, gh200
mkdir -p tools/results/data/$MACHINE/tpch
TPCH_DATA_PATH=/scratch/<you> \
GAMAX_BENCH_STEP_CSV=tools/results/data/$MACHINE/tpch/bench_steps_sf10.csv \
./build/src/cuda/benchmarks/benchmark_tpch_ir \
  -a scale=10 \
  --min-samples 3 \
  --max-noise 5 \
  --timeout 8 \
  --quiet \
  --csv tools/results/data/$MACHINE/tpch/bench_e2e_sf10.csv

# Repeat for SF=30 (replace `10` → `30` in three places).
```

Wall on the H100 reference: ~95 min per scale.

Knobs:
- `--min-samples 3 --max-noise 5 --timeout 8` is the calibration that
  produced the numbers in `docs/TPCH_JOURNEY.md`'s SF=10 table. It
  trades a bit of noise (≤5%) for a 3–4× speedup over NVBench's
  defaults (`--min-samples 10 --max-noise 0.5`).
- `GAMAX_BENCH_STEP_CSV=<path>` enables per-step CSV output. Schema:
  `query, scale, filter, project, sort, group_by, eq_join,
   step_index, step_kind, n_samples, mean_ms, min_ms, max_ms,
   stddev_ms`. Drop the env var to skip the breakdown overhead.
- The bench iterates ~78 k Cartesian cells, ~99% of which are
  pruned cheaply by the at-most-one-set rule + skip-by-presence. Of
  the rest, ~283 actually gather samples per scale.

CSVs land in `tools/results/data/<machine>/tpch/` (gitignored — your
raw numbers). Save them somewhere persistent if you want to share.

## 7. Plot

```bash
# From the worktree root.
for sf in 10 30; do
  python3 tools/visualize/plot_tpch_bench.py \
    --machine <machine> \
    --scale   $sf
done
```

CSV inputs are derived from `tools/results/data/<machine>/tpch/`;
overrides via `--data-dir <path>` and `--out <path>`.

Produces four figures per scale into
`tools/results/pdf/tpch_bench/<machine>/`, each as both PDF (paper-
ready) and JPG (quick review):

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
MACHINE=<machine>                                     # e.g. h100, gh200
for sf in 10 30; do
  TPCH_DATA_PATH=/scratch/<you> \
  GAMAX_MACHINE=$MACHINE \
  GAMAX_NSYS_EXECUTE=1 \
  ./script/run_tpch_nsys.sh \
    tools/results/data/$MACHINE/tpch/bench_e2e_sf$sf.csv $sf
done
```

The driver reads the e2e CSV from §6 (it's the source of truth for
which cells made it through pre-flight), computes per-query
baseline-vs-best e2e ratios, and skips queries with no axis spread
(except their baseline). Defaults:

| env var               | default                          | purpose                                            |
|-----------------------|----------------------------------|----------------------------------------------------|
| `GAMAX_MACHINE`       | _required_                       | machine subdir under `GAMAX_NSYS_OUT` (e.g. `h100`)|
| `GAMAX_NSYS_EXECUTE`  | `0` (dry-run prints commands)    | set to `1` to actually run nsys                    |
| `GAMAX_NSYS_GAP`      | `0.95`                           | profile all cells of queries with `best/baseline < GAP` |
| `GAMAX_NSYS_OUT`      | `/scratch/wubo/gamax_tpch_nsys`  | output base; `<machine>/sf<N>/q<NN>_<label>.nsys-rep` underneath |

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
per scale at default `GAP=0.95` (≈170 cells/scale), ~35 min per
scale at `GAP=1` (≈260 cells/scale). Reports range 5–30 MB each;
budget ~10 GiB of free disk under `GAMAX_NSYS_OUT`.

Open one in the GUI:

```bash
nsys-ui /scratch/wubo/gamax_tpch_nsys/<machine>/sf10/q11_ejn-sort_merge_gfur.nsys-rep
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
# From the worktree root.
for sf in 10 30; do
  for mode in best default; do
    python3 tools/visualize/plot_top_kernels.py \
      --machine  <machine> \
      --scale    $sf \
      --cfg-mode $mode
  done
done
```

CSV inputs and PDF outputs are derived from the machine identifier:
the steps CSV is read from `tools/results/data/<machine>/tpch/`, nsys
reports from `/scratch/wubo/gamax_tpch_nsys/<machine>/`, and outputs
land in `tools/results/pdf/tpch_bench/<machine>/` plus
`tools/results/data/<machine>/tpch/`. Override with `--data-dir`,
`--nsys-base`, `--out-base`.

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
tools/results/pdf/tpch_bench/<machine>/tpch_top_kernels_{best,default}_sf{10,30}.pdf
tools/results/data/<machine>/tpch/top_kernels_{best,default}_sf{10,30}.csv
```

The CSV has `kernel_full, kernel_short, total_ns, total_ms,
n_invocs, n_step_rows` for every distinct kernel name observed in
the chosen cfg's nsys reports — useful for ranking past the top-8
the figure shows.

Wall: ~10–15 s on first call per report (nsys generates the
`.sqlite` index), ~2 s on subsequent calls (reuses the index).

## 10. What to send back

For a perf comparison or a paper figure, the canonical artefacts are:

- **CSVs** (from `tools/results/data/<machine>/tpch/`):
  `bench_e2e_sf{10,30}.csv` + `bench_steps_sf{10,30}.csv` +
  `top_kernels_{best,default}_sf{10,30}.csv`.
- **Verify CSVs**: `sf{1,10,30}_verify.csv` from §5 (proves
  correctness on the new platform).
- **Plots** (from `tools/results/pdf/tpch_bench/<machine>/`):
  `tpch_*_sf{10,30}.pdf` (e2e, breakdown, top-steps, kind-share,
  top-kernels {best,default}).
- **A few representative `.nsys-rep`** files from
  `/scratch/wubo/gamax_tpch_nsys/<machine>/sf{10,30}/` (full-set is
  ~120 MiB on the reference machine but can grow on slower GPUs —
  cherry-pick the queries with the biggest baseline-vs-best gap, e.g.
  Q3, Q5, Q11, Q17 at SF=30).
- **Run metadata**: GPU model + driver, CUDA version, CMake config
  (`build/CMakeCache.txt` → grep `CMAKE_CUDA_ARCHITECTURES`,
  `CMAKE_BUILD_TYPE`, Arrow/Parquet versions), and the host's
  `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv`.

## 11. Common gotchas

- **`Initializing memory pool fails.`** Cold-CUDA-context bug in
  `get_mm()`. Already fixed; if you see it on master after a clean
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
- **`Memory pool has initial size 0 bytes.`** `cudaMemGetInfo`
  returned 0 because the CUDA context wasn't initialized when
  `get_mm()` ran. Should be fixed on master; if it recurs, set
  `--use-cuda-mm` on `run_tpch_ir` to switch to the per-allocation
  manager.
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

## Reference pointers

- `docs/IR_ARCHITECTURE.md` — what each layer of the IR/lowering/exec
  stack does. Read first if you're going to modify the engine.
- `docs/TPCH_JOURNEY.md` — narrative of how the engine got here +
  the reference numbers, plus the corners cut and footguns. The
  perf+measurement arc lives at the end.
- `docs/PARTITION_KERNEL_BUGS.md` — the unsupported shapes the
  axis probe surfaces (multi-key groupby, partition_hash bucket
  overflow). Tactical guards turn them into clean errors.
- `tools/visualize/plot_tpch_bench.py` — the plot script for the
  e2e/breakdown/top-steps/kind-share figures. Has its own module
  docstring; extending the plots usually means adding a function and
  wiring it into `main()`.
- `tools/visualize/plot_top_kernels.py` — the top-kernel attribution
  plot. Reads the per-step CSV + the nsys-rep tree; supports both
  `--cfg-mode best` and `--cfg-mode default`.
- `script/run_tpch_nsys.sh` — Nsight Systems driver. Self-contained,
  no project-specific deps beyond the bench binary + CSV manifest.
