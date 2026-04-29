#!/usr/bin/env bash
# Per-cell Nsight Systems profiling for benchmark_tpch_ir.
#
# Mirrors run_tpch_perf.sh's CLI surface — same (machine, date,
# tag, scale) identifies one logical run; the e2e CSV path is
# derived from those rather than passed in.
#
# Strategy:
#   * Manifest comes from the bench's e2e CSV at
#     <csv-base>/<machine>/tpch/<date>/<tag>/bench_e2e_sf<N>.csv,
#     produced by `script/run_tpch_perf.sh`. Re-extracted fresh on
#     every invocation so it tracks the latest sweep.
#   * For each query we keep:
#       - the baseline cell (always), AND
#       - all cells of queries whose best/baseline ratio is below
#         --gap (default 0.95 — i.e. ≥5% potential win).
#     Queries with no meaningful spread get only their baseline
#     profiled, dropping ~566 cells (full sweep) to ~150–180.
#   * One `nsys profile` invocation per cell (NVBench --profile
#     mode runs each cell exactly once, no convergence loop).
#     Reports land at
#     <out-base>/<machine>/<date>/<tag>/sf<N>/q<NN>_<label>.nsys-rep
#     — mirroring the bench layout.
#   * Pre-flight throwers (multi-key/STRING groupby etc.) are
#     already marked Skipped=Yes by the bench fixture, so they're
#     naturally absent from the manifest.
#
# Required:
#   --machine <name>      e.g. h100, gh200 — must match the
#                         --machine the bench was run with.
#   --scale <N>           TPC-H scale factor.
#   --tag <name>          run identifier — must match the bench's
#                         --tag.
#   --tpch-data <path>    TPC-H parquet root (forwarded as
#                         TPCH_DATA_PATH to the bench binary).
#
# Optional:
#   --date <YYYYMMDD>     run date; default: today. Use this to
#                         profile a bench run from a previous day.
#   --gpu <id>            GPU index for nvbench's -d (default: 0).
#                         Should match the --gpu used at bench
#                         time, otherwise the cells profile on a
#                         different device than they timed on.
#   --csv-base <path>     base for the bench CSV input (default:
#                         tools/results/data).
#   --out-base <path>     base for nsys-rep output (default:
#                         /scratch/wubo/gamax_tpch_nsys).
#   --gap <ratio>         keep all cells of queries with
#                         best/baseline < ratio (default 0.95).
#                         Set to 1.0 to profile everything.
#   --execute             actually run nsys. Default is dry-run:
#                         prints the manifest + planned commands.
#   --conda-env <env>     conda env to activate before running.
#
# Output:
#   <out-base>/<machine>/<date>/<tag>/sf<N>/q<NN>_<label>.nsys-rep
#
# Examples:
#   # dry-run for SF=10 on h100, tag "after-count-all"
#   ./script/run_tpch_nsys.sh \
#       --machine h100 --scale 10 --tag after-count-all \
#       --tpch-data /scratch/wubo
#
#   # actually profile SF=30 on gh200
#   ./script/run_tpch_nsys.sh \
#       --machine gh200 --scale 30 --tag after-count-all \
#       --tpch-data /scratch/wubo --execute

set -euo pipefail

MACHINE=""
SCALE=""
TAG=""
TPCH_DATA=""
DATE=""
GPU="0"
CSV_BASE="tools/results/data"
OUT_BASE="/scratch/wubo/gamax_tpch_nsys"
GAP="0.95"
EXECUTE=0
CONDA_ENV=""

usage() {
    sed -n '2,/^set -e/p' "$0" | sed -e 's/^# \{0,1\}//' -e '/^set -e/d'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --machine)    MACHINE="$2"; shift 2;;
        --scale)      SCALE="$2"; shift 2;;
        --tag)        TAG="$2"; shift 2;;
        --tpch-data)  TPCH_DATA="$2"; shift 2;;
        --date)       DATE="$2"; shift 2;;
        --gpu)        GPU="$2"; shift 2;;
        --csv-base)   CSV_BASE="$2"; shift 2;;
        --out-base)   OUT_BASE="$2"; shift 2;;
        --gap)        GAP="$2"; shift 2;;
        --execute)    EXECUTE=1; shift;;
        --conda-env)  CONDA_ENV="$2"; shift 2;;
        -h|--help)    usage; exit 0;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2;;
    esac
done

# Required-arg validation. Same shape as run_tpch_perf.sh so error
# messages name the missing flag in user-facing form.
missing=()
[[ -z "$MACHINE"   ]] && missing+=("--machine")
[[ -z "$SCALE"     ]] && missing+=("--scale")
[[ -z "$TAG"       ]] && missing+=("--tag")
[[ -z "$TPCH_DATA" ]] && missing+=("--tpch-data")
if (( ${#missing[@]} > 0 )); then
    echo "error: missing required args: ${missing[*]}" >&2
    usage >&2
    exit 2
fi

# Default the date to today when not overridden — matches
# run_tpch_perf.sh's behaviour.
if [[ -z "$DATE" ]]; then
    DATE=$(date +%Y%m%d)
fi

if [[ ! -d "$TPCH_DATA" ]]; then
    echo "error: --tpch-data does not exist: $TPCH_DATA" >&2; exit 2
fi

# Same tag-shape guard as run_tpch_perf.sh.
if [[ "$TAG" =~ ^sf[0-9]+$ ]] || [[ "$TAG" =~ ^sf[0-9]+_ ]]; then
    echo "error: --tag must not encode the scale factor (got '$TAG')." >&2
    exit 2
fi

# Derive the e2e CSV path from machine/date/tag/scale.
CSV="$CSV_BASE/$MACHINE/tpch/$DATE/$TAG/bench_e2e_sf$SCALE.csv"
SF="$SCALE"

if [[ ! -f "$CSV" ]]; then
    echo "error: bench e2e CSV not found at the derived path:" >&2
    echo "       $CSV" >&2
    echo "       Did you run run_tpch_perf.sh with the same" >&2
    echo "       --machine / --date / --tag / --scale? Or pass" >&2
    echo "       --csv-base if your bench output isn't under" >&2
    echo "       tools/results/data." >&2
    exit 2
fi

# Conda activation (same pattern as run_tpch_perf.sh).
if [[ -n "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "error: conda not on PATH; cannot activate '$CONDA_ENV'" >&2
        exit 2
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# Resolve the bench binary relative to the script's worktree root.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
worktree_root=$(cd "$script_dir/.." && pwd)
BENCH_BIN="$worktree_root/build/src/cuda/benchmarks/benchmark_tpch_ir"

if [[ ! -x "$BENCH_BIN" ]]; then
    echo "error: bench binary not executable: $BENCH_BIN" >&2
    exit 2
fi
if ! command -v nsys >/dev/null 2>&1; then
    echo "error: nsys not on PATH" >&2
    exit 2
fi

OUT_DIR="$OUT_BASE/$MACHINE/$DATE/$TAG/sf$SF"
mkdir -p "$OUT_DIR"

# Extract the manifest with python (CSV parsing + per-query gap logic).
manifest=$(python3 - "$CSV" "$SF" "$GAP" <<'PYEOF'
import csv, sys

csv_path, sf, gap = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
NEUTRAL = ("default", "n/a")
AXES = ("filter", "project", "sort", "group_by", "eq_join")
AXIS_SHORT = {"filter":"fil","project":"prj","sort":"srt",
              "group_by":"grp","eq_join":"ejn"}

def short_label(row):
    set_axes = [a for a in AXES if row[a] not in NEUTRAL]
    if not set_axes:
        return "baseline"
    a = set_axes[0]
    return f"{AXIS_SHORT[a]}-{row[a]}"

cells = []
with open(csv_path, newline="") as f:
    for r in csv.DictReader(f):
        if r.get("Skipped") != "No": continue
        if int(r["scale"]) != sf: continue
        r["ms"] = float(r["GPU Time (sec)"]) * 1000.0
        r["q"]  = int(r["query"])
        r["label"] = short_label(r)
        cells.append(r)

# Decide which queries have a meaningful baseline-vs-best spread.
queries_with_gap = set()
queries_total = sorted({c["q"] for c in cells})
for q in queries_total:
    qc = [c for c in cells if c["q"] == q]
    base = next((c for c in qc if c["label"] == "baseline"), None)
    if base is None:
        # Should never happen: every query has a baseline cell.
        sys.stderr.write(f"# WARN: q{q} has no baseline cell, skipping\n")
        continue
    best_ms = min(c["ms"] for c in qc)
    if best_ms / base["ms"] < gap:
        queries_with_gap.add(q)

# Always keep the baseline; keep every cell of queries-with-gap.
kept = []
for c in cells:
    if c["label"] == "baseline" or c["q"] in queries_with_gap:
        kept.append(c)

# Sort: query asc, then label (baseline first, then alphabetical).
def sort_key(c):
    return (c["q"], 0 if c["label"] == "baseline" else 1, c["label"])
kept.sort(key=sort_key)

# Report stats to stderr (visible in dry-run + execute modes).
sys.stderr.write(
    f"# scale={sf} valid_cells={len(cells)} kept={len(kept)} "
    f"queries_with_gap={sorted(queries_with_gap)} "
    f"queries_total={len(queries_total)}\n")

# Emit one manifest line per kept cell.
for c in kept:
    print(",".join([str(c["q"]), c["filter"], c["project"], c["sort"],
                    c["group_by"], c["eq_join"], c["label"]]))
PYEOF
)

# Iterate manifest: one nsys invocation per kept cell.
n_planned=0
n_done=0
n_failed=0
while IFS=, read -r q fil prj srt grp ejn label; do
    [[ -z "$q" ]] && continue
    n_planned=$((n_planned + 1))
    out_prefix="$OUT_DIR/q$(printf '%02d' "$q")_${label}"
    # GAMAX_BENCH_SKIP_PREFLIGHT=1 disables the pre-flight try/catch
    # inside the bench fixture. Without it, the plan would execute
    # twice per cell (pre-flight on cudaStreamDefault + state.exec on
    # NVBench's stream) and nvtx_kern_sum's Total Time would
    # double-count every step. Cells in this manifest already passed
    # pre-flight in the timed run, so re-running the guard is wasteful.
    cmd=(
        env "TPCH_DATA_PATH=$TPCH_DATA" GAMAX_BENCH_SKIP_PREFLIGHT=1
        nsys profile
            -o "$out_prefix"
            --trace=cuda,nvtx
            --force-overwrite=true
            --stats=false
        "$BENCH_BIN" --profile --quiet
            -d "$GPU"
            -a "query=$q" -a "scale=$SF"
            -a "filter=$fil" -a "project=$prj" -a "sort=$srt"
            -a "group_by=$grp" -a "eq_join=$ejn"
    )
    if [[ "$EXECUTE" -eq 1 ]]; then
        printf '[%d/%s] q%d %s … ' "$n_planned" "?" "$q" "$label"
        if "${cmd[@]}" >/dev/null 2>&1; then
            n_done=$((n_done + 1))
            echo "OK  -> ${out_prefix}.nsys-rep"
        else
            n_failed=$((n_failed + 1))
            echo "FAIL"
        fi
    else
        printf '[%d] %s\n' "$n_planned" "${cmd[*]}"
    fi
done <<< "$manifest"

if [[ "$EXECUTE" -eq 1 ]]; then
    echo ""
    echo "Done: planned=$n_planned ok=$n_done fail=$n_failed"
    echo "Output: $OUT_DIR"
else
    echo ""
    echo "DRY RUN: planned=$n_planned cells. Add --execute to actually profile."
    echo "Output would land in: $OUT_DIR"
fi
