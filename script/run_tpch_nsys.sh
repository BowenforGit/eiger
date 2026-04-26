#!/usr/bin/env bash
# Per-cell Nsight Systems profiling for benchmark_tpch_ir.
#
# Strategy:
#   * Manifest comes from an existing benchmark_tpch_ir --csv output
#     (e.g. tools/results/data/bench_e2e_sf{10,30}.csv). Re-extracted
#     fresh on every invocation so it tracks the latest sweep.
#   * For each query, we keep:
#       - the baseline cell (always), AND
#       - all cells of queries whose best/baseline ratio is below
#         GAP_THRESHOLD (default 0.95 — i.e. ≥5% potential win).
#     Queries with no meaningful spread get only their baseline
#     profiled, dropping ~566 cells (full sweep) to ~150–180.
#   * One `nsys profile` invocation per cell (NVBench --profile mode
#     runs each cell exactly once, no convergence loop), reports go
#     to /scratch/wubo/gamax_tpch_nsys/sf<N>/q<NN>_<label>.nsys-rep.
#   * Pre-flight throwers (multi-key/STRING groupby etc.) are already
#     marked Skipped=Yes by the bench fixture, so they're naturally
#     absent from the manifest.
#
# Usage:
#   ./script/run_tpch_nsys.sh <e2e-csv> <scale>
#
#   GAMAX_NSYS_EXECUTE=1   # actually run nsys (default: dry-run, prints
#                          # the manifest + planned commands)
#   GAMAX_NSYS_GAP=0.90    # tighter gap threshold (default 0.95)
#   GAMAX_NSYS_OUT=/path   # override output base (default
#                          # /scratch/wubo/gamax_tpch_nsys)
#
# Examples:
#   # dry-run for SF=10
#   ./script/run_tpch_nsys.sh tools/results/data/bench_e2e_sf10.csv 10
#
#   # actually profile SF=30
#   GAMAX_NSYS_EXECUTE=1 ./script/run_tpch_nsys.sh \
#     tools/results/data/bench_e2e_sf30.csv 30

set -euo pipefail

CSV=${1:?"path to bench_e2e_sf<N>.csv"}
SF=${2:?"scale factor (must match the CSV)"}
OUT_BASE=${GAMAX_NSYS_OUT:-/scratch/wubo/gamax_tpch_nsys}
GAP=${GAMAX_NSYS_GAP:-0.95}
EXECUTE=${GAMAX_NSYS_EXECUTE:-0}

if [[ ! -f "$CSV" ]]; then
    echo "error: CSV not found: $CSV" >&2
    exit 2
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

OUT_DIR="$OUT_BASE/sf$SF"
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
        env GAMAX_BENCH_SKIP_PREFLIGHT=1
        nsys profile
            -o "$out_prefix"
            --trace=cuda,nvtx
            --force-overwrite=true
            --stats=false
        "$BENCH_BIN" --profile --quiet
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
    echo "DRY RUN: planned=$n_planned cells. Set GAMAX_NSYS_EXECUTE=1 to profile."
    echo "Output would land in: $OUT_DIR"
fi
