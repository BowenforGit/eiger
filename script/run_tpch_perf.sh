#!/usr/bin/env bash
# Run benchmark_tpch_ir for one (machine, scale, tag) and write
# the per-cell e2e + per-step CSVs to a dated/tagged folder.
#
# Required:
#   --machine <name>      e.g. h100, gh200
#   --scale <N>           TPC-H scale factor (1 / 10 / 30 / 100 / ...)
#   --tag <name>          run identifier of the caller's choice;
#                         must NOT include scale-factor information —
#                         multiple --scale invocations under the same
#                         tag are expected to share one TAG folder.
#   --tpch-data <path>    TPC-H parquet root (the dir that contains
#                         sf<N>/<table>.parquet)
#
# Optional:
#   --csv-base <path>     CSV output base (default: tools/results/data)
#   --conda-env <env>     conda env name to activate before running
#   --gpu <id>            GPU index (passed to nvbench's -d flag);
#                         default: 0
#
# Output:
#   $CSV_BASE/$MACHINE/tpch/<YYYYMMDD>/<TAG>/bench_e2e_sf<N>.csv
#   $CSV_BASE/$MACHINE/tpch/<YYYYMMDD>/<TAG>/bench_steps_sf<N>.csv
#
# If either output CSV already exists at the path, the script
# fails fast WITHOUT running the bench (no overwrite).
#
# NVBench calibration is hardcoded to match the BENCHMARKING.md
# recipe: --min-samples 3 --max-noise 5 --timeout 8. Changing
# those changes what the resulting numbers mean; if you need
# different settings, edit this script.

set -euo pipefail

MACHINE=""
SCALE=""
TAG=""
TPCH_DATA=""
CSV_BASE="tools/results/data"
CONDA_ENV=""
GPU="0"

usage() {
    sed -n '2,/^set -e/p' "$0" | sed -e 's/^# \{0,1\}//' -e '/^set -e/d'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --machine)    MACHINE="$2"; shift 2;;
        --scale)      SCALE="$2"; shift 2;;
        --tag)        TAG="$2"; shift 2;;
        --tpch-data)  TPCH_DATA="$2"; shift 2;;
        --csv-base)   CSV_BASE="$2"; shift 2;;
        --conda-env)  CONDA_ENV="$2"; shift 2;;
        --gpu)        GPU="$2"; shift 2;;
        -h|--help)    usage; exit 0;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2;;
    esac
done

# Validate required args. We check explicitly (rather than `set -u` /
# `${VAR:?}`) so the error names the missing flag in user-facing form.
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

# Sanity-check the TPC-H data path exists. The actual sf<N>
# subdirectory check happens inside the bench fixture; failing here
# at least catches the wrong-base-path case.
if [[ ! -d "$TPCH_DATA" ]]; then
    echo "error: --tpch-data does not exist: $TPCH_DATA" >&2; exit 2
fi

# Tag must not include scale-factor information. We don't try to
# parse it (the user owns the convention); just refuse the obvious
# `sfN` case so a silently-wrong path doesn't get baked into the
# layout.
if [[ "$TAG" =~ ^sf[0-9]+$ ]] || [[ "$TAG" =~ ^sf[0-9]+_ ]]; then
    echo "error: --tag must not encode the scale factor (got '$TAG')." >&2
    echo "       Multiple scales for the same logical run share one TAG dir." >&2
    exit 2
fi

# Derive paths.
DATE=$(date +%Y%m%d)
OUT_DIR="$CSV_BASE/$MACHINE/tpch/$DATE/$TAG"
E2E_CSV="$OUT_DIR/bench_e2e_sf$SCALE.csv"
STEPS_CSV="$OUT_DIR/bench_steps_sf$SCALE.csv"

# Pre-flight: refuse to overwrite. Either CSV existing means a
# previous run for this (machine, date, tag, scale) already
# happened; the user should pick a new tag or move the existing
# CSVs out of the way before re-running.
for f in "$E2E_CSV" "$STEPS_CSV"; do
    if [[ -e "$f" ]]; then
        echo "error: output CSV already exists: $f" >&2
        echo "       refusing to overwrite. Pick a different --tag" >&2
        echo "       or move the existing CSV(s) out of the way." >&2
        exit 1
    fi
done

# Resolve the bench binary relative to the script's worktree root.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
worktree_root=$(cd "$script_dir/.." && pwd)
BENCH_BIN="$worktree_root/build/src/cuda/benchmarks/benchmark_tpch_ir"
if [[ ! -x "$BENCH_BIN" ]]; then
    echo "error: bench binary not built at: $BENCH_BIN" >&2
    echo "       run: cmake --build build --target benchmark_tpch_ir -j" >&2
    exit 2
fi

# Conda activation. We source the init script first because `conda
# activate` is a shell function, not an executable — it isn't
# available in a non-interactive shell otherwise.
if [[ -n "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "error: conda not on PATH; cannot activate '$CONDA_ENV'" >&2
        exit 2
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

mkdir -p "$OUT_DIR"

echo "TPC-H bench:"
echo "  machine:   $MACHINE"
echo "  scale:     $SCALE"
echo "  tag:       $TAG"
echo "  date:      $DATE"
echo "  gpu:       $GPU"
echo "  data:      $TPCH_DATA"
echo "  e2e CSV:   $E2E_CSV"
echo "  steps CSV: $STEPS_CSV"
echo ""

# Run the bench. NVBench's `-d` selects the device (this is what the
# user asked for via --gpu). The `--csv` is the e2e per-cell output;
# the per-step breakdown comes out via the GAMAX_BENCH_STEP_CSV
# side-channel that the bench fixture writes to.
TPCH_DATA_PATH="$TPCH_DATA" \
GAMAX_BENCH_STEP_CSV="$STEPS_CSV" \
"$BENCH_BIN" \
    -d "$GPU" \
    -a "scale=$SCALE" \
    --min-samples 3 \
    --max-noise 5 \
    --timeout 8 \
    --quiet \
    --csv "$E2E_CSV"

echo ""
echo "Done."
echo "  e2e CSV:   $E2E_CSV"
echo "  steps CSV: $STEPS_CSV"
