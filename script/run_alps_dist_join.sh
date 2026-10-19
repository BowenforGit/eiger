#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Run the ALPS distributed-join throughput sweep across GPU counts and write a
# single CSV for plotting with tools/visualize/plot_alps_dist_join.py.
#
# For each GPU count in --gpus, launches one `mpirun -np <n> alps_dist_join_bench`
# that internally sweeps the per-rank size series x the 5 local-join configs and
# appends rows to the CSV.
#
# Measurement runs under UserSpaceMM (one up-front slab cudaMalloc, then fast
# slab slices) to keep cudaMalloc/cudaFree out of the timing loop;
# GAMAX_MEM_RESERVE_BYTES leaves NCCL room for its transport buffers.
#
# Required:
#   --build <dir>     build directory containing src/cuda/alps/alps_dist_join_bench
# Optional:
#   --machine <name>  output folder name (default: h200)
#   --gpus "1 2 4 8"  GPU counts to sweep (default: "1 2 4 8")
#   --size-exp-min N  min per-rank size exponent (default: 20)
#   --size-exp-max N  max per-rank size exponent (default: 28)
#   --iters N         timed iterations (default: 10)
#   --warmup N        warmup iterations (default: 3)
#   --key-type i32|i64 (default: i32)
#   --csv-base <path> CSV output base (default: tools/results/data)
#   --reserve-bytes N UserSpaceMM device reserve for NCCL (default: 8589934592 = 8 GiB)
#   --mpirun <path>   mpirun launcher (default: /usr/local/mpi/bin/mpirun)
#
# Output: $CSV_BASE/$MACHINE/alps/<YYYYMMDD>/dist_join.csv
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD=""
MACHINE="h200"
GPUS="1 2 4 8"
SIZE_EXP_MIN=20
SIZE_EXP_MAX=28
ITERS=10
WARMUP=3
KEY_TYPE="i32"
CSV_BASE="${REPO_ROOT}/tools/results/data"
RESERVE_BYTES=8589934592
MPIRUN="/usr/local/mpi/bin/mpirun"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)         BUILD="$2"; shift 2;;
    --machine)       MACHINE="$2"; shift 2;;
    --gpus)          GPUS="$2"; shift 2;;
    --size-exp-min)  SIZE_EXP_MIN="$2"; shift 2;;
    --size-exp-max)  SIZE_EXP_MAX="$2"; shift 2;;
    --iters)         ITERS="$2"; shift 2;;
    --warmup)        WARMUP="$2"; shift 2;;
    --key-type)      KEY_TYPE="$2"; shift 2;;
    --csv-base)      CSV_BASE="$2"; shift 2;;
    --reserve-bytes) RESERVE_BYTES="$2"; shift 2;;
    --mpirun)        MPIRUN="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$BUILD" ]]; then echo "--build <dir> is required" >&2; exit 1; fi
BENCH="${BUILD}/src/cuda/alps/alps_dist_join_bench"
if [[ ! -x "$BENCH" ]]; then echo "bench not found/executable: $BENCH" >&2; exit 1; fi

DATE="$(date +%Y%m%d)"
OUTDIR="${CSV_BASE}/${MACHINE}/alps/${DATE}"
mkdir -p "$OUTDIR"
CSV="${OUTDIR}/dist_join.csv"
if [[ -e "$CSV" ]]; then
  echo "CSV already exists, not overwriting: $CSV" >&2
  echo "(remove it or pass a different --machine to start fresh)" >&2
  exit 1
fi

# UserSpaceMM with a reserve large enough for NCCL transport buffers.
export ALPS_USE_CUDA_MM=0
export GAMAX_MEM_RESERVE_BYTES="${RESERVE_BYTES}"

echo "Writing CSV to: $CSV"
echo "GPU counts: $GPUS | sizes 2^${SIZE_EXP_MIN}..2^${SIZE_EXP_MAX} | iters=$ITERS warmup=$WARMUP key=$KEY_TYPE"
echo "Allocator: UserSpaceMM (reserve=${RESERVE_BYTES} bytes)"

# Don't let one GPU-count failure abort the whole sweep.
set +e
for np in $GPUS; do
  echo "=== mpirun -np $np ==="
  "$MPIRUN" -np "$np" "$BENCH" \
    --size-exp-min "$SIZE_EXP_MIN" --size-exp-max "$SIZE_EXP_MAX" \
    --iters "$ITERS" --warmup "$WARMUP" --key-type "$KEY_TYPE" \
    --out "$CSV"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "WARNING: mpirun -np $np exited with $rc; continuing with remaining GPU counts" >&2
  fi
done
set -e

echo "Done. Plot with:"
echo "  python tools/visualize/plot_alps_dist_join.py $CSV"
