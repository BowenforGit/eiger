#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Strong-scaling study of the ALPS distributed inner join on a SINGLE multi-GPU
# node (e.g. an 8x H200 box). The FK relation is held at a fixed TOTAL size while
# the GPU count varies (np 1/2/4/8) -- classic strong scaling -- and we sweep the
# PK:FK ratio and the local-join algorithm.
#
# For each GPU count it launches `mpirun -np <np>` of alps_dist_join_scaling_bench
# in STRONG mode with an absolute --total-fk, sweeping:
#   ratios x {shuffle, broadcast} x {hash, smj, phj}
# and appends rows to ONE CSV (gpus,scaling,strategy,join_algo,ratio,...,timings).
# Per-rank FK = total_fk / np; PK = FK / ratio. Broadcast is auto-skipped by the
# bench when total_pk > fk_per_rank (replicating a PK bigger than the per-GPU FK
# shard is pointless), so for FK:PK=ratio broadcast only runs at np <= ratio.
#
# Runs under UserSpaceMM (one up-front slab; GAMAX_MEM_RESERVE_BYTES leaves NCCL
# its transport buffers) -- REQUIRED: under cudaMallocAsync the per-iteration
# output allocations stall the timed region and inflate partition/join unequally
# across strategies, corrupting the comparison.
#
# Required:
#   --build <dir>      build dir containing src/cuda/alps/alps_dist_join_scaling_bench
# Optional:
#   --machine <name>   output folder (default: h200)
#   --gpus "1 2 4 8"   GPU counts to sweep (default: "1 2 4 8")
#   --total-fk N       absolute total FK rows (default: 1073741824 = 4*2^28 = 2^30)
#   --ratios a,b,c     FK:PK ratios, PK<<FK (default: 128,64,32,16,8,4,2,1
#                      i.e. PK:FK = 1/128 .. 1)
#   --algos a,b,c      local-join algos (default: hash,smj,phj)
#   --strategy s       shuffle | broadcast | both (default: both)
#   --iters N          timed iters (default: 10)
#   --warmup N         warmup iters (default: 3)
#   --reserve-bytes N  UserSpaceMM device reserve left free for NCCL (default 16 GiB)
#   --csv-base <path>  CSV base dir (default: tools/results/data)
#   --csv <path>       exact CSV path (overrides csv-base/machine/date)
#   --append           add rows to an existing CSV instead of refusing (e.g. to
#                      fill in extra --gpus counts; the bench skips the header
#                      when the file is non-empty). Total/ratios/algos must match
#                      the existing rows for the data to be comparable.
#   --mpirun <path>    mpirun (default: /usr/local/mpi/bin/mpirun)
#
# Output: $CSV_BASE/$MACHINE/alps/<YYYYMMDD>/dist_join_strong_scaling.csv
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD=""
MACHINE="h200"
GPUS="1 2 4 8"
TOTAL_FK=1073741824        # 4 * 2^28 = 2^30
RATIOS="128,64,32,16,8,4,2,1"
ALGOS="hash,smj,phj"
STRATEGY="both"
ITERS=10
WARMUP=3
RESERVE_BYTES=16000000000  # 16 GB left free for NCCL transport buffers
CSV_BASE="${REPO_ROOT}/tools/results/data"
CSV=""
APPEND=0
MPIRUN="/usr/local/mpi/bin/mpirun"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)         BUILD="$2"; shift 2;;
    --machine)       MACHINE="$2"; shift 2;;
    --gpus)          GPUS="$2"; shift 2;;
    --total-fk)      TOTAL_FK="$2"; shift 2;;
    --ratios)        RATIOS="$2"; shift 2;;
    --algos)         ALGOS="$2"; shift 2;;
    --strategy)      STRATEGY="$2"; shift 2;;
    --iters)         ITERS="$2"; shift 2;;
    --warmup)        WARMUP="$2"; shift 2;;
    --reserve-bytes) RESERVE_BYTES="$2"; shift 2;;
    --csv-base)      CSV_BASE="$2"; shift 2;;
    --csv)           CSV="$2"; shift 2;;
    --append)        APPEND=1; shift;;
    --mpirun)        MPIRUN="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$BUILD" ]]; then echo "--build <dir> is required" >&2; exit 1; fi
BENCH="${BUILD}/src/cuda/alps/alps_dist_join_scaling_bench"
if [[ ! -x "$BENCH" ]]; then echo "bench not found/executable: $BENCH" >&2; exit 1; fi

if [[ -z "$CSV" ]]; then
  DATE="$(date +%Y%m%d)"
  OUTDIR="${CSV_BASE}/${MACHINE}/alps/${DATE}"
  mkdir -p "$OUTDIR"
  CSV="${OUTDIR}/dist_join_strong_scaling.csv"
fi
if [[ -e "$CSV" && "$APPEND" != "1" ]]; then
  echo "CSV already exists, not overwriting: $CSV" >&2
  echo "(delete it, pass --csv <other path>, or pass --append to add rows)" >&2
  exit 1
fi
if [[ "$APPEND" == "1" ]]; then
  if [[ -e "$CSV" ]]; then
    echo "APPEND mode: adding rows to existing $CSV" >&2
  else
    echo "APPEND mode: $CSV does not exist yet, creating it" >&2
  fi
fi
mkdir -p "$(dirname "$CSV")"

# Run as root inside the container needs Open MPI's explicit opt-in.
ROOT_FLAGS=()
if [[ "$(id -u)" == "0" ]]; then ROOT_FLAGS=(--allow-run-as-root); fi

export ALPS_USE_CUDA_MM=0
export GAMAX_MEM_RESERVE_BYTES="${RESERVE_BYTES}"

echo "CSV:       $CSV"
echo "bench:     $BENCH"
echo "gpus:      $GPUS"
echo "total FK:  $TOTAL_FK rows  | ratios $RATIOS (FK:PK) | algos $ALGOS | strategy $STRATEGY"
echo "iters $ITERS warmup $WARMUP | UserSpaceMM reserve ${RESERVE_BYTES}B for NCCL"

run() {  # np
  local np="$1"
  echo "=== np=$np  (strong, total FK=$TOTAL_FK) ==="
  "$MPIRUN" -np "$np" --bind-to none --oversubscribe "${ROOT_FLAGS[@]}" \
    -x PATH -x LD_LIBRARY_PATH -x ALPS_USE_CUDA_MM -x GAMAX_MEM_RESERVE_BYTES \
    "$BENCH" --scaling strong --strategy "$STRATEGY" \
    --total-fk "$TOTAL_FK" --ratios "$RATIOS" --algos "$ALGOS" \
    --iters "$ITERS" --warmup "$WARMUP" --out "$CSV"
}

for np in $GPUS; do
  run "$np"
done

echo "Done -> $CSV"
echo "Plot:  python tools/visualize/plot_alps_dist_join_scaling.py $CSV"
