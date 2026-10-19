#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Run the ALPS shuffle-only (all-to-all) bandwidth microbenchmark across GPU
# counts and write a single CSV for plotting with
# tools/visualize/plot_alps_shuffle.py.
#
# For each g in --gpus, launches one `mpirun -np g alps_shuffle_bench` that
# sweeps the per-peer message size (bytes each GPU sends to each peer) and
# appends rows to the CSV. Specifying the message size (not a per-GPU total split
# g ways) keeps every g sampled at the same x points and the chunks 16B-aligned.
#
# Allocator: default CudaMM (send/recv allocated once per size, comm uses
# pre-allocated buffers, so there is no per-iteration allocation and NCCL keeps
# its transport memory). No ALPS_USE_CUDA_MM override.
#
# Required:
#   --build <dir>     build dir containing src/cuda/alps/alps_shuffle_bench
# Optional:
#   --machine <name>  output folder (default: h200)
#   --gpus "1 2 4 8"  GPU counts (default: "1 2 4 8")
#   --variants "..."  backend variants (default: "nccl p2p_sm p2p_ce");
#                     each of nccl / p2p_sm / p2p_ce -> one bench invocation per g
#   --msg-exp-min N   min per-peer message size exponent, bytes (default: 16 = 64 KiB)
#   --msg-exp-max N   max per-peer message size exponent, bytes (default: 30 = 1 GiB)
#   --iters N         timed iterations (default: 20)
#   --warmup N        warmup iterations (default: 5)
#   --csv-base <path> CSV output base (default: tools/results/data)
#   --mpirun <path>   mpirun launcher (default: /usr/local/mpi/bin/mpirun)
#
# Output: $CSV_BASE/$MACHINE/alps/<YYYYMMDD>/shuffle.csv
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD=""
MACHINE="h200"
GPUS="1 2 4 8"
VARIANTS="nccl p2p_sm p2p_ce"
MSG_EXP_MIN=16
MSG_EXP_MAX=30
ITERS=20
WARMUP=5
CSV_BASE="${REPO_ROOT}/tools/results/data"
MPIRUN="/usr/local/mpi/bin/mpirun"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)       BUILD="$2"; shift 2;;
    --machine)     MACHINE="$2"; shift 2;;
    --gpus)        GPUS="$2"; shift 2;;
    --variants)    VARIANTS="$2"; shift 2;;
    --msg-exp-min) MSG_EXP_MIN="$2"; shift 2;;
    --msg-exp-max) MSG_EXP_MAX="$2"; shift 2;;
    --iters)       ITERS="$2"; shift 2;;
    --warmup)      WARMUP="$2"; shift 2;;
    --csv-base)    CSV_BASE="$2"; shift 2;;
    --mpirun)      MPIRUN="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$BUILD" ]]; then echo "--build <dir> is required" >&2; exit 1; fi
BENCH="${BUILD}/src/cuda/alps/alps_shuffle_bench"
if [[ ! -x "$BENCH" ]]; then echo "bench not found/executable: $BENCH" >&2; exit 1; fi

DATE="$(date +%Y%m%d)"
OUTDIR="${CSV_BASE}/${MACHINE}/alps/${DATE}"
mkdir -p "$OUTDIR"
CSV="${OUTDIR}/shuffle.csv"
if [[ -e "$CSV" ]]; then
  echo "CSV already exists, not overwriting: $CSV" >&2
  exit 1
fi

echo "Writing CSV to: $CSV"
echo "GPU counts: $GPUS | variants: $VARIANTS | per-peer msg 2^${MSG_EXP_MIN}..2^${MSG_EXP_MAX} B | iters=$ITERS warmup=$WARMUP"

# Map a variant label to alps_shuffle_bench backend flags.
variant_flags() {
  case "$1" in
    nccl)   echo "--backend nccl";;
    p2p_sm) echo "--backend p2p --copy sm";;
    p2p_ce) echo "--backend p2p --copy ce";;
    *) echo "unknown variant: $1 (use nccl|p2p_sm|p2p_ce)" >&2; exit 1;;
  esac
}

set +e
for np in $GPUS; do
  for v in $VARIANTS; do
    echo "=== mpirun -np $np  [$v] ==="
    "$MPIRUN" -np "$np" "$BENCH" \
      $(variant_flags "$v") \
      --msg-exp-min "$MSG_EXP_MIN" --msg-exp-max "$MSG_EXP_MAX" \
      --iters "$ITERS" --warmup "$WARMUP" --out "$CSV"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "WARNING: mpirun -np $np [$v] exited with $rc; continuing" >&2
    fi
  done
done
set -e

echo "Done. Plot with:"
echo "  python tools/visualize/plot_alps_shuffle.py $CSV"
