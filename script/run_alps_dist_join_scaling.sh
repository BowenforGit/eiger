#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Run the ALPS shuffle-vs-broadcast distributed-join scaling sweep across node
# counts and write a single CSV for plotting.
#
# For each GPU count in --gpus it launches a multi-node mpirun (hostfile + rsh on
# the cluster ssh port) of alps_dist_join_scaling_bench in BOTH weak and strong
# scaling modes; the bench sweeps ratios x FK-size x {shuffle,broadcast} and
# appends rows (gpus,scaling,strategy,ratio,size,partition/comm/join,...) to the CSV.
#
# Runs under UserSpaceMM (one up-front slab; GAMAX_MEM_RESERVE_BYTES leaves NCCL
# room) — REQUIRED, else cudaMallocAsync stalls the timed region and inflates
# partition/join unequally across strategies (corrupts the comparison).
#
# Required:
#   --build <dir>      build dir containing src/cuda/alps/alps_dist_join_scaling_bench
# Optional:
#   --machine <name>   output folder (default: gb200)
#   --gpus "4 8 12"    GPU counts (default: "4 8 12"); -N 4 ranks/node
#   --ratios a,b,c     FK:PK ratios, PK<<FK (default: 4,16,64)
#   --weak-exp  "MIN MAX"   weak per-rank FK exponents (default: "22 25")
#   --strong-exp "MIN MAX"  strong total-FK exponents (default: "25 27")
#   --iters N          timed iters (default: 10)
#   --warmup N         warmup iters (default: 3)
#   --reserve-bytes N  UserSpaceMM device reserve left free for NCCL (default 16 GiB)
#   --hostfile <path>  MPI hostfile (default: ~/hosts.ncc)
#   --port N           ssh port for rsh (default: 2222)
#   --iface <name>     NCCL_SOCKET_IFNAME (default: eth)
#   --csv-base <path>  CSV base (default: tools/results/data)
#   --mpirun <path>    mpirun (default: /usr/local/mpi/bin/mpirun)
#
# Output: $CSV_BASE/$MACHINE/alps/<YYYYMMDD>/dist_join_scaling.csv
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD=""
MACHINE="gb200"
GPUS="4 8 12"
RATIOS="4,16,64"
WEAK_EXP="22 25"
STRONG_EXP="25 27"
ITERS=10
WARMUP=3
RESERVE_BYTES=16000000000
HOSTFILE="${HOME}/hosts.ncc"
PORT=2222
IFACE="eth"
CSV_BASE="${REPO_ROOT}/tools/results/data"
MPIRUN="/usr/local/mpi/bin/mpirun"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)         BUILD="$2"; shift 2;;
    --machine)       MACHINE="$2"; shift 2;;
    --gpus)          GPUS="$2"; shift 2;;
    --ratios)        RATIOS="$2"; shift 2;;
    --weak-exp)      WEAK_EXP="$2"; shift 2;;
    --strong-exp)    STRONG_EXP="$2"; shift 2;;
    --iters)         ITERS="$2"; shift 2;;
    --warmup)        WARMUP="$2"; shift 2;;
    --reserve-bytes) RESERVE_BYTES="$2"; shift 2;;
    --hostfile)      HOSTFILE="$2"; shift 2;;
    --port)          PORT="$2"; shift 2;;
    --iface)         IFACE="$2"; shift 2;;
    --csv-base)      CSV_BASE="$2"; shift 2;;
    --mpirun)        MPIRUN="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$BUILD" ]]; then echo "--build <dir> is required" >&2; exit 1; fi
BENCH="${BUILD}/src/cuda/alps/alps_dist_join_scaling_bench"
if [[ ! -x "$BENCH" ]]; then echo "bench not found/executable: $BENCH" >&2; exit 1; fi

DATE="$(date +%Y%m%d)"
OUTDIR="${CSV_BASE}/${MACHINE}/alps/${DATE}"
mkdir -p "$OUTDIR"
CSV="${OUTDIR}/dist_join_scaling.csv"
if [[ -e "$CSV" ]]; then
  echo "CSV already exists, not overwriting: $CSV" >&2
  exit 1
fi

export ALPS_USE_CUDA_MM=0
export GAMAX_MEM_RESERVE_BYTES="${RESERVE_BYTES}"
export NCCL_SOCKET_IFNAME="${IFACE}"

read -r WMIN WMAX <<< "$WEAK_EXP"
read -r SMIN SMAX <<< "$STRONG_EXP"

echo "CSV: $CSV"
echo "gpus: $GPUS | ratios $RATIOS | weak 2^${WMIN}..2^${WMAX} | strong total 2^${SMIN}..2^${SMAX}"
echo "Allocator: UserSpaceMM (reserve ${RESERVE_BYTES}B for NCCL)"

run() {  # scaling emin emax np
  local scaling="$1" emin="$2" emax="$3" np="$4"
  echo "=== np=$np scaling=$scaling fk-exp ${emin}..${emax} ==="
  "$MPIRUN" --hostfile "$HOSTFILE" -np "$np" -N 4 \
    --mca plm_rsh_args "-p ${PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    -x PATH -x LD_LIBRARY_PATH -x NCCL_SOCKET_IFNAME -x ALPS_USE_CUDA_MM -x GAMAX_MEM_RESERVE_BYTES \
    "$BENCH" --scaling "$scaling" --strategy both --ratios "$RATIOS" \
    --fk-exp-min "$emin" --fk-exp-max "$emax" --iters "$ITERS" --warmup "$WARMUP" --out "$CSV"
}

set +e
for np in $GPUS; do
  run weak   "$WMIN" "$WMAX" "$np"
  run strong "$SMIN" "$SMAX" "$np"
done
set -e

echo "Done -> $CSV"
