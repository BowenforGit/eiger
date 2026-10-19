#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 QUERY_ID SCALE_FACTOR [runner args...]" >&2
    exit 2
fi

query_id=$1
scale_factor=$2
shift 2

root=${ALPS_TPCH_ROOT:-/storage/alps-tpch}
binary="$root/build/src/cuda/tpch/run_tpch_ir_mpi"
library_path="$root/build/src/cuda/tpch:$root/build/src/cuda/alps"
library_path+="::$root/build/src/cuda:$root/build/_deps/rmm-build"
library_path+="::$root/build/_deps/rapids_logger-build"
library_path+="::/storage/conda/envs/dev/lib"
library_path+="::/usr/local/cuda/targets/sbsa-linux/lib:/opt/hpcx/ompi/lib"
library_path=${library_path//::/:}

export TPCH_DATA_PATH=${TPCH_DATA_PATH:-/storage/tpch}
export LD_LIBRARY_PATH="$library_path:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth}
export NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-1}
export GAMAX_RMM_RESOURCE=${GAMAX_RMM_RESOURCE:-arena}
export GAMAX_MEM_RESERVE_BYTES=${GAMAX_MEM_RESERVE_BYTES:-12000000000}

exec /usr/local/mpi/bin/mpirun \
    --host gb200-015:4,gb200-005:4 \
    -np 8 \
    --map-by ppr:4:node \
    --mca plm_rsh_args \
        "-p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    -x PATH \
    -x TPCH_DATA_PATH \
    -x LD_LIBRARY_PATH \
    -x NCCL_SOCKET_IFNAME \
    -x NCCL_MNNVL_ENABLE \
    -x GAMAX_RMM_RESOURCE \
    -x GAMAX_MEM_RESERVE_BYTES \
    "$binary" "$query_id" --scale "$scale_factor" "$@"
