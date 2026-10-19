#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Internal per-rank launcher used by profile_distributed_tpch_nsys.sh.

set -euo pipefail

profile_ranks=$1
output_prefix=$2
shift 2

rank=${OMPI_COMM_WORLD_RANK:?must run under Open MPI}
selected=0
if [[ "$profile_ranks" == all || ",${profile_ranks}," == *",${rank},"* ]]; then
    selected=1
fi
if [[ -n "${NSYS_PROFILE_HOST:-}" &&
      "$(hostname)" != "${NSYS_PROFILE_HOST}" ]]; then
    selected=0
fi

if (( selected == 0 )); then
    exec "$@"
fi

nsys_args=(profile \
    --trace=cuda,nvtx,mpi \
    --mpi-impl=openmpi \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=nvtx \
    --nvtx-capture=tpch.profile@gamax.tpch \
    --capture-range-end=stop \
    --force-overwrite=false \
    --output "$output_prefix")

gpu_metrics_devices=${NSYS_GPU_METRICS_DEVICES:-none}
gpu_metrics_rank=${NSYS_GPU_METRICS_RANK:-all}
if [[ "$gpu_metrics_rank" != all && "$rank" != "$gpu_metrics_rank" ]]; then
    gpu_metrics_devices=none
fi
if [[ "$gpu_metrics_devices" == local-rank ]]; then
    gpu_metrics_devices=${OMPI_COMM_WORLD_LOCAL_RANK:?missing MPI local rank}
fi
if [[ "$gpu_metrics_devices" != none ]]; then
    nsys_args+=(
        --gpu-metrics-devices="$gpu_metrics_devices"
        --gpu-metrics-frequency="${NSYS_GPU_METRICS_FREQUENCY:-10000}")
    if [[ -n "${NSYS_GPU_METRICS_SET:-}" ]]; then
        nsys_args+=(--gpu-metrics-set="${NSYS_GPU_METRICS_SET}")
    fi
fi
if [[ "${NSYS_NIC_METRICS:-false}" == true ]]; then
    nsys_args+=(--nic-metrics=true)
fi

exec nsys "${nsys_args[@]}" "$@"
