#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.


# Print the number of GPUs visible to a local CUDA process. The probe executable
# must implement --print-visible-device-count using cudaGetDeviceCount so CUDA,
# rather than shell token parsing, decides where the visible prefix ends.
distributed_tpch_visible_gpu_count() {
    if [[ $# -ne 1 || -z "$1" ]]; then
        echo "error: a CUDA runtime probe executable is required" >&2
        return 2
    fi
    local count
    if ! count=$("$1" --print-visible-device-count); then
        return 2
    fi
    if [[ ! "$count" =~ ^[0-9]+$ ]]; then
        echo "error: CUDA runtime probe returned an invalid count: $count" >&2
        return 2
    fi
    printf '%s\n' "$count"
}

# Make NCCL CTA policy deterministic on every MPI rank. Current CTA variables
# are forwarded only when the launcher received nonempty values. Deprecated
# channel/ring aliases are always removed because NCCL combines them with CTA
# bounds. The returned command prefix also removes unset CTA variables from a
# remote rank's native environment.
distributed_tpch_configure_nccl_cta_environment() {
    if [[ $# -ne 2 ]]; then
        echo "error: MPI-array and rank-prefix-array names are required" >&2
        return 2
    fi
    local -n mpi_array=$1
    local -n rank_prefix_array=$2
    local variable value
    rank_prefix_array=(env)
    for variable in NCCL_MIN_CTAS NCCL_MAX_CTAS; do
        value=${!variable-}
        if [[ -n "$value" ]]; then
            mpi_array+=(-x "$variable")
        else
            rank_prefix_array+=(-u "$variable")
        fi
    done
    for variable in \
        NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS \
        NCCL_MIN_NRINGS NCCL_MAX_NRINGS; do
        rank_prefix_array+=(-u "$variable")
    done
}
