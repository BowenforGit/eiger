#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Internal launcher: profile MPI rank 0 with NCU and run all peers normally.

set -euo pipefail

output_dir=$1
query=$2
shift 2

rank=${OMPI_COMM_WORLD_RANK:?must run under Open MPI}
if [[ "$rank" != 0 ]]; then
    exec "$@"
fi

host=$(hostname)
report_prefix="$output_dir/q$(printf '%02d' "$query")_${host}_rank0"
section_file=${ALPS_TPCH_NCU_SECTION_FILE:?missing NCU section file}
kernel_name=${ALPS_TPCH_NCU_KERNEL_NAME:?missing NCU kernel filter}
exec "${NCU_BIN:-ncu}" \
    --nvtx \
    --nvtx-push-pop-scope process \
    --nvtx-include 'gamax.tpch@tpch.profile/' \
    --kernel-name-base demangled \
    --kernel-name "$kernel_name" \
    --section-folder "$(dirname -- "$section_file")" \
    --section AlpsTpchMetrics \
    --disable-extra-suffixes \
    --clock-control none \
    --target-processes application-only \
    --force-overwrite \
    --export "$report_prefix" \
    "$@"
