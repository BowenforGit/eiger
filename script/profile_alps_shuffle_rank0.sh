#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Profile MPI rank 0 only; all other ranks run the benchmark normally.
set -euo pipefail

output=$1
shift
if [[ ${1:-} == -- ]]; then shift; fi
rank=${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}
if [[ $rank == 0 ]]; then
    exec nsys profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        --output "$output" \
        "$@"
fi
exec "$@"
