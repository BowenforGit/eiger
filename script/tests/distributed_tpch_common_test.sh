#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.


set -euo pipefail

source "$1"

fake_cuda_probe() {
    if [[ "${1:-}" != "--print-visible-device-count" ]]; then
        return 2
    fi
    case "${CUDA_VISIBLE_DEVICES-__unset__}" in
        __unset__) printf '4\n' ;;
        "") printf '0\n' ;;
        0) printf '1\n' ;;
        0,2) printf '2\n' ;;
        0,99|0,-1,2|0,,1) printf '1\n' ;;
        0,0) return 2 ;;
        *) return 2 ;;
    esac
}

expect_count() {
    local visible=$1
    local expected=$2
    local actual
    CUDA_VISIBLE_DEVICES=$visible actual=$(
        distributed_tpch_visible_gpu_count fake_cuda_probe)
    if [[ "$actual" != "$expected" ]]; then
        echo "CUDA_VISIBLE_DEVICES='$visible': expected $expected, got $actual" >&2
        exit 1
    fi
}

expect_count "" 0
expect_count "0" 1
expect_count "0,2" 2
expect_count "0,99" 1
expect_count "0,-1,2" 1
expect_count "0,,1" 1

if CUDA_VISIBLE_DEVICES="0,0" \
    distributed_tpch_visible_gpu_count fake_cuda_probe >/dev/null 2>&1; then
    echo "CUDA runtime probe failure was accepted" >&2
    exit 1
fi
if distributed_tpch_visible_gpu_count >/dev/null 2>&1; then
    echo "missing CUDA runtime probe was accepted" >&2
    exit 1
fi
