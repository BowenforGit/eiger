#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Evaluate stock and synthesized distributed TPC-H plans independently.
# Each preset gets clean timing, five-sample step/phase, memory, and optional
# Nsight Systems passes in separate MPI invocations.

set -uo pipefail

scale=""
ranks=""
physical_plan_dir=""
output_dir=""
queries="1-22"
gpu_metrics_devices="all"
skip_profiles=0
failures=0

usage() {
    cat <<'EOF'
usage: compare_distributed_tpch_plans.sh --scale N --physical-plan-dir PATH [options]

Options:
  --ranks N                MPI ranks (default: visible local GPUs)
  --queries SPEC           query IDs/ranges (default: 1-22)
  --output-dir PATH        evaluation artifact directory
  --gpu-metrics-devices SPEC
                           NSYS GPU IDs to sample (default: all)
  --skip-profiles          omit Nsight Systems collection
  --help                   show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale) scale=$2; shift 2 ;;
        --ranks) ranks=$2; shift 2 ;;
        --queries) queries=$2; shift 2 ;;
        --physical-plan-dir) physical_plan_dir=$2; shift 2 ;;
        --output-dir) output_dir=$2; shift 2 ;;
        --gpu-metrics-devices) gpu_metrics_devices=$2; shift 2 ;;
        --skip-profiles) skip_profiles=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ ! "$scale" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --scale must be positive" >&2
    exit 2
fi
if [[ -z "$physical_plan_dir" || ! -d "$physical_plan_dir" ]]; then
    echo "error: --physical-plan-dir must exist" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
root=$(cd "$script_dir/.." && pwd)
if [[ -z "$output_dir" ]]; then
    output_dir="$root/results/distributed_tpch_plan_comparison/$(date -u +%Y%m%dT%H%M%SZ)/sf$scale"
fi
mkdir -p "$output_dir"
rank_args=()
if [[ -n "$ranks" ]]; then rank_args=(--ranks "$ranks"); fi

for preset in stock synthesized; do
    target="$output_dir/$preset/timing"
    plan_args=()
    if [[ "$preset" == synthesized ]]; then
        plan_args=(--physical-plan-dir "$physical_plan_dir")
    fi
    if ! "$script_dir/run_distributed_tpch_bench.sh" \
        --scale "$scale" --queries "$queries" \
        --output-dir "$target" "${rank_args[@]}" "${plan_args[@]}" \
        -- --iterations 20 --timing-only --eq-join legacy; then
        echo "error: $preset timing evaluation failed" >&2
        failures=$((failures + 1))
    fi

    breakdown_target="$output_dir/$preset/breakdown"
    if ! "$script_dir/run_distributed_tpch_bench.sh" \
        --scale "$scale" --queries "$queries" \
        --output-dir "$breakdown_target" \
        "${rank_args[@]}" "${plan_args[@]}" \
        -- --steps-only --step-iterations 5 --eq-join legacy; then
        echo "error: $preset step/phase evaluation failed" >&2
        failures=$((failures + 1))
    fi

    memory_target="$output_dir/$preset/memory"
    mkdir -p "$memory_target"
    if ! "$script_dir/run_distributed_tpch_bench.sh" \
        --scale "$scale" --queries "$queries" \
        --output-dir "$memory_target" "${rank_args[@]}" "${plan_args[@]}" \
        -- --memory-profile --memory-csv "$memory_target/memory.csv" \
        --eq-join legacy; then
        echo "error: $preset memory evaluation failed" >&2
        failures=$((failures + 1))
    fi

    if (( skip_profiles == 0 )); then
        profile_target="$output_dir/$preset/profiles"
        profile_plan_args=()
        if [[ "$preset" == synthesized ]]; then
            profile_plan_args=(--physical-plan-dir "$physical_plan_dir")
        fi
        if ! "$script_dir/collect_distributed_tpch_nsys.sh" \
            --scales "$scale" --queries "$queries" \
            --output-dir "$profile_target" "${rank_args[@]}" \
            --gpu-metrics-devices "$gpu_metrics_devices" \
            "${profile_plan_args[@]}" -- --eq-join legacy; then
            echo "error: $preset utilization profiling failed" >&2
            failures=$((failures + 1))
        fi
    fi
done

echo "Evaluation artifacts: $output_dir"
if (( failures > 0 )); then
    echo "Evaluation completed with $failures failed stage(s)" >&2
    exit 1
fi
