#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Collect one local-process-tree NSYS report per distributed TPC-H query.

set -uo pipefail

scales="100,300"
queries="1-22"
output_dir=""
physical_plan_dir=""
ranks=""
gpu_metrics_devices="all"
gpu_metrics_frequency="10000"
fail_fast=0
benchmark_args=()

usage() {
    cat <<'EOF'
usage: collect_distributed_tpch_nsys.sh [options] [-- benchmark args]

Options:
  --scales SPEC          comma-separated scale factors (default: 100,300)
  --queries SPEC         comma-separated IDs/ranges (default: 1-22)
  --output-dir PATH      artifact root
  --physical-plan-dir PATH use PATH/qNN.json for each query
  --ranks N              MPI ranks (default: visible local GPUs)
  --gpu-metrics-devices SPEC
                         NSYS GPU IDs to sample (default: all)
  --gpu-metrics-frequency HZ
                         metric sampling frequency (default: 10000)
  --fail-fast            stop after the first failed query
  --help                 show this help

Existing nonempty qNN.nsys-rep files are skipped, making collection resumable.
Each report contains the complete local MPI process tree and all visible GPU
metric streams. Arguments after `--` are forwarded to the benchmark.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scales) scales=$2; shift 2 ;;
        --queries) queries=$2; shift 2 ;;
        --output-dir) output_dir=$2; shift 2 ;;
        --physical-plan-dir) physical_plan_dir=$2; shift 2 ;;
        --ranks) ranks=$2; shift 2 ;;
        --gpu-metrics-devices) gpu_metrics_devices=$2; shift 2 ;;
        --gpu-metrics-frequency) gpu_metrics_frequency=$2; shift 2 ;;
        --fail-fast) fail_fast=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; benchmark_args=("$@"); break ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$gpu_metrics_frequency" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --gpu-metrics-frequency must be a positive integer" >&2
    exit 2
fi
if [[ -n "$ranks" && ! "$ranks" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --ranks must be a positive integer" >&2
    exit 2
fi

declare -A selected_queries=()
IFS=',' read -r -a query_chunks <<< "$queries"
for chunk in "${query_chunks[@]}"; do
    if [[ "$chunk" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        first=${BASH_REMATCH[1]}
        last=${BASH_REMATCH[2]}
        if (( first > last )); then
            echo "error: descending query range: $chunk" >&2
            exit 2
        fi
        for ((query = first; query <= last; ++query)); do
            selected_queries[$query]=1
        done
    elif [[ "$chunk" =~ ^[0-9]+$ ]]; then
        selected_queries[$chunk]=1
    else
        echo "error: invalid query specification: $chunk" >&2
        exit 2
    fi
done
query_ids=()
for ((query = 1; query <= 22; ++query)); do
    if [[ -n ${selected_queries[$query]+x} ]]; then query_ids+=("$query"); fi
done
if (( ${#query_ids[@]} != ${#selected_queries[@]} || ${#query_ids[@]} == 0 )); then
    echo "error: query IDs must be in [1, 22]" >&2
    exit 2
fi

scale_ids=()
IFS=',' read -r -a scale_chunks <<< "$scales"
for scale in "${scale_chunks[@]}"; do
    if [[ ! "$scale" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: scale factors must be positive integers: $scale" >&2
        exit 2
    fi
    scale_ids+=("$scale")
done
if (( ${#scale_ids[@]} == 0 )); then
    echo "error: at least one scale factor is required" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
root=$(cd "$script_dir/.." && pwd)
if [[ -z "$output_dir" ]]; then
    output_dir="$root/results/distributed_tpch_nsys/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$output_dir"

metadata="$output_dir/system_metadata.txt"
if [[ ! -s "$metadata" ]]; then
    {
        date -u +"collected_utc=%Y-%m-%dT%H:%M:%SZ"
        printf 'git_commit='
        git -C "$root" rev-parse HEAD
        if [[ -z $(git -C "$root" status --porcelain) ]]; then
            echo "git_worktree=clean"
        else
            echo "git_worktree=dirty"
        fi
        hostname
        nsys --version
        nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
            --format=csv,noheader
        nvidia-smi topo -m
        nvidia-smi nvlink --status
    } > "$metadata" 2>&1
fi

status_csv="$output_dir/status.csv"
if [[ ! -s "$status_csv" ]]; then
    printf 'timestamp_utc,query,scale,status,exit_code,elapsed_seconds,report,log\n' \
        > "$status_csv"
fi

echo "Distributed TPC-H NSYS collection"
echo "  scales:     ${scale_ids[*]}"
echo "  queries:    ${query_ids[*]}"
echo "  ranks:      ${ranks:-visible local GPUs}"
echo "  devices:    $gpu_metrics_devices"
echo "  frequency:  $gpu_metrics_frequency Hz"
echo "  output:     $output_dir"

failures=0
passes=0
skips=0
for scale in "${scale_ids[@]}"; do
    scale_dir="$output_dir/sf$scale"
    mkdir -p "$scale_dir/logs"
    for query in "${query_ids[@]}"; do
        query_name="q$(printf '%02d' "$query")"
        report="$scale_dir/$query_name.nsys-rep"
        log="$scale_dir/logs/$query_name.log"
        if [[ -s "$report" ]]; then
            echo "== Q$query SF$scale: existing report, skipped =="
            skips=$((skips + 1))
            continue
        fi

        echo "== Q$query SF$scale =="
        command=("$script_dir/profile_distributed_tpch_nsys.sh"
            --query "$query" --scale "$scale" --output-dir "$scale_dir"
            --single-report --gpu-metrics-devices "$gpu_metrics_devices"
            --gpu-metrics-frequency "$gpu_metrics_frequency")
        if [[ -n "$ranks" ]]; then command+=(--ranks "$ranks"); fi
        per_query_args=("${benchmark_args[@]}")
        if [[ -n "$physical_plan_dir" ]]; then
            config="$physical_plan_dir/$query_name.json"
            if [[ ! -s "$config" ]]; then
                echo "error: missing physical-plan config: $config" >&2
                exit 2
            fi
            per_query_args+=(--physical-plan-config "$config")
        fi
        if (( ${#per_query_args[@]} > 0 )); then
            command+=(-- "${per_query_args[@]}")
        fi

        start_seconds=$SECONDS
        "${command[@]}" 2>&1 | tee "$log"
        exit_code=${PIPESTATUS[0]}
        elapsed=$((SECONDS - start_seconds))
        if (( exit_code == 0 )) && [[ -s "$report" ]]; then
            status=pass
            passes=$((passes + 1))
        else
            status=fail
            failures=$((failures + 1))
            if (( exit_code == 0 )); then exit_code=1; fi
        fi
        printf '%s,Q%d,%d,%s,%d,%d,"%s","%s"\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$query" "$scale" \
            "$status" "$exit_code" "$elapsed" "$report" "$log" \
            >> "$status_csv"
        if [[ "$status" == fail && "$fail_fast" == 1 ]]; then
            echo "Stopped after failure: Q$query SF$scale" >&2
            exit 1
        fi
    done
done

echo "Completed: $passes collected, $skips skipped, $failures failed"
echo "Artifacts: $output_dir"
(( failures == 0 ))
