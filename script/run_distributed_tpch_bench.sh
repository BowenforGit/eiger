#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Run distributed TPC-H queries one MPI invocation at a time.
#
# Examples:
#   ./script/run_distributed_tpch_bench.sh --scale 100
#   ./script/run_distributed_tpch_bench.sh \
#       --scale 300 --queries 1,6,9-12 --output-dir /storage/results/sf300
#   ./script/run_distributed_tpch_bench.sh \
#       --scale 100 --hosts gb200-005:4,gb200-016:4 --ranks 8 \
#       --ranks-per-node 4 --ssh-port 2222
#
# Arguments following `--` are forwarded to benchmark_tpch_ir_mpi. By
# default the benchmark performs adaptive warmup, 10 end-to-end samples,
# five step-timing samples, and golden-result verification.

set -uo pipefail

scale=""
queries="1-22"
output_dir=""
physical_plan_dir=""
build_dir=""
tpch_data=""
ranks=""
ranks_per_node=""
hosts=""
ssh_port="2222"
fail_fast=0
benchmark_args=()

usage() {
    sed -n '2,/^set -uo pipefail/p' "$0" |
        sed -e 's/^# \{0,1\}//' -e '/^set -uo pipefail/d'
    cat <<'EOF'

Options:
  --scale N              required TPC-H scale factor
  --queries SPEC         comma-separated IDs/ranges (default: 1-22)
  --output-dir PATH      artifact directory
  --physical-plan-dir PATH use PATH/qNN.json for each query
  --build-dir PATH       CMake build tree (default: EIGER_BUILD_DIR or build-b200/build)
  --tpch-data PATH       root containing sf<N> directories (default: TPCH_DATA_PATH)
  --ranks N              MPI ranks (default: visible local GPUs; required with --hosts)
  --hosts LIST           Open MPI host list; omit for local execution
  --ranks-per-node N     add --map-by ppr:N:node
  --ssh-port N           remote SSH port (default: 2222)
  --fail-fast            stop after the first failed query
  --help                 show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale) scale=$2; shift 2 ;;
        --queries) queries=$2; shift 2 ;;
        --output-dir) output_dir=$2; shift 2 ;;
        --physical-plan-dir) physical_plan_dir=$2; shift 2 ;;
        --build-dir) build_dir=$2; shift 2 ;;
        --tpch-data) tpch_data=$2; shift 2 ;;
        --ranks) ranks=$2; shift 2 ;;
        --hosts) hosts=$2; shift 2 ;;
        --ranks-per-node) ranks_per_node=$2; shift 2 ;;
        --ssh-port) ssh_port=$2; shift 2 ;;
        --fail-fast) fail_fast=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; benchmark_args=("$@"); break ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$scale" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --scale must be a positive integer" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
root=$(cd "$script_dir/.." && pwd)
source "$script_dir/distributed_tpch_common.sh"
default_build=${build_dir:-${EIGER_BUILD_DIR:-$root/build-b200}}
if [[ -z "$build_dir" && -z ${EIGER_BUILD_DIR:-} && ! -d "$default_build" ]]; then
    default_build="$root/build"
fi
if [[ ! -d "$default_build" ]]; then
    echo "error: build directory does not exist: $default_build" >&2
    exit 2
fi
default_build=$(cd -- "$default_build" && pwd)
binary=${ALPS_TPCH_BENCH_BINARY:-$default_build/src/cuda/tpch/benchmark_tpch_ir_mpi}
device_count_binary=${ALPS_CUDA_VISIBLE_DEVICE_COUNT_BINARY:-$default_build/src/cuda/tpch/cuda_visible_device_count}
if [[ ! -x "$binary" ]]; then
    echo "error: benchmark binary is not executable: $binary" >&2
    exit 2
fi

if [[ -z "$ranks" ]]; then
    if [[ -n "$hosts" ]]; then
        echo "error: --ranks is required with --hosts" >&2
        exit 2
    fi
    if [[ ! -x "$device_count_binary" ]]; then
        echo "error: CUDA device-count probe is not executable: $device_count_binary" >&2
        exit 2
    fi
    if ! ranks=$(distributed_tpch_visible_gpu_count "$device_count_binary"); then
        exit 2
    fi
elif [[ -z "$hosts" ]]; then
    if ! visible_ranks=$(distributed_tpch_visible_gpu_count "$device_count_binary"); then
        exit 2
    fi
fi
if [[ ! "$ranks" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --ranks must be a positive integer" >&2
    exit 2
fi
if [[ -z "$hosts" && -n ${visible_ranks:-} ]] &&
   (( ranks > visible_ranks )); then
    echo "error: requested $ranks ranks but CUDA exposes only $visible_ranks GPUs" >&2
    exit 2
fi
if [[ -n "$ranks_per_node" && ! "$ranks_per_node" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --ranks-per-node must be a positive integer" >&2
    exit 2
fi

declare -A selected=()
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
            selected[$query]=1
        done
    elif [[ "$chunk" =~ ^[0-9]+$ ]]; then
        selected[$chunk]=1
    else
        echo "error: invalid query specification: $chunk" >&2
        exit 2
    fi
done
query_ids=()
for ((query = 1; query <= 22; ++query)); do
    if [[ -n ${selected[$query]+x} ]]; then query_ids+=("$query"); fi
done
if (( ${#query_ids[@]} != ${#selected[@]} || ${#query_ids[@]} == 0 )); then
    echo "error: query IDs must be in [1, 22]" >&2
    exit 2
fi

if [[ -z "$output_dir" ]]; then
    output_dir="$root/results/distributed_tpch/$(date -u +%Y%m%dT%H%M%SZ)/sf$scale"
fi
mkdir -p "$output_dir/logs"
e2e_csv="$output_dir/e2e.csv"
steps_csv="$output_dir/steps.csv"
phases_csv="$output_dir/phases.csv"
plans_file="$output_dir/plans.txt"
status_csv="$output_dir/status.csv"
if [[ ! -s "$status_csv" ]]; then
    printf 'timestamp_utc,query,scale,ranks,status,exit_code,log\n' > "$status_csv"
fi

library_path="$default_build/src/cuda/tpch:$default_build/src/cuda/alps"
library_path+=":$default_build/src/cuda:$default_build/_deps/rmm-build"
library_path+=":$default_build/_deps/rapids_logger-build"
library_path+=":/storage/conda/x86_64/envs/eiger-b200-build/lib"
library_path+=":/usr/local/cuda/targets/sbsa-linux/lib:/opt/hpcx/ompi/lib"
export TPCH_DATA_PATH=${tpch_data:-${TPCH_DATA_PATH:-/storage/tpch}}
export LD_LIBRARY_PATH="$library_path:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth}
export NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-1}
export GAMAX_RMM_RESOURCE=${GAMAX_RMM_RESOURCE:-arena}
export GAMAX_MEM_RESERVE_BYTES=${GAMAX_MEM_RESERVE_BYTES:-12000000000}

mpirun_binary=${MPIRUN:-/usr/local/mpi/bin/mpirun}
mpi=("$mpirun_binary" -np "$ranks")
if [[ -n "$hosts" ]]; then
    mpi+=(--host "$hosts" --mca plm_rsh_args
        "-p $ssh_port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")
fi
if [[ -n "$ranks_per_node" ]]; then
    mpi+=(--map-by "ppr:$ranks_per_node:node")
fi
mpi+=(
    -x PATH -x TPCH_DATA_PATH -x LD_LIBRARY_PATH
    -x NCCL_SOCKET_IFNAME -x NCCL_MNNVL_ENABLE
    -x GAMAX_RMM_RESOURCE -x GAMAX_MEM_RESERVE_BYTES)
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    mpi+=(-x CUDA_VISIBLE_DEVICES)
fi
rank_environment=()
distributed_tpch_configure_nccl_cta_environment mpi rank_environment

echo "Distributed TPC-H benchmark"
echo "  scale:      $scale"
echo "  queries:    ${query_ids[*]}"
echo "  ranks:      $ranks"
echo "  hosts:      ${hosts:-local}"
echo "  data:       $TPCH_DATA_PATH"
echo "  output:     $output_dir"

failures=0
passes=0
attempted=0
for query in "${query_ids[@]}"; do
    attempted=$((attempted + 1))
    log="$output_dir/logs/q$(printf '%02d' "$query").log"
    echo "== Q$query SF$scale =="
    command=(
        "${mpi[@]}" "${rank_environment[@]}" "$binary" "$query" --scale "$scale"
        --e2e-csv "$e2e_csv" --steps-csv "$steps_csv"
        --phases-csv "$phases_csv"
        --plan-file "$plans_file")
    if [[ -n "$physical_plan_dir" ]]; then
        config="$physical_plan_dir/q$(printf '%02d' "$query").json"
        if [[ ! -s "$config" ]]; then
            echo "error: missing physical-plan config: $config" >&2
            exit 2
        fi
        command+=(--physical-plan-config "$config")
    fi
    command+=("${benchmark_args[@]}")
    if "${command[@]}" 2>&1 | tee "$log"; then
        status=pass
        exit_code=0
        passes=$((passes + 1))
    else
        exit_code=$?
        status=fail
        failures=$((failures + 1))
    fi
    printf '%s,Q%d,%d,%d,%s,%d,"%s"\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$query" "$scale" "$ranks" \
        "$status" "$exit_code" "$log" >> "$status_csv"
    if [[ "$status" == fail && "$fail_fast" == 1 ]]; then break; fi
done

echo "Completed: $passes passed, $failures failed, $attempted attempted"
echo "Artifacts: $output_dir"
(( failures == 0 ))
