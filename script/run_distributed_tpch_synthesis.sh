#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Synthesize guarded per-query distributed TPC-H physical plans.

set -uo pipefail

scale=""
queries="1-22"
output_dir=""
build_dir=""
tpch_data=""
ranks=""
hosts=""
ranks_per_node=""
ssh_port="2222"
attempt_timeout="21600"
max_restarts="8"
seed="20260728"
query_dir_layout=0
force=0
synthesis_args=()

usage() {
    cat <<'EOF'
usage: run_distributed_tpch_synthesis.sh --scale N [options] [-- binary args]

Options:
  --scale N              required TPC-H scale factor
  --queries SPEC         comma-separated IDs/ranges (default: 1-22)
  --output-dir PATH      artifact directory containing plans/checkpoints
  --build-dir PATH       CMake build tree (default: EIGER_BUILD_DIR or build-b200/build)
  --tpch-data PATH       root containing sf<N> directories (default: TPCH_DATA_PATH)
  --ranks N              MPI ranks (default: visible local GPUs)
  --hosts LIST           Open MPI host list; omit for local execution
  --ranks-per-node N     add --map-by ppr:N:node
  --ssh-port N           remote SSH port (default: 2222)
  --attempt-timeout SEC  hard timeout for one query attempt (default: 21600)
  --max-restarts N       maximum progress-making restarts (default: 8)
  --seed N               randomized interleave seed (default: 20260728)
  --query-dir-layout     store each query's artifacts together under qNN/
  --force                rerun even when a nonempty plan already exists
  --help                 show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale) scale=$2; shift 2 ;;
        --queries) queries=$2; shift 2 ;;
        --output-dir) output_dir=$2; shift 2 ;;
        --build-dir) build_dir=$2; shift 2 ;;
        --tpch-data) tpch_data=$2; shift 2 ;;
        --ranks) ranks=$2; shift 2 ;;
        --hosts) hosts=$2; shift 2 ;;
        --ranks-per-node) ranks_per_node=$2; shift 2 ;;
        --ssh-port) ssh_port=$2; shift 2 ;;
        --attempt-timeout) attempt_timeout=$2; shift 2 ;;
        --max-restarts) max_restarts=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        --query-dir-layout) query_dir_layout=1; shift ;;
        --force) force=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; synthesis_args=("$@"); break ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value in "$scale" "$attempt_timeout" "$max_restarts" "$seed"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "error: scale, timeout, restarts, and seed must be integers" >&2
        exit 2
    fi
done
if (( scale <= 0 || attempt_timeout <= 0 )); then
    echo "error: scale and timeout must be positive" >&2
    exit 2
fi
if [[ -n "$ranks_per_node" && ! "$ranks_per_node" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --ranks-per-node must be a positive integer" >&2
    exit 2
fi
if [[ ! "$ssh_port" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --ssh-port must be a positive integer" >&2
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
binary=${ALPS_TPCH_SYNTH_BINARY:-$default_build/src/cuda/tpch/synthesize_tpch_ir_mpi}
device_count_binary=${ALPS_CUDA_VISIBLE_DEVICE_COUNT_BINARY:-$default_build/src/cuda/tpch/cuda_visible_device_count}
if [[ ! -x "$binary" ]]; then
    echo "error: synthesis binary is not executable: $binary" >&2
    exit 2
fi
if [[ -n "$hosts" && -z "$ranks" ]]; then
    echo "error: --ranks is required with --hosts" >&2
    exit 2
fi
if [[ -z "$hosts" ]]; then
    visible_ranks=$(distributed_tpch_visible_gpu_count "$device_count_binary") || exit 2
    if [[ -z "$ranks" ]]; then ranks=$visible_ranks; fi
fi
if [[ ! "$ranks" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --ranks must be a positive integer" >&2
    exit 2
fi
if [[ -z "$hosts" ]] && (( ranks > visible_ranks )); then
    echo "error: requested $ranks ranks but CUDA exposes only $visible_ranks GPUs" >&2
    exit 2
fi

declare -A selected=()
IFS=',' read -r -a chunks <<< "$queries"
for chunk in "${chunks[@]}"; do
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
    output_dir="$root/results/distributed_tpch_synthesis/$(date -u +%Y%m%dT%H%M%SZ)/sf$scale"
fi
if (( query_dir_layout )); then
    mkdir -p "$output_dir"
else
    mkdir -p "$output_dir/plans" "$output_dir/checkpoints" \
        "$output_dir/candidates" "$output_dir/logs"
fi
status_csv="$output_dir/status.csv"
if [[ ! -s "$status_csv" ]]; then
    printf 'timestamp_utc,query,scale,ranks,status,attempts,exit_code,plan,checkpoint,log\n' \
        > "$status_csv"
fi

checkpoint_digest() {
    local path=$1
    if [[ ! -s "$path" ]]; then
        printf '%s\n' '<missing>'
        return
    fi
    local digest remainder
    read -r digest remainder < <(sha256sum "$path")
    printf '%s\n' "$digest"
}

library_path="$default_build/src/cuda/tpch:$default_build/src/cuda/alps"
library_path+=":$default_build/src/cuda:$default_build/_deps/rmm-build"
library_path+=":$default_build/_deps/rapids_logger-build"
library_path+=":/storage/conda/x86_64/envs/eiger-b200-build/lib"
library_path+=":/usr/local/cuda/targets/x86_64-linux/lib:/opt/hpcx/ompi/lib"
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

echo "Distributed TPC-H physical-plan synthesis"
echo "  scale:      $scale"
echo "  queries:    ${query_ids[*]}"
echo "  ranks:      $ranks"
echo "  data:       $TPCH_DATA_PATH"
echo "  output:     $output_dir"

failures=0
passes=0
for query in "${query_ids[@]}"; do
    name="q$(printf '%02d' "$query")"
    if (( query_dir_layout )); then
        query_dir="$output_dir/$name"
        mkdir -p "$query_dir"
        plan="$query_dir/plan.json"
        checkpoint="$query_dir/checkpoint.json"
        candidates="$query_dir/candidates.csv"
        log="$query_dir/synthesis.log"
    else
        plan="$output_dir/plans/$name.json"
        checkpoint="$output_dir/checkpoints/$name.json"
        candidates="$output_dir/candidates/$name.csv"
        log="$output_dir/logs/$name.log"
    fi
    plan_before_identity=""
    if (( force )) && [[ -s "$plan" ]]; then
        plan_before_identity=$(stat -c '%d:%i' "$plan")
    fi
    if (( ! force )) && [[ -s "$plan" ]]; then
        echo "== Q$query: existing completed plan, skipped =="
        passes=$((passes + 1))
        continue
    fi
    exit_code=1
    attempt=0
    while (( attempt <= max_restarts )); do
        attempt=$((attempt + 1))
        echo "== Q$query attempt $attempt =="
        checkpoint_before=$(checkpoint_digest "$checkpoint")
        command=(
            timeout --signal=TERM --kill-after=30 "$attempt_timeout"
            "${mpi[@]}" "${rank_environment[@]}" \
            "$binary" "$query" --scale "$scale"
            --output-config "$plan" --checkpoint "$checkpoint"
            --candidate-csv "$candidates" --seed "$seed")
        if [[ -s "$checkpoint" ]]; then
            if ! (( force && attempt == 1 )) || [[ -z "$plan_before_identity" ]]; then
                command+=(--resume)
            fi
        fi
        command+=("${synthesis_args[@]}")
        "${command[@]}" 2>&1 | tee -a "$log"
        exit_code=${PIPESTATUS[0]}
        plan_is_fresh=0
        if [[ -s "$plan" ]]; then
            plan_after_identity=$(stat -c '%d:%i' "$plan")
            if [[ -z "$plan_before_identity" ||
                  "$plan_after_identity" != "$plan_before_identity" ]]; then
                plan_is_fresh=1
            fi
        fi
        if (( exit_code == 0 && plan_is_fresh )); then break; fi
        if (( exit_code == 64 )); then
            echo "Q$query terminal synthesis failure; not restarting"
            break
        fi
        checkpoint_after=$(checkpoint_digest "$checkpoint")
        if [[ "$checkpoint_after" == "$checkpoint_before" ]]; then
            echo "Q$query made no checkpoint progress; not restarting"
            break
        fi
        echo "Q$query checkpoint advanced; restarting from saved progress"
    done
    if (( exit_code == 0 && plan_is_fresh )); then
        status=pass
        passes=$((passes + 1))
    else
        status=fail
        failures=$((failures + 1))
    fi
    printf '%s,Q%d,%d,%d,%s,%d,%d,"%s","%s","%s"\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$query" "$scale" "$ranks" \
        "$status" "$attempt" "$exit_code" "$plan" "$checkpoint" "$log" \
        >> "$status_csv"
done

echo "Completed: $passes passed, $failures failed"
echo "Artifacts: $output_dir"
(( failures == 0 ))
