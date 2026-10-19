#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Profile one distributed TPC-H query. Loading and warmup occur before capture;
# only the single `tpch.profile` NVTX range is recorded.

set -euo pipefail

query=""
scale=""
output_dir=""
build_dir=""
tpch_data=""
ranks=""
hosts=""
ranks_per_node=""
ssh_port="2222"
profile_ranks="all"
profile_host=""
gpu_metrics_devices="none"
gpu_metrics_frequency="10000"
gpu_metrics_set=""
gpu_metrics_rank="all"
nic_metrics="false"
single_report="false"
benchmark_args=()

usage() {
    cat <<'EOF'
usage: profile_distributed_tpch_nsys.sh --query N --scale N [options] [-- benchmark args]

Options:
  --output-dir PATH      report directory
  --build-dir PATH       CMake build tree (default: EIGER_BUILD_DIR or build-b200/build)
  --tpch-data PATH       root containing sf<N> directories (default: TPCH_DATA_PATH)
  --ranks N              MPI ranks (default: visible local GPUs; required with --hosts)
  --hosts LIST           Open MPI host list; omit for local execution
  --ranks-per-node N     add --map-by ppr:N:node
  --ssh-port N           remote SSH port (default: 2222)
  --profile-ranks SPEC   comma-separated MPI ranks (default: all)
  --profile-host HOST    profile only ranks whose hostname exactly matches HOST
  --gpu-metrics-devices SPEC
                         none, all, cuda-visible, local-rank, or GPU IDs
                         (default: none). local-rank selects the MPI local
                         rank's GPU and is intended for multi-node profiling.
  --gpu-metrics-frequency HZ
                         GPU metric sampling frequency (default: 10000)
  --gpu-metrics-set NAME optional NSYS GPU metric-set alias
  --gpu-metrics-rank RANK
                         only this global MPI rank collects GPU metrics;
                         all selected ranks still collect CUDA/NVTX traces
  --nic-metrics          collect NIC/HCA metrics
  --single-report        profile the local MPI process tree into qNN.nsys-rep

Reports are named qNN_<hostname>_rank<RANK>.nsys-rep. Golden verification
runs after capture because NSYS stops collection, but does not kill the rank,
when the outer profile range ends. --single-report is local-only and produces
one report containing every local rank; use it for system-wide GPU metrics.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --query) query=$2; shift 2 ;;
        --scale) scale=$2; shift 2 ;;
        --output-dir) output_dir=$2; shift 2 ;;
        --build-dir) build_dir=$2; shift 2 ;;
        --tpch-data) tpch_data=$2; shift 2 ;;
        --ranks) ranks=$2; shift 2 ;;
        --hosts) hosts=$2; shift 2 ;;
        --ranks-per-node) ranks_per_node=$2; shift 2 ;;
        --ssh-port) ssh_port=$2; shift 2 ;;
        --profile-ranks) profile_ranks=$2; shift 2 ;;
        --profile-host) profile_host=$2; shift 2 ;;
        --gpu-metrics-devices) gpu_metrics_devices=$2; shift 2 ;;
        --gpu-metrics-frequency) gpu_metrics_frequency=$2; shift 2 ;;
        --gpu-metrics-set) gpu_metrics_set=$2; shift 2 ;;
        --gpu-metrics-rank) gpu_metrics_rank=$2; shift 2 ;;
        --nic-metrics) nic_metrics="true"; shift ;;
        --single-report) single_report="true"; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; benchmark_args=("$@"); break ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ ! "$query" =~ ^[0-9]+$ ]] || (( query < 1 || query > 22 )); then
    echo "error: --query must be in [1, 22]" >&2
    exit 2
fi
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
rank_launcher=${ALPS_TPCH_PROFILE_RANK_LAUNCHER:-$script_dir/profile_distributed_tpch_rank.sh}
[[ -x "$binary" ]] || { echo "error: missing binary: $binary" >&2; exit 2; }
[[ -x "$rank_launcher" ]] || { echo "error: missing rank launcher: $rank_launcher" >&2; exit 2; }
command -v nsys >/dev/null || { echo "error: nsys is not on PATH" >&2; exit 2; }
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
if [[ "$profile_ranks" != all && ! "$profile_ranks" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "error: --profile-ranks must be 'all' or comma-separated ranks" >&2
    exit 2
fi
if [[ "$single_report" == true && -n "$hosts" ]]; then
    echo "error: --single-report only supports a local MPI process tree" >&2
    exit 2
fi
if [[ "$single_report" == true && "$profile_ranks" != all ]]; then
    echo "error: --profile-ranks cannot be combined with --single-report" >&2
    exit 2
fi
if [[ ! "$gpu_metrics_frequency" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --gpu-metrics-frequency must be a positive integer" >&2
    exit 2
fi
if [[ "$gpu_metrics_rank" != all && ! "$gpu_metrics_rank" =~ ^[0-9]+$ ]]; then
    echo "error: --gpu-metrics-rank must be 'all' or a nonnegative rank" >&2
    exit 2
fi
if [[ "$gpu_metrics_rank" != all ]] && (( gpu_metrics_rank >= ranks )); then
    echo "error: --gpu-metrics-rank is outside [0, $((ranks - 1))]" >&2
    exit 2
fi
if [[ "$profile_ranks" != all ]]; then
    IFS=',' read -r -a selected_ranks <<< "$profile_ranks"
    for selected_rank in "${selected_ranks[@]}"; do
        if (( 10#$selected_rank >= ranks )); then
            echo "error: profile rank $selected_rank is outside [0, $((ranks - 1))]" >&2
            exit 2
        fi
    done
fi
if [[ -z "$output_dir" ]]; then
    output_dir="$root/results/distributed_tpch_profiles/$(date -u +%Y%m%dT%H%M%SZ)/sf$scale"
fi
mkdir -p "$output_dir"

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
export NSYS_GPU_METRICS_DEVICES=$gpu_metrics_devices
export NSYS_GPU_METRICS_FREQUENCY=$gpu_metrics_frequency
export NSYS_GPU_METRICS_SET=$gpu_metrics_set
export NSYS_GPU_METRICS_RANK=$gpu_metrics_rank
export NSYS_NIC_METRICS=$nic_metrics
export NSYS_PROFILE_HOST=$profile_host

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
    -x GAMAX_RMM_RESOURCE -x GAMAX_MEM_RESERVE_BYTES
    -x NSYS_GPU_METRICS_DEVICES -x NSYS_GPU_METRICS_FREQUENCY
    -x NSYS_GPU_METRICS_SET -x NSYS_GPU_METRICS_RANK
    -x NSYS_NIC_METRICS -x NSYS_PROFILE_HOST)
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    mpi+=(-x CUDA_VISIBLE_DEVICES)
fi
rank_environment=()
distributed_tpch_configure_nccl_cta_environment mpi rank_environment

output_prefix="$output_dir/q$(printf '%02d' "$query")_%h_rank%q{OMPI_COMM_WORLD_RANK}"
if [[ "$single_report" == true ]]; then
    nsys_args=(profile
        --trace=cuda,nvtx,mpi
        --mpi-impl=openmpi
        --sample=none
        --cpuctxsw=none
        --trace-fork-before-exec=true
        --capture-range=nvtx
        --nvtx-capture=tpch.profile@gamax.tpch
        --capture-range-end=stop
        --force-overwrite=false
        --output "$output_dir/q$(printf '%02d' "$query")")
    if [[ "$gpu_metrics_devices" != none ]]; then
        nsys_args+=(
            --gpu-metrics-devices="$gpu_metrics_devices"
            --gpu-metrics-frequency="$gpu_metrics_frequency")
        if [[ -n "$gpu_metrics_set" ]]; then
            nsys_args+=(--gpu-metrics-set="$gpu_metrics_set")
        fi
    fi
    if [[ "$nic_metrics" == true ]]; then
        nsys_args+=(--nic-metrics=true)
    fi
    exec nsys "${nsys_args[@]}" \
        "${mpi[@]}" "${rank_environment[@]}" \
        "$binary" "$query" --scale "$scale" --profile \
        "${benchmark_args[@]}"
fi

exec "${mpi[@]}" \
    "${rank_environment[@]}" \
    "$rank_launcher" \
        "$profile_ranks" "$output_prefix" \
        "$binary" "$query" --scale "$scale" --profile \
        "${benchmark_args[@]}"
