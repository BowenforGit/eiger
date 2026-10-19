#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Profile selected non-NCCL rank-0 kernels in the distributed TPC-H profile range.

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
section_file=""
kernel_name=""
benchmark_args=()

usage() {
    cat <<'EOF'
usage: profile_distributed_tpch_ncu.sh --query N --scale N --section-file PATH \
       --kernel-name REGEX \
       [options] [-- benchmark args]

Options:
  --output-dir PATH      report directory
  --build-dir PATH       CMake build tree
  --tpch-data PATH       root containing sf<N> directories
  --ranks N              MPI ranks (required with --hosts)
  --hosts LIST           Open MPI host list; omit for local execution
  --ranks-per-node N     add --map-by ppr:N:node
  --ssh-port N           remote SSH port (default: 2222)
  --section-file PATH    native NCU section containing counters and launch metrics
  --kernel-name REGEX    NCU demangled kernel-name expression
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
        --section-file) section_file=$2; shift 2 ;;
        --kernel-name) kernel_name=$2; shift 2 ;;
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
if [[ ! -f "$section_file" ]]; then
    echo "error: --section-file must name a readable NCU section: $section_file" >&2
    exit 2
fi
if [[ -z "$kernel_name" ]]; then
    echo "error: --kernel-name is required" >&2
    exit 2
fi
section_file=$(cd -- "$(dirname -- "$section_file")" && pwd)/$(basename -- "$section_file")

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
root=$(cd "$script_dir/.." && pwd)
source "$script_dir/distributed_tpch_common.sh"
default_build=${build_dir:-${EIGER_BUILD_DIR:-$root/build}}
default_build=$(cd -- "$default_build" && pwd)
binary=${ALPS_TPCH_BENCH_BINARY:-$default_build/src/cuda/tpch/benchmark_tpch_ir_mpi}
device_count_binary=${ALPS_CUDA_VISIBLE_DEVICE_COUNT_BINARY:-}
if [[ -z "$device_count_binary" ]]; then
    device_count_binary=$default_build/src/cuda/tpch/cuda_visible_device_count
fi
rank_launcher=${ALPS_TPCH_NCU_RANK_LAUNCHER:-}
if [[ -z "$rank_launcher" ]]; then
    rank_launcher=$script_dir/profile_distributed_tpch_ncu_rank.sh
fi
[[ -x "$binary" ]] || { echo "error: missing binary: $binary" >&2; exit 2; }
if [[ ! -x "$rank_launcher" ]]; then
    echo "error: missing rank launcher: $rank_launcher" >&2
    exit 2
fi
if ! command -v "${NCU_BIN:-ncu}" >/dev/null; then
    echo "error: ncu is not on PATH" >&2
    exit 2
fi

if [[ -z "$ranks" ]]; then
    if [[ -n "$hosts" ]]; then
        echo "error: --ranks is required with --hosts" >&2
        exit 2
    fi
    ranks=$(distributed_tpch_visible_gpu_count "$device_count_binary")
fi
if [[ ! "$ranks" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --ranks must be a positive integer" >&2
    exit 2
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
export ALPS_TPCH_NCU_SECTION_FILE=$section_file
export ALPS_TPCH_NCU_KERNEL_NAME=$kernel_name

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
    -x ALPS_TPCH_NCU_SECTION_FILE -x ALPS_TPCH_NCU_KERNEL_NAME)
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    mpi+=(-x CUDA_VISIBLE_DEVICES)
fi
if [[ -n ${NCU_BIN:-} ]]; then
    mpi+=(-x NCU_BIN)
fi
rank_environment=()
distributed_tpch_configure_nccl_cta_environment mpi rank_environment

exec "${mpi[@]}" "${rank_environment[@]}" \
    "$rank_launcher" "$output_dir" "$query" \
    "$binary" "$query" --scale "$scale" --profile \
    "${benchmark_args[@]}"
