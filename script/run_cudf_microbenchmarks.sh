#!/bin/bash

machine=$1
GPUID=${2:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

TODAY=$(date +%Y%m%d)

# Check if pylibcudf is installed
if ! pip list | grep "pylibcudf"; then
    echo "Error: pylibcudf is not installed. Please install it before running this script."
    exit 1
fi

OUTPUT_DIR=exp_results/${machine}/cudf_microbenchmarks
mkdir -p $OUTPUT_DIR

# Run all cuDF micro-benchmarks
benchmarks=(
    # "join"
    # "groupby"
    # "expr"
    "strings"
    # "string_sort"
    # "sort_table"
)

for b in "${benchmarks[@]}"; do
    OUTPUT_DIR=$OUTPUT_DIR python3 src/cuda/benchmarks/cudf/bench_${b}.py
done