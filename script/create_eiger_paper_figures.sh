#!/bin/bash

# This script generates plots for the Eiger paper benchmarks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

benchmarks=(
    "expr"
    "groupby"
    "ieq_join"
    "sort_fused"
    "string_sort"
    "string"
)

export CUDF_RESULTS_STATS="min"

for b in "${benchmarks[@]}"; do
    echo "+++++ Generating plot for benchmark: $b +++++"
    python3 tools/visualize/plot_${b}.py
    if [ $? -ne 0 ]; then
        exit 1
    fi
done