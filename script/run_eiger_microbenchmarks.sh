#!/bin/bash

machine=$1
BUILDDIR=${2:-build}
GPUID=${3:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

TODAY=$(date +%Y%m%d)
# PROFILE="--profile"
PROFILE=""

# Run all Eiger expression micro-benchmarks
mkdir -p exp_results/expr/$machine
./${BUILDDIR}/src/cuda/benchmarks/benchmark_expr -d $GPUID $PROFILE --csv exp_results/expr/${machine}/expr_${machine}_${TODAY}.csv

# Run all Eiger string micro-benchmarks
mkdir -p exp_results/string/$machine
./${BUILDDIR}/src/cuda/benchmarks/benchmark_string -d $GPUID $PROFILE -b string_partial_match --csv exp_results/string/${machine}/stringops_partial_match_${machine}_${TODAY}.csv --timeout 30
./${BUILDDIR}/src/cuda/benchmarks/benchmark_string -d $GPUID $PROFILE -b string_full_match --csv exp_results/string/${machine}/stringops_full_match_${machine}_${TODAY}.csv --timeout 30
./${BUILDDIR}/src/cuda/benchmarks/benchmark_string_sort -d $GPUID $PROFILE --csv exp_results/string/${machine}/string_sort_${machine}_${TODAY}.csv --timeout 30

# Run all Eiger groupby micro-benchmarks
mkdir -p exp_results/groupby/$machine
./${BUILDDIR}/src/cuda/benchmarks/benchmark_groupby -d $GPUID $PROFILE --csv exp_results/groupby/${machine}/groupby_max_${machine}_${TODAY}.csv

# Run all Eiger join micro-benchmarks
mkdir -p exp_results/join/$machine
for b in vary_left_table_size vary_right_table_size vary_match_ratio vary_zipfian_factor; 
do
    ./${BUILDDIR}/src/cuda/benchmarks/benchmark_ieq_join -d $GPUID $PROFILE -b $b --csv exp_results/join/${machine}/${b}_${machine}_${TODAY}.csv
done