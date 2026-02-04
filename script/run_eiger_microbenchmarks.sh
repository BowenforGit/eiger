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
mkdir -p exp_results/${machine}/expr
./${BUILDDIR}/src/cuda/benchmarks/benchmark_expr -d $GPUID $PROFILE --csv exp_results/${machine}/expr/expr_${machine}_${TODAY}.csv

# Run all Eiger sort_table micro-benchmarks
mkdir -p exp_results/${machine}/sort
./${BUILDDIR}/src/cuda/benchmarks/benchmark_sort_table -d $GPUID $PROFILE -b sort_table_numeric -a num_key_cols=[1,2,4] -a num_nonkey_cols=[1,2,4] --csv exp_results/${machine}/sort/sort_table_${machine}_${TODAY}.csv

# Run all Eiger sort_table fused-key micro-benchmarks
mkdir -p exp_results/${machine}/sort
./${BUILDDIR}/src/cuda/benchmarks/benchmark_sort_table_fused_key -d $GPUID $PROFILE --csv exp_results/${machine}/sort/sort_table_fused_key_${machine}_${TODAY}.csv


# Run all Eiger string micro-benchmarks
mkdir -p exp_results/${machine}/string
./${BUILDDIR}/src/cuda/benchmarks/benchmark_string -d $GPUID $PROFILE -b string_partial_match --csv exp_results/${machine}/string/stringops_partial_match_${machine}_${TODAY}.csv --timeout 30
./${BUILDDIR}/src/cuda/benchmarks/benchmark_string -d $GPUID $PROFILE -b string_full_match --csv exp_results/${machine}/string/stringops_full_match_${machine}_${TODAY}.csv --timeout 30
./${BUILDDIR}/src/cuda/benchmarks/benchmark_string_sort -d $GPUID $PROFILE --csv exp_results/${machine}/string/string_sort_${machine}_${TODAY}.csv --timeout 30
./${BUILDDIR}/src/cuda/benchmarks/benchmark_string -d $GPUID $PROFILE -b string_partial_match_bad_align --csv exp_results/${machine}/string/stringops_partial_match_bad_align_${machine}_${TODAY}.csv --timeout 30
./${BUILDDIR}/src/cuda/benchmarks/benchmark_string -d $GPUID $PROFILE -b string_full_match_bad_align --csv exp_results/${machine}/string/stringops_full_match_bad_align_${machine}_${TODAY}.csv --timeout 30

# Run all Eiger groupby micro-benchmarks
mkdir -p exp_results/${machine}/groupby
./${BUILDDIR}/src/cuda/benchmarks/benchmark_groupby -d $GPUID $PROFILE --csv exp_results/${machine}/groupby/groupby_max_${machine}_${TODAY}.csv

# Run all Eiger join micro-benchmarks
mkdir -p exp_results/${machine}/join
for b in vary_left_table_size vary_right_table_size vary_match_ratio vary_zipfian_factor; 
do
    ./${BUILDDIR}/src/cuda/benchmarks/benchmark_ieq_join -d $GPUID $PROFILE -b $b --csv exp_results/${machine}/join/${b}_${machine}_${TODAY}.csv
done