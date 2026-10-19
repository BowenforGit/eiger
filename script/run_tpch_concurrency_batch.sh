#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Batch (sustained-throughput) phase of the TPC-H concurrency study, on a
# data-selected subset of queries. Runs, per selected query, an isolated K-iter
# batch baseline (B_Q), and per pair (i<=j) a concurrent K-iter batch under BOTH
# submission policies (self-pairs: dedicated only, since alternating == dedicated).
# Appends its aggregated rows to the study's summary.csv / raw.csv.
#
# Usage: run_tpch_concurrency_batch.sh <bin> <data-dir> <sf> <iters> Q1 Q2 ...
set -u
BIN=${1:?bin}; OUT=${2:?data-dir}; SF=${3:?sf}; ITERS=${4:?iters}; shift 4
QUERIES=("$@")
export TPCH_DATA_PATH=${TPCH_DATA_PATH:-/storage/tpch}
RUN_TIMEOUT=${RUN_TIMEOUT:-900}
RUNS="$OUT/batch_runs"; mkdir -p "$RUNS"

run_one() {   # <name> <args...>
    local name="$1"; shift
    local pref="$RUNS/$name"
    [ -f "$pref.summary.csv" ] && { echo "skip(done) $name"; return 0; }
    [ -f "$pref.failed" ]      && { echo "skip(fail) $name"; return 0; }
    echo ">> $name"
    if timeout "$RUN_TIMEOUT" "$BIN" "$@" --out "$pref" > "$pref.log" 2>&1; then :; else
        echo $? > "$pref.failed"; echo "   FAIL: $(tail -n1 "$pref.log")"; fi
}

echo "== batch isolated baselines =="
for q in "${QUERIES[@]}"; do
    run_one "biso_q${q}" --mode batch-isolated --qa "$q" --iters "$ITERS" --sf "$SF"
done

echo "== batch concurrent pairs (dedicated + alternating) =="
for ((i=0; i<${#QUERIES[@]}; i++)); do
    for ((j=i; j<${#QUERIES[@]}; j++)); do
        a=${QUERIES[i]}; b=${QUERIES[j]}
        run_one "bconc_q${a}_q${b}_dedicated" \
            --mode batch-concurrent --qa "$a" --qb "$b" --iters "$ITERS" --policy dedicated --sf "$SF"
        if [ "$a" != "$b" ]; then
            run_one "bconc_q${a}_q${b}_alternating" \
                --mode batch-concurrent --qa "$a" --qb "$b" --iters "$ITERS" --policy alternating --sf "$SF"
        fi
    done
done

echo "== re-aggregate all runs (single-shot + batch) into masters (idempotent) =="
for suf in summary raw; do
    master="$OUT/$suf.csv"; tmp="$master.tmp"; h=0; : > "$tmp"
    for f in "$OUT"/runs/*."$suf".csv "$RUNS"/*."$suf".csv; do
        [ -e "$f" ] || continue
        if [ $h -eq 0 ]; then cat "$f" >> "$tmp"; h=1; else tail -n +2 "$f" >> "$tmp"; fi
    done
    mv "$tmp" "$master"
done
nfail=$(ls "$RUNS"/*.failed 2>/dev/null | wc -l)
echo "== batch done. failed: $nfail =="
[ "$nfail" -gt 0 ] && ls "$RUNS"/*.failed 2>/dev/null | sed 's#.*/##;s/.failed//;s/^/   FAILED: /'
exit 0
