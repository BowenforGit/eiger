#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Resumable driver for the two-query TPC-H concurrency sweep.
#
# Runs one SUBPROCESS PER CELL (isolated query or concurrent pair) so a hard
# CUDA-context corruption / OOM in one cell can't poison the rest of the sweep
# (mirrors script/run_tpch_axis_probe.sh). Resumable: a cell whose per-run
# summary already exists is skipped; a crashed cell leaves a .failed marker.
#
# Usage:
#   run_tpch_concurrency.sh <bin> <data-dir> [sf] [qmin] [qmax]
# Env:
#   TPCH_DATA_PATH (default /storage/tpch), RUN_TIMEOUT (s, default 900),
#   REFRESH_EVERY (concurrent cells between isolated refresh rounds, default 50),
#   EXTRA_ARGS (passed to every invocation, e.g. "--min-samples 10 --max-samples 30")
set -u

BIN=${1:?benchmark binary path required}
OUTDIR=${2:?output data dir required}
SF=${3:-10}
QMIN=${4:-1}
QMAX=${5:-22}
export TPCH_DATA_PATH=${TPCH_DATA_PATH:-/storage/tpch}
RUN_TIMEOUT=${RUN_TIMEOUT:-900}
REFRESH_EVERY=${REFRESH_EVERY:-50}
EXTRA_ARGS=${EXTRA_ARGS:-}

RUNS="$OUTDIR/runs"
mkdir -p "$RUNS"

# ---- one metadata.json for the whole sweep --------------------------------
meta="$OUTDIR/metadata.json"
if [[ ! -f "$meta" ]]; then
    commit=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)
    dirty=$(git -C "$(dirname "$0")/.." status --porcelain 2>/dev/null | head -c1)
    smi=$(nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader 2>/dev/null | head -1)
    {
        echo "{"
        echo "  \"date\": \"$(date -Is)\","
        echo "  \"host\": \"$(hostname)\","
        echo "  \"git_commit\": \"$commit\","
        echo "  \"git_dirty\": $([[ -n "$dirty" ]] && echo true || echo false),"
        echo "  \"gpu\": \"$smi\","
        echo "  \"scale_factor\": $SF,"
        echo "  \"tpch_data_path\": \"$TPCH_DATA_PATH\","
        echo "  \"run_timeout_s\": $RUN_TIMEOUT,"
        echo "  \"refresh_every\": $REFRESH_EVERY,"
        echo "  \"extra_args\": \"$EXTRA_ARGS\","
        echo "  \"clock_pinning\": \"unavailable (non-root); SM clock + throttle recorded per sample\""
        echo "}"
    } > "$meta"
    echo "wrote $meta"
fi

run_one() {   # <name> <args...>
    local name="$1"; shift
    local pref="$RUNS/$name"
    if [[ -f "$pref.summary.csv" ]]; then echo "skip(done) $name"; return 0; fi
    if [[ -f "$pref.failed"      ]]; then echo "skip(fail) $name"; return 0; fi
    echo ">> $name"
    if timeout "$RUN_TIMEOUT" "$BIN" "$@" $EXTRA_ARGS --out "$pref" > "$pref.log" 2>&1; then
        :
    else
        local rc=$?
        echo "$rc" > "$pref.failed"
        echo "   FAIL(rc=$rc): $(tail -n1 "$pref.log")"
    fi
}

mapfile -t Q < <(seq "$QMIN" "$QMAX")

echo "== Phase 1: isolated baselines (round 0) =="
for q in "${Q[@]}"; do
    run_one "iso_q${q}_r0" --mode isolated --qa "$q" --sf "$SF" --refresh-round 0
done

echo "== Phase 2: concurrent pairs (i<=j) + periodic isolated refresh =="
cell=0; round=0
for ((i=0; i<${#Q[@]}; i++)); do
    for ((j=i; j<${#Q[@]}; j++)); do
        run_one "conc_q${Q[i]}_q${Q[j]}" --mode concurrent --qa "${Q[i]}" --qb "${Q[j]}" --sf "$SF"
        cell=$((cell+1))
        if (( cell % REFRESH_EVERY == 0 )); then
            round=$((round+1))
            echo "   -- isolated refresh round $round (after $cell concurrent cells) --"
            for q in "${Q[@]}"; do
                run_one "iso_q${q}_r${round}" --mode isolated --qa "$q" --sf "$SF" --refresh-round "$round"
            done
        fi
    done
done

echo "== Aggregate per-run CSVs into masters =="
aggregate() {  # <glob-suffix> <out>
    local suf="$1" out="$2" hdr=0
    : > "$out"
    # Include batch_runs/ so re-running this driver after the batch phase does
    # NOT drop batch rows from the masters (both drivers aggregate the same set).
    for f in "$RUNS"/*."$suf".csv "$OUTDIR"/batch_runs/*."$suf".csv; do
        [[ -e "$f" ]] || continue
        if [[ $hdr -eq 0 ]]; then cat "$f" >> "$out"; hdr=1; else tail -n +2 "$f" >> "$out"; fi
    done
    local n; n=$(wc -l < "$out"); (( n > 0 )) && n=$((n - 1)) || n=0
    echo "   $out : $n rows"
}
aggregate summary "$OUTDIR/summary.csv"
aggregate raw     "$OUTDIR/raw.csv"

nfail=$(ls "$RUNS"/*.failed 2>/dev/null | wc -l)
echo "== done. failed cells: $nfail =="
[[ "$nfail" -gt 0 ]] && ls "$RUNS"/*.failed 2>/dev/null | sed 's#.*/##;s/\.failed//' | sed 's/^/   FAILED: /'
exit 0
