#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Resumable driver for the GREEN-CONTEXT SM-partitioning TPC-H concurrency study.
#
# Extends the base two-query concurrency sweep with spatial partitioning: each
# query runs on its own DISJOINT green-context SM slice instead of time-sharing
# the full device. Compares four strategies per pair (serial-shortest-first,
# time-shared, partitioned-balanced, partitioned-proportional) plus the
# per-query SM-scaling curve T_Q(N) and a full split sweep on the showcase pair.
#
# Design: src/cuda/benchmarks/tpch_concurrency/DESIGN_greenctx.md
#
# One SUBPROCESS PER CELL (mirrors run_tpch_concurrency.sh): a hard CUDA-context
# corruption / OOM / (formerly) green-ctx failure in one cell can't poison the
# rest. Resumable: a cell whose per-run summary exists is skipped; a crashed
# cell leaves a .failed marker (records the green-safe subset empirically).
#
# Usage:
#   run_tpch_concurrency_green.sh <bin> <out-dir> [sf]
# Env:
#   TPCH_DATA_PATH (default /storage/tpch), RUN_TIMEOUT (s, default 900),
#   EXTRA_ARGS (passed to every invocation, e.g. "--min-samples 10 --max-samples 30")
set -u

BIN=${1:?benchmark binary path required}
OUTDIR=${2:?output data dir required}
SF=${3:-10}
export TPCH_DATA_PATH=${TPCH_DATA_PATH:-/storage/tpch}
RUN_TIMEOUT=${RUN_TIMEOUT:-900}
EXTRA_ARGS=${EXTRA_ARGS:-}

RUNS="$OUTDIR/runs"
mkdir -p "$RUNS"

# ---- study parameters ------------------------------------------------------
ALL_Q=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
# SM-scaling grid (single green partition). 132 = green@full-device (overhead
# control ~= native). Curve is interpolated to actual split sizes for S_part.
SCALE_SMS=(8 16 32 64 96 132)
# Representative pairs "qa qb" (qa gets the proportional n_a share). Chosen from
# GH200 native T^full + green-report efficiency to span lopsided / saturating /
# under-utilizing regimes. Q22+Q21 is the showcase (full split sweep below).
PAIRS=(
  "22 21"   # extreme lopsided (25x) -- SHOWCASE
  "22 9"    # lopsided (tiny x big scaler)
  "16 21"   # lopsided (saturate-early x biggest)
  "2 5"     # lopsided (launch-bound x scaler)
  "21 21"   # biggest saturating self-pair (control: partition should lose most)
  "9 9"     # saturating self-pair (control)
  "22 22"   # under-utilizing self-pair
  "2 16"    # both under-utilizing (time-share won prior)
)
# Queries needing a native-baseline refresh round (the pair-queries).
REFRESH_Q=(2 5 9 16 21 22)
# Showcase full split sweep: Q22 (=qa) SM request grid; Q21 gets the remainder.
SHOWCASE="22 21"
SWEEP_SMS=(8 16 32 48 64 84 100 116 124)

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

# Median native T^full for a query (col 10 = median; metric col 6 == "T").
tfull() { awk -F, '$1=="iso_single" && $6=="T"{print $10; exit}' \
              "$RUNS/iso_q$1_r0.summary.csv" 2>/dev/null; }
# Device SM total (native rows record it in sms_a = col 18).
device_total() { awk -F, '$1=="iso_single"{print $18; exit}' \
                     "$RUNS/iso_q22_r0.summary.csv" 2>/dev/null; }

echo "== Phase 1: native full-device baselines T^full (round 0, all 22) =="
for q in "${ALL_Q[@]}"; do
    run_one "iso_q${q}_r0" --mode isolated --qa "$q" --sf "$SF" --refresh-round 0
done

TOTAL=$(device_total); TOTAL=${TOTAL:-132}
BAL=$(( (TOTAL/2/8)*8 ))          # closest-balanced request (align 8), e.g. 64
echo "   device SM total = $TOTAL ; balanced request = $BAL"

# ---- metadata.json (now that TOTAL is known) ------------------------------
meta="$OUTDIR/metadata.json"
if [[ ! -f "$meta" ]]; then
    root="$(cd "$(dirname "$0")/.." && pwd)"
    commit=$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)
    dirty=$(git -C "$root" status --porcelain 2>/dev/null | head -c1)
    smi=$(nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader 2>/dev/null | head -1)
    {
        echo "{"
        echo "  \"study\": \"tpch_concurrency_greenctx\","
        echo "  \"date\": \"$(date -Is)\","
        echo "  \"host\": \"$(hostname)\","
        echo "  \"git_commit\": \"$commit\","
        echo "  \"git_dirty\": $([[ -n "$dirty" ]] && echo true || echo false),"
        echo "  \"gpu\": \"$smi\","
        echo "  \"device_sm_total\": $TOTAL,"
        echo "  \"sm_alignment\": 8,"
        echo "  \"scale_factor\": $SF,"
        echo "  \"scale_sms_grid\": \"${SCALE_SMS[*]}\","
        echo "  \"balanced_request\": $BAL,"
        echo "  \"tpch_data_path\": \"$TPCH_DATA_PATH\","
        echo "  \"run_timeout_s\": $RUN_TIMEOUT,"
        echo "  \"extra_args\": \"$EXTRA_ARGS\","
        echo "  \"green_safe_note\": \"all 22 green-safe on GH200/sm_90 (CUB-green blocker was sm_100/Blackwell-only)\","
        echo "  \"clock_pinning\": \"unavailable (non-root); SM clock + throttle recorded per sample\","
        echo "  \"s_part_note\": \"S_part denominators interpolated from the coarse T_Q(N) curve to actual partition size\""
        echo "}"
    } > "$meta"
    echo "wrote $meta"
fi

echo "== Phase 2: SM-scaling curve iso_green (all 22 x {${SCALE_SMS[*]}}) =="
# N=8 (smallest) doubles as the green-verify gate: a query that fails green
# leaves a .failed marker here and is caught before pairing.
for n in "${SCALE_SMS[@]}"; do
    for q in "${ALL_Q[@]}"; do
        run_one "isog_q${q}_sms${n}" --mode isolated --qa "$q" --sf "$SF" --sms "$n"
    done
done

echo "== Phase 3: time-shared concurrent (native full device) for the pairs =="
for p in "${PAIRS[@]}"; do
    read -r a b <<<"$p"
    run_one "conc_q${a}_q${b}" --mode concurrent --qa "$a" --qb "$b" --sf "$SF"
done

echo "== Phase 4: partitioned concurrent (balanced + proportional) for the pairs =="
for p in "${PAIRS[@]}"; do
    read -r a b <<<"$p"
    # balanced (~50/50) -- always
    run_one "cg_q${a}_q${b}_balanced" --mode concurrent --qa "$a" --qb "$b" --sf "$SF" \
            --sms-a "$BAL" --policy balanced
    # proportional -- skip for self-pairs (proportional == balanced there)
    if [[ "$a" != "$b" ]]; then
        ta=$(tfull "$a"); tb=$(tfull "$b")
        if [[ -n "$ta" && -n "$tb" ]]; then
            na=$(awk -v t="$TOTAL" -v x="$ta" -v y="$tb" \
                     'BEGIN{n=int(t*x/(x+y)+0.5); if(n<1)n=1; if(n>=t)n=t-1; print n}')
            echo "   proportional Q${a}+Q${b}: T=$ta/$tb -> n_a=$na"
            run_one "cg_q${a}_q${b}_proportional" --mode concurrent --qa "$a" --qb "$b" \
                    --sf "$SF" --sms-a "$na" --policy proportional
        else
            echo "   WARN: missing T^full for Q${a}/Q${b}; skipping proportional"
        fi
    fi
done

echo "== Phase 5: showcase full split sweep ($SHOWCASE) =="
read -r sa sb <<<"$SHOWCASE"
for n in "${SWEEP_SMS[@]}"; do
    run_one "sweep_q${sa}_q${sb}_sms${n}" --mode concurrent --qa "$sa" --qb "$sb" \
            --sf "$SF" --sms-a "$n" --policy sweep
done

echo "== Phase 6: native baseline refresh (round 1, pair-queries) =="
for q in "${REFRESH_Q[@]}"; do
    run_one "iso_q${q}_r1" --mode isolated --qa "$q" --sf "$SF" --refresh-round 1
done

echo "== Aggregate per-run CSVs into masters =="
aggregate() {  # <suffix> <out>
    local suf="$1" out="$2" hdr=0
    : > "$out"
    for f in "$RUNS"/*."$suf".csv; do
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
