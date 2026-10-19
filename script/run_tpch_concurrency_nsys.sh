#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Capture a few nsys traces of selected concurrency cells in --profile mode
# (1 warmup + 2 measured, NVTX-rich) so the overlap is visible on the timeline
# and the traces stay small.
#
# Usage:
#   run_tpch_concurrency_nsys.sh <bin> <outdir> <sf> SPEC [SPEC ...]
# SPEC = qa:qb[:mode[:policy]]
#   qa:qb                      -> concurrent single-shot   (e.g. 9:21)
#   qa:-:isolated              -> isolated single-shot
#   qa:qb:batch-concurrent:alternating
set -u
BIN=${1:?bin}; OUT=${2:?outdir}; SF=${3:?sf}; shift 3
export TPCH_DATA_PATH=${TPCH_DATA_PATH:-/storage/tpch}
mkdir -p "$OUT"

for spec in "$@"; do
    IFS=: read -r qa qb mode policy <<<"$spec"
    mode=${mode:-concurrent}
    qbtag=""
    [ -n "${qb:-}" ] && [ "$qb" != "-" ] && qbtag="_q$qb"
    name="nsys_${mode}_q${qa}${qbtag}${policy:+_$policy}"
    args=(--mode "$mode" --qa "$qa" --sf "$SF" --profile --out "/tmp/$name")
    [ -n "${qb:-}" ] && [ "$qb" != "-" ] && args+=(--qb "$qb")
    [ -n "${policy:-}" ] && args+=(--policy "$policy" --iters 10)
    echo ">> $name : ${args[*]}"
    nsys profile --trace=cuda,nvtx,osrt --force-overwrite true \
         -o "$OUT/$name" "$BIN" "${args[@]}" > "$OUT/$name.nsyslog" 2>&1
    echo "   rc=$? -> $OUT/$name.nsys-rep ($(du -h "$OUT/$name.nsys-rep" 2>/dev/null | cut -f1))"
done
