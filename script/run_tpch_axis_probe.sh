#!/usr/bin/env bash
# Per-query wrapper around `tpch_ir_axis_probe`.
#
# Why per-query: a CUDA-context-corrupting cell (illegal_address etc.)
# poisons all subsequent CUDA calls in the same process — kernels on
# downstream queries then fail with `cudaErrorInvalidDevice` or worse,
# segfault. Spawning one subprocess per query keeps each query's
# context isolated, so a bad cell only contaminates that query's
# remaining cells (NOT cells in other queries).
#
# Usage:
#   run_tpch_axis_probe.sh <probe-binary> <scale> <out-csv> [query-min] [query-max]
#
# Example:
#   build/script/run_tpch_axis_probe.sh \
#       build/src/cuda/benchmarks/tpch_ir_axis_probe \
#       1 /tmp/sf1_probe.csv

set -u

probe="${1:?probe binary path required}"
sf="${2:?scale factor required}"
out="${3:?output CSV path required}"
qmin="${4:-1}"
qmax="${5:-22}"

if [[ ! -x "$probe" ]]; then
    echo "error: probe binary not executable: $probe" >&2
    exit 2
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

rm -f "$out"
header_done=0
total_pass=0
total_fail=0

for q in $(seq "$qmin" "$qmax"); do
    csv="$tmpdir/q${q}.csv"
    # `--quiet` suppresses the per-100 progress to stderr.
    # An exit code != 0 here is expected for queries whose context
    # gets corrupted mid-sweep — we still consume whatever rows the
    # probe wrote before exiting.
    "$probe" --scale "$sf" --out "$csv" \
             --query-min "$q" --query-max "$q" --quiet \
             >/dev/null 2>&1
    rc=$?
    if [[ ! -f "$csv" ]]; then
        echo "Q$q: probe produced no output (exit=$rc)" >&2
        continue
    fi
    if [[ $header_done -eq 0 ]]; then
        cat "$csv" >> "$out"
        header_done=1
    else
        tail -n +2 "$csv" >> "$out"
    fi
    pq=$(awk -F, '$8=="pass"{n++} END{print n+0}' "$csv")
    fq=$(awk -F, '$8=="fail"{n++} END{print n+0}' "$csv")
    total_pass=$((total_pass + pq))
    total_fail=$((total_fail + fq))
    msg="Q$q: pass=$pq fail=$fq"
    if [[ $rc -ne 0 ]]; then
        msg="$msg (exit=$rc — context corruption likely; subsequent cells may be missing)"
    fi
    echo "$msg" >&2
done

echo "" >&2
echo "Total: pass=$total_pass fail=$total_fail" >&2
echo "Output: $out" >&2
