#!/usr/bin/env python3
"""Per-query baseline/best kernel-only execution time, from the
per-cell Nsight Systems reports.

For each query in the e2e CSV:
  * baseline = the all-default cell (axis=baseline / label=default)
  * best     = the cell with the minimum GPU Time (same selection as
               the "best-cfg" column of the per-query e2e summary)

Both cells' .nsys-rep are opened with
`nsys stats --report nvtx_kern_sum --format csv`. We sum kernel time
across rows whose NVTX range matches `:step[<N>]:*`, normalized by
NVTX Inst per kernel row (defends against pre-flight double-counting
from older runs). The result is "pure kernel execution time" — no
harness overhead, no parquet decode, no H2D copies.

Output: CSV on stdout, one row per query —
  query, baseline_kernel_ms, best_kernel_ms, best_label

Usage:
  python3 tools/visualize/extract_kernel_times.py \\
      --machine h100 --scale 10 > /tmp/kernel_times_h100_sf10.csv
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

AXES = ("filter", "project", "sort", "group_by", "eq_join")
AXIS_SHORT = {"filter":"fil","project":"prj","sort":"srt",
              "group_by":"grp","eq_join":"ejn"}
NEUTRAL = ("default", "n/a")
NVTX_RE = re.compile(r":step\[(\d+)\]:")


def cfg_label(row):
    """Compact label for a config — matches run_tpch_nsys.sh's naming."""
    set_axes = [a for a in AXES if row[a] not in NEUTRAL]
    if not set_axes:
        return "baseline"
    a = set_axes[0]
    return f"{AXIS_SHORT[a]}-{row[a]}"


def find_report(nsys_dir: Path, scale: int, q: int, label: str):
    """Locate the .nsys-rep for (scale, q, label). Falls back to the
    baseline cell if the requested label isn't on disk (some cells
    are gated out by run_tpch_nsys.sh's GAP filter)."""
    direct = nsys_dir / f"sf{scale}" / f"q{q:02d}_{label}.nsys-rep"
    if direct.exists():
        return direct
    fallback = nsys_dir / f"sf{scale}" / f"q{q:02d}_baseline.nsys-rep"
    if fallback.exists():
        return fallback
    return None


def kernel_time_ns(report: Path) -> float:
    """Sum kernel ns across `:step[N]:*` NVTX ranges, normalized by
    NVTX Inst per kernel row."""
    proc = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_kern_sum", "--format", "csv",
         "--quiet", str(report)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"error: nsys stats failed for {report}:\n"
                 f"{proc.stderr[:400]}")
    started = False
    header = None
    total_ns = 0.0
    n_rows = 0
    for r in csv.reader(proc.stdout.splitlines()):
        if not r: continue
        if not started:
            if r[0].strip() == "NVTX Range":
                header = r; started = True
            continue
        d = dict(zip(header, r))
        if not NVTX_RE.search(d.get("NVTX Range", "")):
            continue
        try:
            ns = int(d.get("Total Time (ns)", "0"))
            inst = int(d.get("NVTX Inst", "1")) or 1
        except ValueError:
            continue
        total_ns += ns / inst
        n_rows += 1
    if n_rows == 0:
        sys.exit(
            f"error: no `:step[N]:` rows in nvtx_kern_sum for {report}. "
            f"Common cause: nsys version mismatch (the host that recorded "
            f"this report is newer than the host running nsys stats).")
    return total_ns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--machine", required=True,
                   help="machine identifier — picks the e2e CSV under "
                        "<data-dir>/<machine>/tpch/ and the nsys reports "
                        "under <nsys-base>/<machine>/")
    p.add_argument("--scale", required=True, type=int)
    p.add_argument("--data-dir", default="tools/results/data")
    p.add_argument("--nsys-base", default="/scratch/wubo/gamax_tpch_nsys")
    args = p.parse_args()

    machine, scale = args.machine, int(args.scale)
    e2e_csv = (Path(args.data_dir) / machine / "tpch"
               / f"bench_e2e_sf{scale}.csv")
    nsys_dir = Path(args.nsys_base) / machine
    if not e2e_csv.exists():
        sys.exit(f"error: missing e2e CSV: {e2e_csv}")

    cells_by_q = {}
    with open(e2e_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Skipped") != "No": continue
            if int(r["scale"]) != scale: continue
            q = int(r["query"])
            ms = float(r["GPU Time (sec)"]) * 1000.0
            cells_by_q.setdefault(q, []).append({
                **{a: r[a] for a in AXES},
                "ms":    ms,
                "label": cfg_label(r),
            })

    print("query,baseline_kernel_ms,best_kernel_ms,best_label")
    for q in sorted(cells_by_q):
        cells = cells_by_q[q]
        base = next((c for c in cells if c["label"] == "baseline"), None)
        if base is None:
            sys.stderr.write(f"# WARN: q{q} has no baseline cell — skipped\n")
            continue
        best = min(cells, key=lambda c: c["ms"])
        base_rep = find_report(nsys_dir, scale, q, "baseline")
        best_rep = find_report(nsys_dir, scale, q, best["label"])
        if base_rep is None:
            sys.exit(f"error: missing baseline nsys-rep for q{q}")
        if best_rep is None:
            sys.exit(
                f"error: missing best nsys-rep for q{q} "
                f"(label={best['label']})")
        base_ns = kernel_time_ns(base_rep)
        best_ns = kernel_time_ns(best_rep)
        sys.stderr.write(
            f"# q{q:02d}  baseline={base_ns/1e6:7.2f} ms  "
            f"best={best_ns/1e6:7.2f} ms  ({best['label']})\n")
        print(f"{q},{base_ns/1e6:.3f},{best_ns/1e6:.3f},{best['label']}")


if __name__ == "__main__":
    main()
