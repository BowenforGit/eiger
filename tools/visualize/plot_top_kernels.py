#!/usr/bin/env python3
"""Top-N most expensive CUDA kernels across the entire TPC-H workload
under best-cfg-per-step.

Pipeline:
  1) From the per-step CSV (GAMAX_BENCH_STEP_CSV), determine the
     best-cfg config that minimizes mean_ms for each (query, step_index).
  2) Group those (q, si) by (q, cfg_label) so we open each
     .nsys-rep at most once. Label format matches run_tpch_nsys.sh:
       q<NN>_<label>.nsys-rep   under  /scratch/.../sf<scale>/
  3) For each report, run `nsys stats --report nvtx_kern_sum --format csv`
     once. Result rows are (NVTX range, kernel name, total ns, ...).
  4) Filter rows whose NVTX range matches `:step[<si>]:*` for any
     step_index this report contributes a "best" for.
  5) Aggregate kernel totals globally — two kernels are the same iff
     their fully-demangled names are bit-equal.
  6) Plot the top-N (default 8) as a bar chart with a short, paper-
     friendly kernel-name label.

Caveats:
  * The nsys --profile cell is a single execution; mean_ms in the
    steps CSV is averaged. The kernel mix is deterministic for a given
    cfg, so kernel-share ratios from the --profile run faithfully
    represent the best-cfg behaviour — exact ms values may drift
    slightly from the bench's timed mean.
  * Reports whose label is missing on disk (cells gated out by
    GAMAX_NSYS_GAP) fall back to the baseline cell's report. The
    baseline is always profiled, so the fallback is always available.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.spines.top"]   = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.linewidth"]    = 0.8
matplotlib.rcParams["axes.edgecolor"]    = "#444444"
matplotlib.rcParams["axes.labelcolor"]   = "#222222"
matplotlib.rcParams["xtick.color"]       = "#222222"
matplotlib.rcParams["ytick.color"]       = "#222222"
import matplotlib.pyplot as plt
import pandas as pd

AXES = ("filter", "project", "sort", "group_by", "eq_join")
AXIS_SHORT = {"filter":"fil","project":"prj","sort":"srt",
              "group_by":"grp","eq_join":"ejn"}
NEUTRAL = ("default", "n/a")
NVTX_RE = re.compile(r":step\[(\d+)\]:")


def cfg_label(row) -> str:
    """Compact label for a config — matches run_tpch_nsys.sh's naming."""
    set_axes = [a for a in AXES if row[a] not in NEUTRAL]
    if not set_axes:
        return "baseline"
    a = set_axes[0]
    return f"{AXIS_SHORT[a]}-{row[a]}"


def find_report(nsys_dir: Path, scale: int, q: int, label: str) -> Path | None:
    """Locate the .nsys-rep for (scale, q, label). Falls back to the
    baseline cell if the requested label isn't present (some cells
    are gated out by run_tpch_nsys.sh's GAP filter)."""
    direct = nsys_dir / f"sf{scale}" / f"q{q:02d}_{label}.nsys-rep"
    if direct.exists():
        return direct
    fallback = nsys_dir / f"sf{scale}" / f"q{q:02d}_baseline.nsys-rep"
    if fallback.exists():
        return fallback
    return None


def run_nvtx_kern_sum(report: Path) -> list[dict]:
    """Run `nsys stats --report nvtx_kern_sum --format csv` and parse
    the resulting CSV rows. The first call on a .nsys-rep creates a
    .sqlite sibling (slow); subsequent calls reuse it (fast)."""
    proc = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_kern_sum", "--format", "csv",
         "--quiet", str(report)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(
            f"# WARN nsys stats failed for {report}: {proc.stderr[:200]}\n")
        return []

    rows = []
    # Parse the CSV portion of stdout — nsys may print a banner/path
    # line first; skip everything until we find the header row.
    started = False
    reader = csv.reader(proc.stdout.splitlines())
    header = None
    for r in reader:
        if not r:
            continue
        if not started:
            if r[0].strip() == "NVTX Range":
                header = r
                started = True
            continue
        rows.append(dict(zip(header, r)))
    return rows


def short_kernel_name(full: str) -> str:
    """Trim a fully-qualified+demangled kernel name to something
    paper-friendly. Returns the function path, e.g.
      `gamax::groupby::hash::compute_groupid_kernel`.
    """
    import re as _re
    s = full.strip().strip('"')
    if s.startswith("void "):
        s = s[5:]
    # Drop function arguments `(...)` at the tail.
    open_paren = s.find("(")
    if open_paren > 0:
        s = s[:open_paren]
    # Drop `<unnamed>` segments BEFORE we strip generic <...> brackets
    # — otherwise we'd leave behind dangling `::` pairs.
    s = s.replace("<unnamed>", "")
    # Strip outermost `<...>` template args.
    out = []
    depth = 0
    for ch in s:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    s = "".join(out)
    # Squash `cub::detail::` → `cub::` and `gamax::experimental::` →
    # `gamax::` for brevity.
    s = s.replace("gamax::experimental::", "gamax::")
    s = s.replace("cub::detail::", "cub::")
    # Collapse runs of `:` to exactly two (handles `::::` left behind
    # by `<unnamed>` removal).
    s = _re.sub(r":{3,}", "::", s)
    # Trim leading/trailing `::`.
    s = s.strip(":").strip()
    if len(s) > 60:
        s = s[:57] + "…"
    return s


def disambiguate_labels(rows: list[dict]) -> list[str]:
    """Given an ordered list of rows with `kernel_short` and
    `kernel_full`, return a parallel list of display labels where any
    short-label collisions are disambiguated with a `#1`, `#2` suffix
    in the row's encounter order. The full-name remains the identity
    key — the suffix only resolves visual ambiguity in the bar chart.
    """
    seen_count = defaultdict(int)
    total_count = defaultdict(int)
    for r in rows:
        total_count[r["kernel_short"]] += 1
    out = []
    for r in rows:
        short = r["kernel_short"]
        if total_count[short] > 1:
            seen_count[short] += 1
            out.append(f"{short} #{seen_count[short]}")
        else:
            out.append(short)
    return out


def aggregate_kernels(steps_csv: Path, scale: int, nsys_dir: Path,
                      cfg_mode: str = "best"):
    """For each (q, step_index), pick a cfg according to `cfg_mode`:
       * `best`    — the cfg minimizing mean_ms for that step (oracle
                     per-step picker; matches the breakdown plot's
                     "best-per-step" composite).
       * `default` — always the all-default cell, i.e. the bench
                     baseline (filter=nofuse, project=nofuse,
                     sort=no_skf, group_by=hash, eq_join=hash_gfur).
    Open the matching nsys-rep, sum kernel times within `:step[si]:*`
    NVTX ranges. Returns a DataFrame with full kernel name, short
    name, total_ms, total_ns, n_invocations, n_step_rows.
    """
    df = pd.read_csv(steps_csv, keep_default_na=False, na_values=[""])
    df = df[df["scale"] == scale].copy()
    if df.empty:
        sys.exit(f"steps CSV has no rows at scale={scale}")

    df["cfg_label"] = df.apply(cfg_label, axis=1)

    if cfg_mode == "best":
        # Per (q, si), pick the row with minimum mean_ms.
        idx = df.groupby(["query", "step_index"])["mean_ms"].idxmin()
        picked = df.loc[idx, ["query", "step_index", "step_kind",
                              "cfg_label", "mean_ms"]].reset_index(drop=True)
    elif cfg_mode == "default":
        # Per (q, si), keep only the baseline cell (cfg_label="baseline").
        picked = (df[df["cfg_label"] == "baseline"]
                  [["query", "step_index", "step_kind", "cfg_label",
                    "mean_ms"]]
                  .reset_index(drop=True))
        if picked.empty:
            sys.exit("default mode: no rows with cfg_label='baseline' "
                     "in steps CSV")
    else:
        sys.exit(f"unknown cfg_mode: {cfg_mode!r}")

    # (q, cfg_label) → set of step_indices we want to attribute.
    by_report = defaultdict(set)
    for _, r in picked.iterrows():
        by_report[(int(r["query"]), r["cfg_label"])].add(int(r["step_index"]))

    sys.stderr.write(
        f"# cfg_mode={cfg_mode} scale={scale} "
        f"unique reports needed = {len(by_report)} "
        f"(steps to attribute = {len(picked)})\n")

    # Aggregate kernels: full-name → (total_ns, n_invocations, n_steps).
    kern_total_ns = defaultdict(int)
    kern_invocs   = defaultdict(int)
    kern_steps    = defaultdict(int)
    missing_reports = 0
    processed = 0
    for (q, label), wanted_sis in sorted(by_report.items()):
        report = find_report(nsys_dir, scale, q, label)
        if report is None:
            sys.stderr.write(f"# skip q{q} {label}: no nsys-rep on disk\n")
            missing_reports += 1
            continue
        rows = run_nvtx_kern_sum(report)
        # NVTX range like ":step[10]:GroupBy" — parse the index.
        for r in rows:
            m = NVTX_RE.search(r.get("NVTX Range", ""))
            if not m: continue
            si = int(m.group(1))
            if si not in wanted_sis: continue
            kname = r.get("Kernel Name", "").strip().strip('"')
            try:
                ns        = int(r.get("Total Time (ns)", "0"))
                invocs    = int(r.get("Kern Inst", "0"))
                nvtx_inst = int(r.get("NVTX Inst", "1")) or 1
            except ValueError:
                continue
            # The bench fixture runs the plan twice per cell under
            # `nsys profile --profile`: once in the pre-flight try/catch
            # on cudaStreamDefault, once inside NVBench's state.exec
            # lambda. Both fire the executor's :step[N]:<Kind> NVTX
            # range, so nvtx_kern_sum's Total Time and Kern Inst are
            # summed across both executions (NVTX Inst = 2). Normalize
            # by NVTX Inst to recover per-execution numbers.
            per_exec_ns      = ns / nvtx_inst
            per_exec_invocs  = invocs / nvtx_inst
            kern_total_ns[kname] += per_exec_ns
            kern_invocs[kname]   += per_exec_invocs
            kern_steps[kname]    += 1
        processed += 1
        if processed % 20 == 0:
            sys.stderr.write(f"# processed {processed}/{len(by_report)}\n")

    if missing_reports:
        sys.stderr.write(
            f"# WARN: {missing_reports} (q, label) reports missing\n")

    rows = []
    for kname, ns in kern_total_ns.items():
        rows.append({
            "kernel_full":  kname,
            "kernel_short": short_kernel_name(kname),
            "total_ns":     int(round(ns)),
            "total_ms":     ns / 1e6,
            "n_invocs":     int(round(kern_invocs[kname])),
            "n_step_rows":  kern_steps[kname],
        })
    out = pd.DataFrame(rows).sort_values("total_ns", ascending=False)
    return out


# Stable per-kernel-family colours so the same kernel gets the same
# colour across SF=10 and SF=30 figures.
KERNEL_FAMILY_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def plot_top_kernels(agg: pd.DataFrame, out_path: Path, scale: int,
                    top_n: int = 8, cfg_mode: str = "best"):
    top = agg.head(top_n).reset_index(drop=True)
    total = float(agg["total_ms"].sum())
    top["pct"] = top["total_ms"] / total * 100.0
    labels = disambiguate_labels(top.to_dict(orient="records"))

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = [KERNEL_FAMILY_COLORS[i % len(KERNEL_FAMILY_COLORS)]
              for i in range(len(top))]
    bars = ax.bar(range(len(top)), top["pct"], color=colors,
                  edgecolor="white", linewidth=0.6, width=0.72)

    ymax = float(top["pct"].max())
    for b, pct, ms, n in zip(bars, top["pct"], top["total_ms"],
                             top["n_invocs"]):
        ax.text(b.get_x() + b.get_width()/2, pct + ymax*0.02,
                f"{pct:.1f}%\n{ms:.1f} ms\n×{int(n)}",
                ha="center", va="bottom", fontsize=11, color="#222222",
                linespacing=1.15)

    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=30, fontsize=11,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel(
        f"% of total {'best-cfg' if cfg_mode == 'best' else 'default-cfg'} "
        f"kernel time", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
            color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax * 1.32)
    cfg_blurb = ("best cfg per step" if cfg_mode == "best"
                 else "default cfg, bench baseline")
    ax.set_title(
        f"Top-{top_n} most expensive CUDA kernels  ·  {cfg_blurb}  "
        f"·  SF={scale}  ·  H100 NVL",
        fontsize=15, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    jpg = out_path.with_suffix(".jpg")
    fig.savefig(jpg, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")
    print(f"  total {cfg_mode}-cfg kernel time = {total:.2f} ms across "
          f"{len(agg)} distinct kernel names; top-{top_n} share = "
          f"{float(top['pct'].sum()):.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", required=True,
                   help="per-step CSV (GAMAX_BENCH_STEP_CSV output)")
    p.add_argument("--scale", required=True, type=int,
                   help="scale factor (must match the steps CSV)")
    p.add_argument("--nsys-dir", required=True,
                   help="base dir holding sf<N>/q<NN>_<label>.nsys-rep")
    p.add_argument("--out", required=True,
                   help="output PDF path")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--cfg-mode", choices=["best", "default"], default="best",
                   help="`best`: per-step min over all cfgs (oracle). "
                        "`default`: always use the baseline cell.")
    p.add_argument("--csv-out",
                   help="optional path to dump the full kernel "
                        "aggregation table for paper / spreadsheet use")
    args = p.parse_args()

    agg = aggregate_kernels(Path(args.steps), args.scale,
                            Path(args.nsys_dir), cfg_mode=args.cfg_mode)
    if agg.empty:
        sys.exit("no kernel rows aggregated — check report paths + steps CSV")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_top_kernels(agg, out, args.scale, args.top_n,
                     cfg_mode=args.cfg_mode)

    if args.csv_out:
        agg.to_csv(args.csv_out, index=False)
        print(f"wrote {args.csv_out}")


if __name__ == "__main__":
    main()
