#!/usr/bin/env python3
"""Top-N most expensive CUDA kernels across the entire TPC-H workload
under best-cfg-per-step.

Pipeline:
  1) From the per-step CSV (GAMAX_BENCH_STEP_CSV), determine the
     best-cfg config that minimizes mean_ms for each (query, step_index).
  2) Group those (q, si) by (q, cfg_label) so we open each
     .nsys-rep at most once. Label format matches run_tpch_nsys.sh:
       q<NN>_<label>.nsys-rep   under  <nsys-base>/<machine>/sf<scale>/
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

# Display names per machine identifier — used in plot titles. Falls
# back to the raw identifier for unknown machines.
MACHINE_NAMES = {
    "h100":  "H100 NVL",
    "gh200": "Grace Hopper GH200",
}


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


# Kernels whose template parameters are functors/lambdas (so different
# specializations are different programs, not "the same kernel
# instantiated for different scalar types"). These must NOT be
# collapsed across specializations when aggregating.
NON_COLLAPSIBLE_KERNELS = (
    "cub::detail::transform::transform_kernel",
    "cub::detail::for_each::static_kernel",
)


def family_key(full: str) -> str:
    """Aggregation key for `aggregate_kernels`. Defaults to the short
    `<top-namespace>::<leaf>` form so different template specializations
    of the same kernel (e.g. `DeviceRadixSortOnesweepKernel<int32>` and
    `<int64>`) collapse into one row. The two CUB kernels in
    `NON_COLLAPSIBLE_KERNELS` keep the full mangled name as the key —
    their template args are user-supplied lambdas / functors, so the
    specializations represent fundamentally different bodies.
    """
    for exc in NON_COLLAPSIBLE_KERNELS:
        if exc in full:
            return full
    return short_kernel_name(full)


def short_kernel_name(full: str) -> str:
    """Trim a fully-qualified+demangled kernel name to
    `<top-namespace>::<leaf>`. Long chains like
    `cuco::detail::open_addressing_ns::…::probe_kernel` collapse to
    `cuco::probe_kernel`; the top-level namespace stays so it's
    obvious which library/component the kernel comes from. Same-leaf
    collisions across libraries get disambiguated downstream by
    `disambiguate_labels` with `#1`, `#2` suffixes.
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
    # Collapse runs of `:` to exactly two (handles `::::` left behind
    # by `<unnamed>` removal).
    s = _re.sub(r":{3,}", "::", s)
    # Trim leading/trailing `::`.
    s = s.strip(":").strip()
    # Keep `<top>::<leaf>` for chains of 3+; pass through 1- and
    # 2-segment names unchanged.
    parts = s.split("::")
    if len(parts) >= 3:
        s = f"{parts[0]}::{parts[-1]}"
    if len(s) > 60:
        s = s[:57] + "…"
    return s


def disambiguate_labels(rows: list[dict]) -> list[str]:
    """Given an ordered list of rows, return a parallel list of display
    labels. Rows from a kernel-short family with multiple rows in the
    full aggregation (`n_short_total > 1`) get a `#<short_seq>` suffix
    where `short_seq` is the row's total_ms-rank within its family
    (1 = highest). This is meaningful for non-collapsible kernels
    (`cub::transform_kernel`, `cub::static_kernel`) whose specializations
    appear as separate rows — even when only one ends up in the top-N
    bar chart, the #N suffix tells the reader which specialization it
    is and that others exist below the cut.
    """
    out = []
    for r in rows:
        short = r["kernel_short"]
        if int(r.get("n_short_total", 1)) > 1:
            out.append(f"{short} #{int(r['short_seq'])}")
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

    # Aggregate kernels: family-key → (total_ns, n_invocations, n_steps).
    # The family-key is `short_kernel_name(full)` for most kernels (so
    # different template specializations collapse into one row) but the
    # full mangled name for the two NON_COLLAPSIBLE_KERNELS — their
    # template args are user-supplied bodies, not just element types.
    kern_total_ns  = defaultdict(int)
    kern_invocs    = defaultdict(int)
    kern_steps     = defaultdict(int)
    kern_repr_full = {}             # key → first full name encountered
    kern_n_specs   = defaultdict(set)  # key → set of full names rolled in
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
            key = family_key(kname)
            kern_total_ns[key] += per_exec_ns
            kern_invocs[key]   += per_exec_invocs
            kern_steps[key]    += 1
            kern_n_specs[key].add(kname)
            if key not in kern_repr_full:
                kern_repr_full[key] = kname
        processed += 1
        if processed % 20 == 0:
            sys.stderr.write(f"# processed {processed}/{len(by_report)}\n")

    if missing_reports:
        sys.stderr.write(
            f"# WARN: {missing_reports} (q, label) reports missing\n")

    rows = []
    for key, ns in kern_total_ns.items():
        repr_full = kern_repr_full[key]
        rows.append({
            "kernel_full":       repr_full,
            "kernel_short":      short_kernel_name(repr_full),
            "n_specializations": len(kern_n_specs[key]),
            "total_ns":          int(round(ns)),
            "total_ms":          ns / 1e6,
            "n_invocs":          int(round(kern_invocs[key])),
            "n_step_rows":       kern_steps[key],
        })
    if not rows:
        # Common cause: nsys version on this host is older than the
        # one used to record the reports — `nsys stats` then fails the
        # .nsys-rep → .sqlite export and yields no kernel rows.
        sys.exit(
            f"error: no kernel rows aggregated from {processed} report(s) "
            f"under {nsys_dir}. Verify `nsys stats --report nvtx_kern_sum "
            f"--format csv <report>` produces a populated CSV (a version "
            f"mismatch between recording and analysis nsys will silently "
            f"export-fail).")
    out = (pd.DataFrame(rows)
             .sort_values("total_ns", ascending=False)
             .reset_index(drop=True))
    # Per-short-name rank (1 = highest total_ms in the group) and
    # group size. `disambiguate_labels` consumes these to add `#N`
    # suffixes to non-collapsible kernels' specializations.
    out["short_seq"]     = out.groupby("kernel_short").cumcount() + 1
    out["n_short_total"] = (out.groupby("kernel_short")["kernel_short"]
                              .transform("size"))
    return out


# Stable per-kernel-family colours so the same kernel gets the same
# colour across SF=10 and SF=30 figures.
KERNEL_FAMILY_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def plot_top_kernels(agg: pd.DataFrame, out_path: Path, scale: int,
                    machine_display: str, top_n: int = 8,
                    cfg_mode: str = "best", metric: str = "total"):
    """Plot the top-N kernels under one of two rankings:

    * `metric="total"` (default) — rank by total accumulated time.
      Bars are % of workload total; per-bar annotation shows
      pct / total_ms / kernel-count. Top-right callout shows the
      cumulative top-N share.
    * `metric="avg"` — rank by average per-invocation time
      (`total_ms / n_invocs`). Bars are absolute ms/invoc; per-bar
      annotation shows avg_ms / n_invocs / total_ms. Useful for
      spotting kernels that are individually slow vs. just
      called-a-lot.
    """
    if metric not in ("total", "avg"):
        raise ValueError(f"metric must be 'total' or 'avg', got {metric!r}")

    workload_total_ms = float(agg["total_ms"].sum())
    if metric == "total":
        ranked = agg.sort_values("total_ms", ascending=False)
    else:
        ranked = agg.copy()
        ranked["avg_ms"] = (ranked["total_ms"]
                            / ranked["n_invocs"].clip(lower=1))
        ranked = ranked.sort_values("avg_ms", ascending=False)
    top = ranked.head(top_n).reset_index(drop=True)
    if metric == "total":
        top["pct"] = top["total_ms"] / workload_total_ms * 100.0
        bar_heights = top["pct"]
        ylabel = (f"% of total "
                  f"{'best-cfg' if cfg_mode == 'best' else 'default-cfg'} "
                  f"kernel time")
        title_metric = "total kernel time"
    else:
        bar_heights = top["avg_ms"]
        ylabel = "average kernel time per invocation (ms)"
        title_metric = "per-invocation kernel time"
    labels = disambiguate_labels(top.to_dict(orient="records"))

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = [KERNEL_FAMILY_COLORS[i % len(KERNEL_FAMILY_COLORS)]
              for i in range(len(top))]
    bars = ax.bar(range(len(top)), bar_heights, color=colors,
                  edgecolor="white", linewidth=0.6, width=0.72)

    ymax = float(bar_heights.max())
    for i, (b, ms, n) in enumerate(zip(bars, top["total_ms"], top["n_invocs"])):
        h = float(bar_heights.iloc[i])
        if metric == "total":
            pct = float(top["pct"].iloc[i])
            label = f"{pct:.1f}%\n{ms:.1f} ms\n×{int(n)}"
        else:
            avg = float(top["avg_ms"].iloc[i])
            label = f"{avg:.3g} ms\n×{int(n)}\ntotal {ms:.1f} ms"
        ax.text(b.get_x() + b.get_width()/2, h + ymax*0.02,
                label, ha="center", va="bottom", fontsize=11,
                color="#222222", linespacing=1.15)

    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=30, fontsize=11,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
            color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax * 1.32)

    # Top-right callout. For `total` mode show the cumulative top-N
    # share; for `avg` mode there is no "share of" interpretation —
    # surface the workload's total kernel time instead.
    if metric == "total":
        callout = (f"Top-{top_n} kernels = {float(top['pct'].sum()):.1f}% "
                   f"of total kernel time")
    else:
        callout = (f"Workload total kernel time = "
                   f"{workload_total_ms:.1f} ms across "
                   f"{len(agg)} kernel families")
    ax.text(0.99, 0.97, callout,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12, color="#222222",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4",
                      edgecolor="#bbbbbb", linewidth=0.6))
    cfg_blurb = ("best cfg per step" if cfg_mode == "best"
                 else "default cfg, bench baseline")
    ax.set_title(
        f"Top-{top_n} CUDA kernels by {title_metric}  ·  {cfg_blurb}  "
        f"·  SF={scale}  ·  {machine_display}",
        fontsize=15, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    jpg = out_path.with_suffix(".jpg")
    fig.savefig(jpg, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")
    if metric == "total":
        print(f"  total {cfg_mode}-cfg kernel time = "
              f"{workload_total_ms:.2f} ms across {len(agg)} kernel "
              f"families; top-{top_n} share = "
              f"{float(top['pct'].sum()):.1f}%")
    else:
        print(f"  slowest avg kernel time = {ymax:.3g} ms; "
              f"workload total = {workload_total_ms:.2f} ms across "
              f"{len(agg)} kernel families")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--machine", required=True,
                   help="machine identifier — picks the input CSV "
                        "under <data-dir>/<machine>/tpch/<date>/<tag>/ "
                        "and the nsys reports under "
                        "<nsys-base>/<machine>/<date>/<tag>/")
    p.add_argument("--date", required=True,
                   help="run date as YYYYMMDD; matches the dated "
                        "folder produced by run_tpch_perf.sh")
    p.add_argument("--tag", required=True,
                   help="run tag (the --tag passed to "
                        "run_tpch_perf.sh / GAMAX_TAG for "
                        "run_tpch_nsys.sh)")
    p.add_argument("--scale", required=True, type=int,
                   help="scale factor (must match the steps CSV)")
    p.add_argument("--cfg-mode", choices=["best", "default"], default="best",
                   help="`best`: per-step min over all cfgs (oracle). "
                        "`default`: always use the baseline cell.")
    p.add_argument("--data-dir", default="tools/results/data",
                   help="base dir for input/output CSVs (default: "
                        "tools/results/data)")
    p.add_argument("--nsys-base", default="/scratch/wubo/gamax_tpch_nsys",
                   help="base dir holding "
                        "<machine>/<date>/<tag>/sf<N>/q<NN>_<label>.nsys-rep "
                        "(default: /scratch/wubo/gamax_tpch_nsys)")
    p.add_argument("--out-base", default="tools/results/pdf/tpch_bench",
                   help="base output dir; figures land in "
                        "<out-base>/<machine>/<date>/<tag>/ (default: "
                        "tools/results/pdf/tpch_bench)")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--metric", choices=["total", "avg"], default="total",
                   help="`total`: rank by accumulated kernel time "
                        "(default). `avg`: rank by mean per-invocation "
                        "kernel time (total_ms / n_invocs) — surfaces "
                        "individually-slow kernels rather than ones that "
                        "are slow in aggregate because they're hot.")
    args = p.parse_args()

    machine = args.machine
    date = args.date
    tag = args.tag
    scale = int(args.scale)
    mode = args.cfg_mode
    metric = args.metric
    display = MACHINE_NAMES.get(machine, machine)

    run_dir = Path(args.data_dir) / machine / "tpch" / date / tag
    steps_csv = run_dir / f"bench_steps_sf{scale}.csv"
    if not steps_csv.exists():
        sys.exit(f"error: missing steps CSV: {steps_csv}")
    nsys_dir = Path(args.nsys_base) / machine / date / tag

    # `total` ranking keeps the legacy filename for back-compat;
    # `avg` ranking gets an `_avg_` infix so both can coexist.
    pdf_stem = (f"tpch_top_kernels_{mode}_sf{scale}" if metric == "total"
                else f"tpch_top_kernels_avg_{mode}_sf{scale}")
    out_pdf = Path(args.out_base) / machine / date / tag / f"{pdf_stem}.pdf"
    csv_out = run_dir / f"{pdf_stem.replace('tpch_', '')}.csv"

    agg = aggregate_kernels(steps_csv, scale, nsys_dir, cfg_mode=mode)
    if agg.empty:
        sys.exit("no kernel rows aggregated — check report paths + steps CSV")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plot_top_kernels(agg, out_pdf, scale, display, args.top_n,
                     cfg_mode=mode, metric=metric)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(csv_out, index=False)
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
