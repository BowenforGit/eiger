#!/usr/bin/env python3
"""Plot benchmark_tpch_ir results.

Inputs are derived from --machine, --date, --tag, --scale by
convention (matching `script/run_tpch_perf.sh`'s output layout):
  e2e CSV   : <data-dir>/<machine>/tpch/<date>/<tag>/bench_e2e_sf<N>.csv
  steps CSV : <data-dir>/<machine>/tpch/<date>/<tag>/bench_steps_sf<N>.csv

Outputs land in <out>/<machine>/<date>/<tag>/. Two figures:
  1) tpch_e2e        : 22 subfigures (2 rows × 11 cols), one bar per config
  2) tpch_breakdown  : same layout, stacked bars per step + an extra
                       "best-of-each-step" bar composed of per-step minima

Both figures use the bench's one-hot sweep convention:
  * one bar for the all-default baseline
  * one bar per non-default value of each axis (with all other axes at
    their bench-local default)

Configurations are labelled compactly (e.g. "default", "filter=fuse",
"group_by=sort_gftr"), grouped by which axis is varied.
"""

import argparse
import os
import sys
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
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

AXES = ["filter", "project", "sort", "group_by", "eq_join"]
NA = "n/a"
DEFAULT = "default"

# Display names per machine identifier — used in plot titles. Falls
# back to the raw identifier (e.g. for an unknown machine the title
# just shows the directory name).
MACHINE_NAMES = {
    "h100":  "H100 NVL",
    "gh200": "Grace Hopper GH200",
}

# Muted, paper-friendly palette per StepKind (Tableau-10 + a few
# lighter complements for the rarer kinds). Hand-tuned so that the
# join family (EqJoin / SemiJoin / AntiJoin) sit on a coherent warm
# spectrum and the reduction family (GroupBy / ScalarAgg / Distinct)
# stays cool — adjacent stack segments don't clash.
STEP_COLORS = {
    "Source":     "#bab0ac",   # muted gray
    "Filter":     "#4e79a7",   # muted blue
    "Project":    "#f28e2b",   # muted orange
    "Reorder":    "#bab0ac",   # gray (low-cost glue step)
    "Sort":       "#9c755f",   # muted brown
    "TopK":       "#d37295",   # rose
    "GroupBy":    "#59a14f",   # muted green
    "ScalarAgg":  "#edc948",   # mustard
    "EqJoin":     "#e15759",   # muted red
    "SemiJoin":   "#76b7b2",   # muted teal
    "AntiJoin":   "#b07aa1",   # muted purple
    "Distinct":   "#a0cbe8",   # light blue
    "Concat":     "#ffbe7d",   # light orange
    "CrossJoin":  "#8cd17d",   # light green
}


AXIS_SHORT = {
    "filter":   "fil",
    "project":  "prj",
    "sort":     "srt",
    "group_by": "grp",
    "eq_join":  "ejn",
}

VALUE_SHORT = {
    # join algos
    "hash_gfur":           "hash",
    "sort_merge_gfur":     "smerge.ur",
    "sort_merge_gftr":     "smerge.tr",
    "partition_hash_gfur": "phash.ur",
    "partition_hash_gftr": "phash.tr",
    # groupby algos
    "sort_gftr":           "sort.tr",
    "sort_gftr_opt":       "sort.tr.opt",
    "sort_gfur":           "sort.ur",
    "partition_gftr":      "part.tr",
    "partition_gfur":      "part.ur",
}

def short_label(row: pd.Series) -> str:
    """Compact label for a config: which single axis (if any) is non-
    default-non-n/a, else 'default'."""
    set_axes = [a for a in AXES if row[a] not in (NA, DEFAULT)]
    if not set_axes:
        return "default"
    if len(set_axes) == 1:
        a = set_axes[0]
        v = VALUE_SHORT.get(row[a], row[a])
        return f"{AXIS_SHORT[a]}.{v}"
    # Shouldn't happen under the at-most-one-set rule; render verbosely.
    return ",".join(f"{AXIS_SHORT[a]}.{VALUE_SHORT.get(row[a], row[a])}"
                     for a in set_axes)


def axis_for(row: pd.Series) -> str:
    set_axes = [a for a in AXES if row[a] not in (NA, DEFAULT)]
    return set_axes[0] if len(set_axes) == 1 else "baseline"


def order_key(row: pd.Series):
    """Sort cells: baseline first, then grouped by axis (filter, project,
    sort, group_by, eq_join), then by hint value alphabetically."""
    set_axes = [a for a in AXES if row[a] not in (NA, DEFAULT)]
    if not set_axes:
        return (-1, "", "")
    a = set_axes[0]
    return (AXES.index(a), a, row[a])


# pandas would parse the literal string "n/a" as NaN by default; the
# axis sentinel must round-trip exactly. Pass na_filter=False on every
# read to keep the strings intact.
def load_e2e(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df = df[df["Skipped"] == "No"].copy()
    df["query"] = df["query"].astype(int)
    df["scale"] = df["scale"].astype(int)
    df["gpu_ms"] = df["GPU Time (sec)"].astype(float) * 1000.0
    df["label"] = df.apply(short_label, axis=1)
    df["axis"] = df.apply(axis_for, axis=1)
    df["sort_key"] = df.apply(order_key, axis=1)
    return df


def load_steps(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df["query"] = df["query"].astype(int)
    df["scale"] = df["scale"].astype(int)
    df["mean_ms"] = df["mean_ms"].astype(float)
    df["min_ms"]  = df["min_ms"].astype(float)
    df["label"]   = df.apply(short_label, axis=1)
    df["axis"]    = df.apply(axis_for, axis=1)
    df["sort_key"] = df.apply(order_key, axis=1)
    return df


# Axis-family palette (mirrors STEP_COLORS so the e2e bars use the
# same colour as the stack segment for the operator they vary).
AXIS_PALETTE = {
    "baseline":  "#7f7f7f",
    "filter":    "#4e79a7",
    "project":   "#f28e2b",
    "sort":      "#9c755f",
    "group_by":  "#59a14f",
    "eq_join":   "#e15759",
}

def axis_color(axis: str) -> str:
    """Bar background colour for the e2e figure: one colour per axis
    family so the eye groups bars by which operator is being varied."""
    return AXIS_PALETTE.get(axis, "#666666")


E2E_PALETTE = {
    "baseline": "#7f7f7f",   # neutral gray — same as the gray ring in the muted set
    "worst":    "#e15759",   # muted red
    "best":     "#59a14f",   # muted green
}

def plot_e2e(df: pd.DataFrame, out_path: Path, scale: int,
             machine_display: str):
    """22 subfigures, three bars per query (baseline / worst-cell /
    best-cell), where each bar is the e2e GPU time of that
    configuration. baseline = the all-default cell (which matches
    bench_defaults: filter/project=nofuse, sort=no_skf, group_by=hash,
    eq_join=hash_gfur). worst/best are the slowest and fastest cells
    in the one-hot sweep for that query.
    """
    fig, axes = plt.subplots(2, 11, figsize=(28, 9), sharey=False)
    axes = axes.flatten()

    BAR_LABELS = ["baseline", "worst", "best"]
    BAR_COLORS = [E2E_PALETTE[k] for k in BAR_LABELS]

    for i, q in enumerate(range(1, 23)):
        ax = axes[i]
        sub = df[df["query"] == q].copy()
        if sub.empty:
            ax.set_title(f"Q{q}\n(no data)", fontsize=14)
            ax.axis("off")
            continue

        baseline = sub[sub["axis"] == "baseline"]
        if baseline.empty:
            baseline = sub[sub["label"] == "default"]
        base_ms  = float(baseline["gpu_ms"].iloc[0]) if len(baseline) else float("nan")
        worst_ms = float(sub["gpu_ms"].max())
        best_ms  = float(sub["gpu_ms"].min())
        values   = [base_ms, worst_ms, best_ms]

        bars = ax.bar(range(3), values, color=BAR_COLORS, width=0.7,
                      edgecolor="white", linewidth=0.4)

        # Numeric value above each bar, helps a paper reader pick out
        # numbers without zooming.
        ymax = max(values)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width()/2, v + ymax*0.025,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=10,
                    color="#222222")

        ax.set_xticks(range(3))
        ax.set_xticklabels(BAR_LABELS, fontsize=11)
        ax.set_title(f"Q{q}", fontsize=15, fontweight="bold")
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
                color="#bbbbbb", zorder=0)
        ax.set_axisbelow(True)
        # Headroom for the value labels.
        ax.set_ylim(0, ymax * 1.2)
        if i % 11 == 0:
            ax.set_ylabel("GPU time (ms)", fontsize=13)

    legend_handles = [
        mpatches.Patch(color=E2E_PALETTE["baseline"], label="baseline (all default)"),
        mpatches.Patch(color=E2E_PALETTE["worst"],    label="worst single-axis variation"),
        mpatches.Patch(color=E2E_PALETTE["best"],     label="best single-axis variation"),
    ]
    fig.suptitle(f"TPC-H per-query end-to-end GPU time  ·  baseline / worst / "
                 f"best  ·  SF={scale}  ·  {machine_display}",
                 fontsize=17, fontweight="bold", y=0.99)
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.005), fontsize=12, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    jpg_path = out_path.with_suffix(".jpg")
    fig.savefig(jpg_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def plot_breakdown(df: pd.DataFrame, out_path: Path, scale: int,
                   machine_display: str):
    """22 subfigures, three stacked bars per query:

      * baseline     — the all-default cell, which matches bench_defaults
                       (filter=nofuse, project=nofuse, sort=no_skf,
                        group_by=hash, eq_join=hash_gfur). Each segment
                       is the actual mean_ms of that step under this
                       cell.
      * worst-per-step — composite. For each step_index, take
                         max(mean_ms) across configs. Sum the maxes.
      * best-per-step  — composite. For each step_index, take
                         min(mean_ms) across configs. Sum the mins.
                         An oracle lower bound assuming the engine
                         could pick a different hint per step instance.

    Per-step (not per-kind): repeated kinds (e.g. several EqJoins in
    Q9) each pick their own winning/losing config independently.
    """
    queries = sorted(df["query"].unique())
    fig, axes = plt.subplots(2, 11, figsize=(28, 9), sharey=False)
    axes = axes.flatten()

    used_kinds = set()

    BAR_LABELS = ["baseline", "worst", "best"]

    for i, q in enumerate(range(1, 23)):
        ax = axes[i]
        sub = df[df["query"] == q].copy()
        if sub.empty:
            ax.set_title(f"Q{q}\n(no data)", fontsize=14)
            ax.axis("off")
            continue

        step_indices = sorted(sub["step_index"].unique())
        # step_kind per step_index — stable across configs (lowering
        # changes hints, not plan structure).
        step_kinds = {
            si: sub[sub["step_index"] == si]["step_kind"].iloc[0]
            for si in step_indices
        }
        used_kinds.update(step_kinds.values())

        # Baseline cell = all-default cell. The bench's bench_defaults
        # binds default to the requested hints exactly.
        baseline = sub[sub["axis"] == "baseline"]
        if baseline.empty:
            baseline = sub[sub["label"] == "default"]

        bars = {label: [] for label in BAR_LABELS}
        for si in step_indices:
            kind = step_kinds[si]
            step_rows = sub[sub["step_index"] == si]
            b = baseline[baseline["step_index"] == si]
            base_ms = float(b["mean_ms"].iloc[0]) if len(b) else 0.0
            worst_ms = float(step_rows["mean_ms"].max()) if len(step_rows) else 0.0
            best_ms  = float(step_rows["mean_ms"].min()) if len(step_rows) else 0.0
            bars["baseline"].append((kind, base_ms))
            bars["worst"].append((kind, worst_ms))
            bars["best"].append((kind, best_ms))

        # Render 3 bars; stack segments per step_index in plan order.
        x_positions = list(range(len(BAR_LABELS)))
        for x, label in zip(x_positions, BAR_LABELS):
            bottom = 0.0
            for kind, ms in bars[label]:
                ax.bar([x], [ms], bottom=[bottom], width=0.7,
                       color=STEP_COLORS.get(kind, "#aaaaaa"),
                       edgecolor="white", linewidth=0.4)
                bottom += ms

        ax.set_xticks(x_positions)
        ax.set_xticklabels(BAR_LABELS, fontsize=11)
        ax.set_title(f"Q{q}", fontsize=15, fontweight="bold")
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
                color="#bbbbbb", zorder=0)
        ax.set_axisbelow(True)
        if i % 11 == 0:
            ax.set_ylabel("GPU time (ms)", fontsize=13)

    # Shared legend: one entry per used StepKind.
    used_sorted = [k for k in STEP_COLORS if k in used_kinds]
    used_sorted += sorted(used_kinds - set(STEP_COLORS))
    legend_handles = [
        mpatches.Patch(color=STEP_COLORS.get(k, "#aaaaaa"), label=k)
        for k in used_sorted
    ]
    fig.suptitle(
        f"TPC-H per-step breakdown  ·  baseline / worst-per-step / "
        f"best-per-step  ·  SF={scale}  ·  {machine_display}",
        fontsize=17, fontweight="bold", y=0.99)
    # Split the StepKind legend into two roughly-equal rows: ceil(N/2)
    # per row puts e.g. 13 entries as 7 + 6 instead of 10 + 3.
    n_legend = len(legend_handles)
    legend_ncol = (n_legend + 1) // 2
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=legend_ncol,
               bbox_to_anchor=(0.5, -0.01), fontsize=12, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    jpg_path = out_path.with_suffix(".jpg")
    fig.savefig(jpg_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def plot_kind_share(df: pd.DataFrame, out_path: Path, scale: int,
                    machine_display: str):
    """One bar per StepKind that appears in the workload. Each bar's
    height is the % contribution of that kind to the global
    "best-per-step" workload total — i.e. for every (query,
    step_index) we take min(mean_ms) across configs, then group those
    minima by StepKind and sum. Bars are sorted descending by %, all
    kinds shown (no truncation). Bar colour matches the breakdown
    plot's STEP_COLORS palette.
    """
    best_idx = df.groupby(["query", "step_index"])["mean_ms"].idxmin()
    step_best = df.loc[best_idx, ["step_kind", "mean_ms"]].rename(
        columns={"step_kind": "kind", "mean_ms": "best_ms"})

    by_kind = (step_best.groupby("kind")["best_ms"]
               .agg(["sum", "count"])
               .rename(columns={"sum": "total_ms", "count": "n_steps"})
               .reset_index())
    total = float(by_kind["total_ms"].sum())
    by_kind["pct"] = by_kind["total_ms"] / total * 100.0
    by_kind = by_kind.sort_values("pct", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [STEP_COLORS.get(k, "#aaaaaa") for k in by_kind["kind"]]
    bars = ax.bar(range(len(by_kind)), by_kind["pct"], color=colors,
                  edgecolor="white", linewidth=0.6, width=0.72)

    ymax = float(by_kind["pct"].max())
    for b, pct, ms, n in zip(bars, by_kind["pct"], by_kind["total_ms"],
                             by_kind["n_steps"]):
        ax.text(b.get_x() + b.get_width()/2, pct + ymax*0.02,
                f"{pct:.1f}%\n{ms:.1f} ms\n×{int(n)}",
                ha="center", va="bottom", fontsize=10.5, color="#222222",
                linespacing=1.15)

    ax.set_xticks(range(len(by_kind)))
    ax.set_xticklabels(by_kind["kind"], rotation=30, fontsize=12,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("% of total best-per-step time across 22 queries",
                  fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
            color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax * 1.32)

    ax.set_title(f"Time share per operator kind (best cfg per step)  "
                 f"·  SF={scale}  ·  {machine_display}",
                 fontsize=15, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    jpg_path = out_path.with_suffix(".jpg")
    fig.savefig(jpg_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")
    print(f"  {len(by_kind)} kinds · workload best-per-step total = "
          f"{total:.2f} ms · top-3 share = "
          f"{float(by_kind['pct'].head(3).sum()):.1f}%")


# Each per-axis column lists the best cfg value for each step of the
# matching kind. The bench fixture's `sort` axis governs both Sort and
# TopK (they share the sort_table_options hint), so the `sort` column
# rolls in steps of either kind. SemiJoin / AntiJoin are not bench-axis-
# governed (single implementation each) so they don't appear here.
AXIS_TO_KINDS = {
    "filter":   ["Filter"],
    "project":  ["Project"],
    "sort":     ["Sort", "TopK"],
    "group_by": ["GroupBy"],
    "eq_join":  ["EqJoin"],
}

# bench_defaults::* in tpch_ir_nvbench.cu picks one concrete value per
# axis as the "default" sentinel. Expand "default" to that concrete
# value in the per-axis columns — otherwise the CSV would show both
# "default" and the concrete value (e.g. "nofuse") as if they were
# distinct cfgs, when they're really the same physical configuration.
# Keep this dict in sync with `bench_defaults::*` in the bench fixture.
AXIS_DEFAULT_VALUE = {
    "filter":   "nofuse",     # bench_defaults::filter   — fuse_operations = false
    "project":  "nofuse",     # bench_defaults::project  — fuse_operations = false
    "sort":     "no_skf",     # bench_defaults::sort     — smart_keys_fusion = false
    "group_by": "hash",       # bench_defaults::group_by — groupby_algo = HASH
    "eq_join":  "hash_gfur",  # bench_defaults::eq_join  — HASH + GFUR
}


def resolve_axis_value(axis: str, value: str) -> str:
    """Translate the bench's "default" sentinel into its concrete value
    (e.g. filter=default → "nofuse"). Other values pass through."""
    return AXIS_DEFAULT_VALUE[axis] if value == "default" else value


def best_axis_per_step(df_steps_q: pd.DataFrame, axis: str,
                       kinds: list[str]) -> str:
    """For one query's df_steps slice, find the per-step minimum
    mean_ms among rows that vary `axis` (or the baseline cell), keep
    only steps whose step_kind is in `kinds`, and emit a compact
    string of the form "Step 5: fuse; Step 12: nofuse". The numbering
    is the absolute `step_index` in the lowered ExecPlan (so a reader
    can locate the step in the plan dump directly). Returns "" when
    no matching step exists.
    """
    if df_steps_q.empty:
        return ""
    of_kind = df_steps_q[df_steps_q["step_kind"].isin(kinds)]
    if of_kind.empty:
        return ""
    # Only rows where this axis is varied (or the all-default baseline)
    # are valid candidates for "what's the best value of <axis>"; rows
    # that vary an unrelated axis would give a misleadingly default
    # `axis` value.
    relevant = of_kind[of_kind["axis"].isin([axis, "baseline"])]
    if relevant.empty:
        return ""
    parts = []
    for si in sorted(of_kind["step_index"].unique()):
        step_rows = relevant[relevant["step_index"] == si]
        if step_rows.empty:
            continue
        best_row = step_rows.loc[step_rows["mean_ms"].idxmin()]
        parts.append(
            f"Step {int(si)}: {resolve_axis_value(axis, best_row[axis])}")
    return "; ".join(parts)


def write_per_query_summary(df: pd.DataFrame, df_steps: pd.DataFrame,
                            out_path: Path):
    """Per-query e2e summary CSV. One row per query 1..22; queries
    with no data in the input are skipped (no NaN rows).

    Columns:
      query, baseline-cfg, worst-cfg, best-cfg,
      best-vs-worst-speedup, best-vs-baseline-speedup,
      filter, project, sort, group_by, eq_join

    `baseline-cfg` / `worst-cfg` / `best-cfg` hold GPU time in ms
    (e2e from the bench's `Skipped=No` rows — matches the prior
    cross-machine summary tables in this project, naming kept for
    paste-compat). Speedups are `worst/best` and `baseline/best`.

    The five per-axis columns each list the best-performing axis
    value for each step of the matching StepKind, in plan order:
    `"Step 5: fuse; Step 12: nofuse"`. The numbering is the absolute
    `step_index` in the lowered ExecPlan, so a reader can cross-
    reference back to the plan dump directly. Empty string for
    queries / axes with no matching step. The `sort` column rolls in
    both Sort and TopK steps (they share the bench's `sort` axis).
    """
    rows = []
    for q in range(1, 23):
        sub = df[df["query"] == q]
        if sub.empty:
            continue
        baseline = sub[sub["axis"] == "baseline"]
        if baseline.empty:
            baseline = sub[sub["label"] == "default"]
        base_ms  = (float(baseline["gpu_ms"].iloc[0])
                    if len(baseline) else float("nan"))
        worst_ms = float(sub["gpu_ms"].max())
        best_ms  = float(sub["gpu_ms"].min())

        steps_q = df_steps[df_steps["query"] == q]
        per_axis = {axis: best_axis_per_step(steps_q, axis, kinds)
                    for axis, kinds in AXIS_TO_KINDS.items()}

        rows.append({
            "query":                    q,
            "baseline-cfg":             round(base_ms, 3),
            "worst-cfg":                round(worst_ms, 3),
            "best-cfg":                 round(best_ms, 3),
            "best-vs-worst-speedup":    (round(worst_ms / best_ms, 3)
                                         if best_ms > 0 else ""),
            "best-vs-baseline-speedup": (round(base_ms / best_ms, 3)
                                         if best_ms > 0 and base_ms == base_ms
                                         else ""),
            **per_axis,
        })
    out = pd.DataFrame(rows, columns=[
        "query", "baseline-cfg", "worst-cfg", "best-cfg",
        "best-vs-worst-speedup", "best-vs-baseline-speedup",
        "filter", "project", "sort", "group_by", "eq_join"])
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


def plot_top_steps(df: pd.DataFrame, out_path: Path, scale: int,
                   machine_display: str, top_n: int = 8):
    """Top-N most expensive single steps across the entire 22-query
    workload, assuming each step picks its own best configuration
    (per-step min). Each bar's height is the % contribution of that
    step to the global "best-per-step" workload total. Bar colour
    encodes StepKind. Bar label encodes (query, kind, rank) so a
    reader can identify the step uniquely. Annotation above each bar
    includes the picked configuration (which axis hint won for that
    step).
    """
    # Per-(query, step_index) best row — pick the config that
    # achieved the minimum mean_ms.
    best_idx = df.groupby(["query", "step_index"])["mean_ms"].idxmin()
    step_best = df.loc[best_idx, [
        "query", "step_index", "step_kind", "mean_ms",
        "filter", "project", "sort", "group_by", "eq_join",
    ]].rename(columns={"step_kind": "kind", "mean_ms": "best_ms"}).reset_index(drop=True)

    # Rank repeated kinds within each query (1-based) so we can
    # disambiguate e.g. Q9's five EqJoins.
    step_best = step_best.sort_values(["query", "step_index"]).reset_index(drop=True)
    step_best["kind_rank"] = step_best.groupby(["query", "kind"]).cumcount() + 1
    step_best["kind_count"] = step_best.groupby(["query", "kind"])["kind"].transform("count")

    def label(r):
        q, k, r1, n = int(r["query"]), r["kind"], int(r["kind_rank"]), int(r["kind_count"])
        return f"Q{q}·{k}" if n == 1 else f"Q{q}·{k}#{r1}"
    step_best["label"] = step_best.apply(label, axis=1)

    # Picked-cfg label: which single axis (if any) was non-default-non-na.
    def cfg_label(r):
        AXES_LOCAL = ["filter", "project", "sort", "group_by", "eq_join"]
        set_axes = [a for a in AXES_LOCAL if r[a] not in ("default", "n/a")]
        if not set_axes:
            return "baseline"
        # Use the same compact value formatting as elsewhere.
        a = set_axes[0]
        v = VALUE_SHORT.get(r[a], r[a])
        return f"{AXIS_SHORT[a]}={v}"
    step_best["cfg"] = step_best.apply(cfg_label, axis=1)

    total = float(step_best["best_ms"].sum())
    step_best["pct"] = step_best["best_ms"] / total * 100.0

    top = step_best.nlargest(top_n, "pct").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [STEP_COLORS.get(k, "#aaaaaa") for k in top["kind"]]
    bars = ax.bar(range(len(top)), top["pct"], color=colors,
                  edgecolor="white", linewidth=0.6, width=0.72)

    # Annotation above each bar — % / ms / picked cfg (axis=value).
    ymax = float(top["pct"].max())
    for b, pct, ms, cfg in zip(bars, top["pct"], top["best_ms"], top["cfg"]):
        ax.text(b.get_x() + b.get_width()/2, pct + ymax*0.02,
                f"{pct:.1f}%\n{ms:.1f} ms\n[{cfg}]",
                ha="center", va="bottom", fontsize=10.5, color="#222222",
                linespacing=1.15)

    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["label"], rotation=30, fontsize=12,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("% of total best-per-step time across 22 queries",
                  fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
            color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax * 1.32)

    ax.set_title(f"Top-{top_n} most expensive steps (best cfg per step)  "
                 f"·  SF={scale}  ·  {machine_display}",
                 fontsize=15, fontweight="bold", pad=12)

    # Legend: only the kinds actually appearing in top-N.
    kinds_in_plot = list(dict.fromkeys(top["kind"].tolist()))
    legend_handles = [
        mpatches.Patch(color=STEP_COLORS.get(k, "#aaaaaa"), label=k)
        for k in kinds_in_plot
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=11, frameon=False, ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    jpg_path = out_path.with_suffix(".jpg")
    fig.savefig(jpg_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")
    print(f"  workload best-per-step total = {total:.2f} ms across "
          f"{len(step_best)} steps; top-{top_n} sum = "
          f"{float(top['best_ms'].sum()):.2f} ms "
          f"({float(top['pct'].sum()):.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--machine", required=True,
                   help="machine identifier (subdir name under "
                        "<data-dir> and <out>); known names get a "
                        "pretty title, unknown ones use the raw value")
    p.add_argument("--date", required=True,
                   help="run date as YYYYMMDD; matches the dated "
                        "folder produced by run_tpch_perf.sh")
    p.add_argument("--tag", required=True,
                   help="run tag (the --tag passed to "
                        "run_tpch_perf.sh)")
    p.add_argument("--scale", required=True, type=int,
                   help="scale factor (must match the CSVs)")
    p.add_argument("--data-dir", default="tools/results/data",
                   help="base dir holding "
                        "<machine>/tpch/<date>/<tag>/bench_{e2e,steps}_sf<N>.csv "
                        "(default: tools/results/data)")
    p.add_argument("--out", default="tools/results/pdf/tpch_bench",
                   help="base output dir; figures land in "
                        "<out>/<machine>/<date>/<tag>/ (default: "
                        "tools/results/pdf/tpch_bench)")
    args = p.parse_args()

    machine = args.machine
    date = args.date
    tag = args.tag
    scale = int(args.scale)
    display = MACHINE_NAMES.get(machine, machine)

    csv_dir = Path(args.data_dir) / machine / "tpch" / date / tag
    e2e_csv = csv_dir / f"bench_e2e_sf{scale}.csv"
    steps_csv = csv_dir / f"bench_steps_sf{scale}.csv"
    if not e2e_csv.exists():
        sys.exit(f"error: missing e2e CSV: {e2e_csv}")
    if not steps_csv.exists():
        sys.exit(f"error: missing steps CSV: {steps_csv}")

    df_e2e = load_e2e(str(e2e_csv))
    df_steps = load_steps(str(steps_csv))

    if df_e2e.empty:
        print("error: e2e CSV has no non-skipped rows", file=sys.stderr)
        sys.exit(1)
    df_e2e = df_e2e[df_e2e["scale"] == scale]
    df_steps = df_steps[df_steps["scale"] == scale]
    if df_e2e.empty:
        sys.exit(f"error: no rows at scale={scale} in {e2e_csv}")

    out_dir = Path(args.out) / machine / date / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_e2e(df_e2e, out_dir / f"tpch_e2e_sf{scale}.pdf", scale, display)
    plot_breakdown(df_steps, out_dir / f"tpch_breakdown_sf{scale}.pdf",
                   scale, display)
    plot_top_steps(df_steps, out_dir / f"tpch_top_steps_sf{scale}.pdf",
                   scale, display)
    plot_kind_share(df_steps, out_dir / f"tpch_kind_share_sf{scale}.pdf",
                    scale, display)
    # Per-query summary CSV — same numbers as the e2e plot, in a
    # paste-into-Sheets shape. Lands in the data dir (alongside the
    # bench CSVs) rather than the PDF dir; it's a CSV.
    write_per_query_summary(
        df_e2e, df_steps,
        csv_dir / f"per_query_summary_sf{scale}.csv")


if __name__ == "__main__":
    main()
