#!/usr/bin/env python3
"""Plot benchmark_tpch_ir results.

Inputs:
  --e2e   : NVBench --csv output (per-cell GPU Time)
  --steps : GAMAX_BENCH_STEP_CSV output (per-step mean_ms breakdown)

Two figures:
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

AXES = ["filter", "project", "sort", "group_by", "eq_join"]
NA = "n/a"
DEFAULT = "default"

# Stable colour palette per StepKind. Step kinds beyond this list fall
# back to matplotlib's default cycle.
STEP_COLORS = {
    "Source":     "#888888",
    "Filter":     "#1f77b4",
    "Project":    "#ff7f0e",
    "Reorder":    "#9467bd",
    "Sort":       "#8c564b",
    "TopK":       "#e377c2",
    "GroupBy":    "#2ca02c",
    "ScalarAgg":  "#bcbd22",
    "EqJoin":     "#d62728",
    "SemiJoin":   "#17becf",
    "AntiJoin":   "#7f7f7f",
    "Distinct":   "#aec7e8",
    "Concat":     "#ffbb78",
    "CrossJoin":  "#98df8a",
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


AXIS_PALETTE = {
    "baseline":  "#444444",
    "filter":    "#1f77b4",
    "project":   "#ff7f0e",
    "sort":      "#8c564b",
    "group_by":  "#2ca02c",
    "eq_join":   "#d62728",
}

def axis_color(axis: str) -> str:
    """Bar background colour for the e2e figure: one colour per axis
    family so the eye groups bars by which operator is being varied."""
    return AXIS_PALETTE.get(axis, "#666666")


def plot_e2e(df: pd.DataFrame, out_path: Path, scale: int):
    """22 subfigures, one bar per cell, coloured by which axis is set."""
    queries = sorted(df["query"].unique())
    fig, axes = plt.subplots(2, 11, figsize=(44, 11), sharey=False)
    axes = axes.flatten()

    for i, q in enumerate(range(1, 23)):
        ax = axes[i]
        sub = df[df["query"] == q].copy()
        if sub.empty:
            ax.set_title(f"Q{q} (no data)", fontsize=10)
            ax.axis("off")
            continue
        sub = sub.sort_values("sort_key")
        labels = sub["label"].tolist()
        values = sub["gpu_ms"].tolist()
        colors = [axis_color(a) for a in sub["axis"].tolist()]
        ax.bar(range(len(labels)), values, color=colors, edgecolor="black",
               linewidth=0.3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=70, fontsize=6.5,
                           ha="right", rotation_mode="anchor")
        ax.set_title(f"Q{q}", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6, zorder=0)
        if i % 11 == 0:
            ax.set_ylabel("GPU time (ms)", fontsize=9)

    # Shared legend by axis-family colour.
    legend_handles = [
        mpatches.Patch(color="#444444", label="baseline (all default)"),
        mpatches.Patch(color="#1f77b4", label="filter=*"),
        mpatches.Patch(color="#ff7f0e", label="project=*"),
        mpatches.Patch(color="#8c564b", label="sort=*"),
        mpatches.Patch(color="#2ca02c", label="group_by=*"),
        mpatches.Patch(color="#d62728", label="eq_join=*"),
    ]
    fig.suptitle(f"TPC-H per-query end-to-end GPU time, SF={scale}, "
                 f"H100 NVL, UserSpaceMM",
                 fontsize=13)
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.01), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    # Also save a JPG sibling for quick inspection.
    jpg_path = out_path.with_suffix(".jpg")
    fig.savefig(jpg_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def plot_breakdown(df: pd.DataFrame, out_path: Path, scale: int):
    """22 subfigures, stacked bars: each step contributes one segment.
    An extra "best-of-each-step" bar is appended per query, composed of
    the per-step minima across configs.

    Step ordering uses (step_index) within a config — kinds may repeat
    so we colour by step_kind but keep the index for stable ordering.
    """
    queries = sorted(df["query"].unique())
    fig, axes = plt.subplots(2, 11, figsize=(44, 11), sharey=False)
    axes = axes.flatten()

    used_kinds = set()

    for i, q in enumerate(range(1, 23)):
        ax = axes[i]
        sub = df[df["query"] == q].copy()
        if sub.empty:
            ax.set_title(f"Q{q} (no data)", fontsize=10)
            ax.axis("off")
            continue

        # Per-config sorted list.
        configs = (sub[["label", "axis", "sort_key"]]
                   .drop_duplicates()
                   .sort_values("sort_key"))
        labels = configs["label"].tolist()

        # Per-step matrix: rows = configs, cols = step_index in
        # plan-natural order. Cells = mean_ms (NaN if missing).
        step_indices = sorted(sub["step_index"].unique())
        # step_kind label per step_index — take the first observed value.
        step_kinds = {
            si: sub[sub["step_index"] == si]["step_kind"].iloc[0]
            for si in step_indices
        }
        used_kinds.update(step_kinds.values())

        # Build per-config stack. Order by step_index ascending (matches
        # plan-natural order in the executor).
        bottom = np.zeros(len(labels))
        for si in step_indices:
            kind = step_kinds[si]
            colour = STEP_COLORS.get(kind, None)
            heights = []
            for lbl in labels:
                cell = sub[(sub["label"] == lbl) & (sub["step_index"] == si)]
                heights.append(float(cell["mean_ms"].iloc[0]) if len(cell) else 0.0)
            ax.bar(range(len(labels)), heights, bottom=bottom,
                   color=colour, edgecolor="black", linewidth=0.2)
            bottom += np.array(heights)

        # Best-of-each-step composite bar. For each step_index, take
        # the min mean_ms across configs that exercised that step.
        best_total_by_kind = {}
        bottom_best = 0.0
        x_best = len(labels)
        for si in step_indices:
            kind = step_kinds[si]
            heights_for_step = sub[sub["step_index"] == si]["mean_ms"]
            if heights_for_step.empty:
                continue
            best = float(heights_for_step.min())
            ax.bar([x_best], [best], bottom=[bottom_best],
                   color=STEP_COLORS.get(kind, None),
                   edgecolor="black", linewidth=0.2)
            bottom_best += best
            best_total_by_kind[kind] = best_total_by_kind.get(kind, 0.0) + best
        # Mark the composite bar with a thicker outline.
        if x_best == len(labels):
            ax.bar([x_best], [bottom_best], bottom=[0.0],
                   facecolor="none", edgecolor="black", linewidth=1.5,
                   linestyle="--")

        all_labels = labels + ["best-of-step"]
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=70, fontsize=6.5,
                           ha="right", rotation_mode="anchor")
        ax.set_title(f"Q{q}", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6, zorder=0)
        if i % 11 == 0:
            ax.set_ylabel("GPU time (ms)", fontsize=9)

    # Shared legend: one entry per used StepKind.
    used_sorted = [k for k in STEP_COLORS if k in used_kinds]
    used_sorted += sorted(used_kinds - set(STEP_COLORS))
    legend_handles = [
        mpatches.Patch(color=STEP_COLORS.get(k, "#aaaaaa"), label=k)
        for k in used_sorted
    ]
    legend_handles.append(
        mpatches.Patch(facecolor="white", edgecolor="black", linestyle="--",
                       label="best-of-step composite (dashed outline)")
    )
    fig.suptitle(f"TPC-H per-step breakdown, SF={scale}, H100 NVL, "
                 f"UserSpaceMM",
                 fontsize=13)
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(8, len(legend_handles)),
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    jpg_path = out_path.with_suffix(".jpg")
    fig.savefig(jpg_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e2e",   required=True, help="NVBench --csv output")
    p.add_argument("--steps", required=True, help="per-step CSV from "
                   "GAMAX_BENCH_STEP_CSV")
    p.add_argument("--out",   default="tools/results/pdf/tpch_bench",
                   help="output directory (figures named tpch_e2e.pdf "
                        "and tpch_breakdown.pdf)")
    args = p.parse_args()

    df_e2e = load_e2e(args.e2e)
    df_steps = load_steps(args.steps)

    if df_e2e.empty:
        print("error: e2e CSV has no non-skipped rows", file=sys.stderr)
        sys.exit(1)
    scales = sorted(df_e2e["scale"].unique())
    if len(scales) != 1:
        # The plot title is per-scale; if multiple scales are present,
        # caller should slice the CSV. Pick the first and warn.
        print(f"note: multiple scales present {scales}; using scale={scales[0]}",
              file=sys.stderr)
    scale = int(scales[0])
    df_e2e = df_e2e[df_e2e["scale"] == scale]
    df_steps = df_steps[df_steps["scale"] == scale]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_e2e(df_e2e, out_dir / f"tpch_e2e_sf{scale}.pdf", scale)
    plot_breakdown(df_steps, out_dir / f"tpch_breakdown_sf{scale}.pdf", scale)


if __name__ == "__main__":
    main()
