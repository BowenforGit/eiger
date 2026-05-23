#!/usr/bin/env python3
"""Plot benchmark_tpch_ir results.

Inputs are derived from --machine, --date, --tag, --scale by
convention (matching `script/run_tpch_perf.sh`'s output layout):
  e2e CSV   : <data-dir>/<machine>/tpch/<date>/<tag>/bench_e2e_sf<N>.csv
  steps CSV : <data-dir>/<machine>/tpch/<date>/<tag>/bench_steps_sf<N>.csv

Outputs land in <out>/<machine>/<date>/<tag>/.

Plot semantics decoupled from the bench:
  * On CSV load every axis value "default" is resolved to its concrete
    nvbench-default name via AXIS_DEFAULT_VALUE — after this step the
    "default" sentinel is gone and the plot operates on concrete
    implementation names throughout.
  * The plot defines its own baseline cfg via PLOT_BASELINE. The
    per-step baseline is the matching step's time from the cell that
    sets axis A to PLOT_BASELINE[A] (a "one-hot" cell under the
    at-most-one-set bench sweep). The e2e baseline is the sum of
    per-step baselines — a synthetic composite, not a single measured
    cell.
  * "Best" per step = the alternative axis value (≠ baseline) with the
    smallest step time, but only if it beats baseline by at least
    BEST_VS_BASELINE_THRESHOLD. Otherwise best == baseline. This
    filters out apparent wins that are within measurement noise and
    also rejects "same impl as baseline, different cell occurrence"
    by construction (the candidate set excludes cells where the
    step-matching axis matches the plot baseline).
  * The e2e "best" applies the same threshold against the synth e2e
    baseline.

Outputs (per scale):
  * tpch_e2e         : 22 subfigures, 2 bars per query (baseline / best)
  * tpch_breakdown   : 22 subfigures, 2 stacked bars per query (same)
  * tpch_top_steps   : top-N most-expensive single steps under best
  * tpch_kind_share  : % time per StepKind under best
  * per_query_summary CSV : baseline / best / speedup + per-axis best
                            step choices
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

# Axis list — ordered as they appear in the CSV header. `distinct`
# sits between `group_by` and `eq_join` to match the bench harness.
AXES = ["filter", "project", "sort", "group_by", "distinct", "eq_join"]
NA = "n/a"
DEFAULT = "default"

# Display names per machine identifier.
MACHINE_NAMES = {
    "h100":  "H100 NVL",
    "gh200": "Grace Hopper GH200",
}

# Muted, paper-friendly palette per StepKind.
STEP_COLORS = {
    "Source":     "#bab0ac",
    "Filter":     "#4e79a7",
    "Project":    "#f28e2b",
    "Reorder":    "#bab0ac",
    "Sort":       "#9c755f",
    "TopK":       "#d37295",
    "GroupBy":    "#59a14f",
    "ScalarAgg":  "#edc948",
    "EqJoin":     "#e15759",
    "SemiJoin":   "#76b7b2",
    "AntiJoin":   "#b07aa1",
    "Distinct":   "#a0cbe8",
    "Concat":     "#ffbe7d",
    "CrossJoin":  "#8cd17d",
}

AXIS_SHORT = {
    "filter":   "fil",
    "project":  "prj",
    "sort":     "srt",
    "group_by": "grp",
    "distinct": "dst",
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
    # distinct algos
    "hash_skf":            "hash.skf",
    "sort_skf":            "sort.skf",
    # "sort" passes through as-is
}

# Axis-family palette (mirrors STEP_COLORS so the e2e bars and the
# kind-share bars use the same colour as the stack segment for the
# operator they vary).
AXIS_PALETTE = {
    "baseline":  "#7f7f7f",
    "filter":    "#4e79a7",
    "project":   "#f28e2b",
    "sort":      "#9c755f",
    "group_by":  "#59a14f",
    "distinct":  "#a0cbe8",
    "eq_join":   "#e15759",
}

# bench_defaults::* in tpch_ir_nvbench.cu — concrete value the
# nvbench "default" sentinel resolves to. Keep this dict in sync
# with `bench_defaults::*` in the bench fixture.
AXIS_DEFAULT_VALUE = {
    "filter":   "nofuse",
    "project":  "nofuse",
    "sort":     "no_skf",
    "group_by": "hash",
    "distinct": "hash",
    "eq_join":  "hash_gfur",
}

# Which StepKinds a given bench axis controls.
AXIS_TO_KINDS = {
    "filter":   ["Filter"],
    "project":  ["Project"],
    "sort":     ["Sort", "TopK"],
    "group_by": ["GroupBy"],
    "distinct": ["Distinct"],
    "eq_join":  ["EqJoin"],
}

# Plot-side baseline cfg. Decoupled from the nvbench defaults.
# Filter and project flip from nvbench `nofuse` to `fuse` — a paper-
# friendly default that wins ~everywhere and is a codegen choice
# rather than an algorithm choice. Other axes stay at nvbench
# defaults (which match the engine's auto path), so the plot's
# baseline reads as "engine default with the obvious codegen wins
# turned on."
PLOT_BASELINE = {
    "filter":   "fuse",
    "project":  "fuse",
    "sort":     "no_skf",
    "group_by": "hash",
    "distinct": "hash",
    "eq_join":  "hash_gfur",
}

# Best-cfg gate: a non-baseline cfg is "promoted" to best only when
# it's at least this many percent faster than baseline. Otherwise
# best == baseline (no winner clears the noise floor).
BEST_VS_BASELINE_THRESHOLD = 0.05  # 5%

# Font sizes for paper insertion. Single dict so any downstream
# tweak is one edit. Calibrated for a two-column paper at ~3.4
# in/column; bump uniformly if inserting at single-column width.
FONT_SIZES = {
    "subplot_title":  18,
    "xticks":         14,
    "yticks":         13,
    "ylabel":         15,
    "suptitle":       20,
    "legend":         14,
    "annotation":     12,
    "section_title":  18,   # used by plot_top_steps / plot_kind_share
}

# Two-bar palette: gray for the baseline composite, green for the
# gated best. When best == baseline (no winner), the bars are the
# same height by construction (the bench harness's noise floor wasn't
# crossed); the green bar reads as "no measured single-axis variation
# beats baseline by ≥5%."
TWO_BAR_PALETTE = {
    "baseline": "#7f7f7f",
    "best":     "#59a14f",
}


# ── Axis normalization + helpers ──────────────────────────────────

def resolve_axis_value(axis: str, value: str) -> str:
    """Translate the bench's "default" sentinel into its concrete
    value (e.g. filter=default → "nofuse")."""
    if value == DEFAULT:
        return AXIS_DEFAULT_VALUE[axis]
    return value


def normalize_axes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 'default' in each axis column with its concrete
    nvbench-default value. 'n/a' passes through. After this step the
    dataframe holds only concrete implementation names (or 'n/a' for
    kinds absent from the plan) — the "default" sentinel is gone and
    downstream code works against concrete values throughout."""
    for a in AXES:
        if a in df.columns:
            df[a] = df[a].apply(
                lambda v, ax=a: resolve_axis_value(ax, v) if v != NA else v)
    return df


def _is_all_default(row: pd.Series) -> bool:
    """True when every axis on this row is at its nvbench-default
    concrete value (or 'n/a' for kinds absent from the query's plan).
    Operates on normalized rows (no 'default' sentinel)."""
    for a in AXES:
        if a not in row:
            continue
        v = row[a]
        if v == NA:
            continue
        if v != AXIS_DEFAULT_VALUE[a]:
            return False
    return True


def short_label(row: pd.Series) -> str:
    """Compact label for a config: which single axis (if any) is at a
    non-nvbench-default value, else 'default'. Operates on normalized
    axes."""
    set_axes = []
    for a in AXES:
        if a not in row:
            continue
        v = row[a]
        if v == NA:
            continue
        if v != AXIS_DEFAULT_VALUE[a]:
            set_axes.append(a)
    if not set_axes:
        return "default"
    if len(set_axes) == 1:
        a = set_axes[0]
        v = VALUE_SHORT.get(row[a], row[a])
        return f"{AXIS_SHORT[a]}.{v}"
    return ",".join(f"{AXIS_SHORT[a]}.{VALUE_SHORT.get(row[a], row[a])}"
                     for a in set_axes)


def axis_for(row: pd.Series) -> str:
    """The single axis varied from nvbench default for this row, or
    'baseline' if none varied. Operates on normalized axes."""
    set_axes = []
    for a in AXES:
        if a not in row:
            continue
        v = row[a]
        if v == NA:
            continue
        if v != AXIS_DEFAULT_VALUE[a]:
            set_axes.append(a)
    return set_axes[0] if len(set_axes) == 1 else "baseline"


def order_key(row: pd.Series):
    """Sort cells: baseline first, then grouped by axis (filter,
    project, sort, group_by, distinct, eq_join), then by hint value
    alphabetically. Operates on normalized axes."""
    set_axes = []
    for a in AXES:
        if a not in row:
            continue
        v = row[a]
        if v == NA:
            continue
        if v != AXIS_DEFAULT_VALUE[a]:
            set_axes.append(a)
    if not set_axes:
        return (-1, "", "")
    a = set_axes[0]
    return (AXES.index(a), a, row[a])


# ── CSV loading ───────────────────────────────────────────────────

# pandas would parse the literal "n/a" as NaN by default; the axis
# sentinel must round-trip exactly. Pass keep_default_na=False so the
# strings survive.
def _inject_missing_axes(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compat: older CSVs may predate the `distinct` axis (or
    any future axis) — inject any missing columns as 'n/a' so the
    rest of the script can assume the full AXES list is present."""
    for a in AXES:
        if a not in df.columns:
            df[a] = NA
    return df


def load_e2e(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df = df[df["Skipped"] == "No"].copy()
    df = _inject_missing_axes(df)
    df["query"] = df["query"].astype(int)
    df["scale"] = df["scale"].astype(int)
    df["gpu_ms"] = df["GPU Time (sec)"].astype(float) * 1000.0
    df = normalize_axes(df)
    df["label"] = df.apply(short_label, axis=1)
    df["axis"] = df.apply(axis_for, axis=1)
    df["sort_key"] = df.apply(order_key, axis=1)
    return df


def load_steps(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df = _inject_missing_axes(df)
    df["query"] = df["query"].astype(int)
    df["scale"] = df["scale"].astype(int)
    df["mean_ms"] = df["mean_ms"].astype(float)
    df["min_ms"]  = df["min_ms"].astype(float)
    df = normalize_axes(df)
    df["label"]   = df.apply(short_label, axis=1)
    df["axis"]    = df.apply(axis_for, axis=1)
    df["sort_key"] = df.apply(order_key, axis=1)
    return df


# ── Per-axis-value cell lookup ────────────────────────────────────

def _matches_one_hot_for(row: pd.Series, axis: str, value: str) -> bool:
    """True iff `row` is the one-hot cell that sets `axis` to `value`
    and leaves every other axis at its nvbench default (or 'n/a' when
    the kind is absent from the query's plan).

    Under the at-most-one-set bench sweep there is exactly one such
    row per (query, axis, value) tuple — or zero if the value never
    appeared in the sweep (axis is 'n/a' for queries without the
    step kind, or the cell threw at preflight)."""
    if row[axis] != value:
        return False
    for other in AXES:
        if other == axis:
            continue
        v = row[other]
        if v != NA and v != AXIS_DEFAULT_VALUE[other]:
            return False
    return True


def find_one_hot_step_rows(df_steps_q: pd.DataFrame,
                           axis: str, value: str) -> pd.DataFrame:
    """All step rows belonging to the one-hot cell for `axis=value`.
    Empty df if no such cell exists for this query (e.g. axis is
    'n/a' for that query's plan)."""
    mask = df_steps_q.apply(
        lambda r: _matches_one_hot_for(r, axis, value), axis=1)
    return df_steps_q[mask]


def find_all_default_step_rows(df_steps_q: pd.DataFrame) -> pd.DataFrame:
    """Rows of the all-default cell — every axis at nvbench default
    (or 'n/a'). Used to enumerate the plan's step structure."""
    mask = df_steps_q.apply(_is_all_default, axis=1)
    return df_steps_q[mask]


# ── Per-step gated best ───────────────────────────────────────────

# `axis` for a StepKind — the bench axis that controls its
# implementation. None for kinds that don't have a bench axis
# (Reorder, ScalarAgg, SemiJoin, AntiJoin, CrossJoin, Concat,
# SourceTable) — those steps don't vary across cells, so their bar
# segment is the same for baseline and best.
_KIND_TO_AXIS = {}
for _axis, _kinds in AXIS_TO_KINDS.items():
    for _k in _kinds:
        _KIND_TO_AXIS[_k] = _axis


def per_step_baseline_ms(df_steps_q: pd.DataFrame, axis: str,
                         step_index: int) -> float:
    """Plot-baseline time for one step under `axis`'s plot-baseline
    value. NaN if no matching cell exists."""
    rows = find_one_hot_step_rows(df_steps_q, axis, PLOT_BASELINE[axis])
    step_rows = rows[rows["step_index"] == step_index]
    if len(step_rows) == 0:
        return float("nan")
    return float(step_rows["mean_ms"].iloc[0])


def gated_best_per_step(df_steps_q: pd.DataFrame, axis: str,
                        step_index: int):
    """Gated best for one (axis, step_index).

    Returns (baseline_ms, best_ms, best_value) where best_value is
    the alternative axis value that won, or PLOT_BASELINE[axis] when
    baseline holds.

    A non-baseline value wins iff it's the fastest of all distinct
    alternatives AND beats baseline by at least
    BEST_VS_BASELINE_THRESHOLD. Else best == baseline (i.e. the same
    implementation as baseline wins or no alternative clears the
    noise floor)."""
    baseline_value = PLOT_BASELINE[axis]
    baseline_ms = per_step_baseline_ms(df_steps_q, axis, step_index)
    if not (baseline_ms == baseline_ms):  # NaN
        return None, None, None

    distinct_values = df_steps_q[axis].unique()
    candidates = []
    for v in distinct_values:
        if v == baseline_value or v == NA:
            continue
        rows = find_one_hot_step_rows(df_steps_q, axis, v)
        step_rows = rows[rows["step_index"] == step_index]
        if len(step_rows) == 0:
            continue
        cand_ms = float(step_rows["mean_ms"].iloc[0])
        candidates.append((v, cand_ms))
    if not candidates:
        return baseline_ms, baseline_ms, baseline_value

    best_v, best_alt_ms = min(candidates, key=lambda p: p[1])
    if best_alt_ms < baseline_ms * (1.0 - BEST_VS_BASELINE_THRESHOLD):
        return baseline_ms, best_alt_ms, best_v
    return baseline_ms, baseline_ms, baseline_value


def compute_gated_best_per_step(df_steps: pd.DataFrame) -> pd.DataFrame:
    """Per-query, per-step gated best. Returns a DataFrame keyed by
    (query, step_index) with columns: kind, axis, baseline_ms,
    best_ms, best_value. For step kinds not on a bench axis, both
    baseline_ms and best_ms equal the all-default cell's time (the
    step doesn't vary across cells under the at-most-one-set rule)."""
    rows = []
    for q in sorted(df_steps["query"].unique()):
        sub = df_steps[df_steps["query"] == q]
        ref_rows = find_all_default_step_rows(sub)
        if len(ref_rows) == 0:
            ref_rows = sub
        for si in sorted(ref_rows["step_index"].unique()):
            step_row = ref_rows[ref_rows["step_index"] == si]
            if len(step_row) == 0:
                continue
            kind = step_row["step_kind"].iloc[0]
            axis = _KIND_TO_AXIS.get(kind)
            ms_default = float(step_row["mean_ms"].iloc[0])
            if axis is None:
                rows.append({
                    "query": q, "step_index": si, "kind": kind,
                    "axis": None,
                    "baseline_ms": ms_default, "best_ms": ms_default,
                    "best_value": "baseline",
                })
                continue
            base_ms, best_ms, best_val = gated_best_per_step(sub, axis, si)
            if base_ms is None:
                base_ms = ms_default
            if best_ms is None:
                best_ms = base_ms
            rows.append({
                "query": q, "step_index": si, "kind": kind,
                "axis": axis,
                "baseline_ms": base_ms, "best_ms": best_ms,
                "best_value": (best_val if best_val is not None
                               else PLOT_BASELINE[axis]),
            })
    return pd.DataFrame(rows)


# ── E2E baseline synthesis ────────────────────────────────────────

def e2e_baseline_synth_ms(df_steps_q: pd.DataFrame) -> float:
    """Plot-baseline e2e time for one query = sum of per-step
    baselines. The all-default cell's step set defines the plan's
    structure; per-step baseline pulls each step's time from the cell
    that matches plot baseline on that step's axis. For non-axis-
    governed step kinds, falls back to the all-default cell's time."""
    ref_rows = find_all_default_step_rows(df_steps_q)
    if len(ref_rows) == 0:
        ref_rows = df_steps_q

    total = 0.0
    for si in sorted(ref_rows["step_index"].unique()):
        step_row = ref_rows[ref_rows["step_index"] == si]
        if len(step_row) == 0:
            continue
        kind = step_row["step_kind"].iloc[0]
        axis = _KIND_TO_AXIS.get(kind)
        ms_default = float(step_row["mean_ms"].iloc[0])
        if axis is None:
            total += ms_default
            continue
        ms = per_step_baseline_ms(df_steps_q, axis, si)
        if ms != ms:  # NaN — plot-baseline cell missing
            total += ms_default
        else:
            total += ms
    return total


# ── Plots ─────────────────────────────────────────────────────────

def plot_e2e(df: pd.DataFrame, df_steps: pd.DataFrame, out_path: Path,
             scale: int, machine_display: str):
    """22 subfigures, 2 bars per query: baseline (plot-side composite
    of per-step baselines) and best (gated min across all measured
    cells; equals baseline when no cell beats it by ≥5%)."""
    fig, axes = plt.subplots(2, 11, figsize=(28, 9), sharey=False)
    axes = axes.flatten()

    BAR_LABELS = ["baseline", "best"]
    BAR_COLORS = [TWO_BAR_PALETTE[k] for k in BAR_LABELS]

    for i, q in enumerate(range(1, 23)):
        ax = axes[i]
        sub = df[df["query"] == q].copy()
        sub_steps = df_steps[df_steps["query"] == q]
        if sub.empty:
            ax.set_title(f"Q{q}\n(no data)",
                         fontsize=FONT_SIZES["subplot_title"])
            ax.axis("off")
            continue

        base_ms = e2e_baseline_synth_ms(sub_steps)
        best_cell_ms = float(sub["gpu_ms"].min())
        if best_cell_ms < base_ms * (1.0 - BEST_VS_BASELINE_THRESHOLD):
            best_ms = best_cell_ms
        else:
            best_ms = base_ms
        values = [base_ms, best_ms]

        bars = ax.bar(range(2), values, color=BAR_COLORS, width=0.6,
                      edgecolor="white", linewidth=0.4)

        ymax = max(values) if max(values) > 0 else 1.0
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width()/2, v + ymax*0.025,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=FONT_SIZES["annotation"], color="#222222")

        ax.set_xticks(range(2))
        ax.set_xticklabels(BAR_LABELS, fontsize=FONT_SIZES["xticks"])
        ax.set_title(f"Q{q}",
                     fontsize=FONT_SIZES["subplot_title"],
                     fontweight="bold")
        ax.tick_params(axis="y", labelsize=FONT_SIZES["yticks"])
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
                color="#bbbbbb", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ymax * 1.2)
        if i % 11 == 0:
            ax.set_ylabel("GPU time (ms)",
                          fontsize=FONT_SIZES["ylabel"])

    legend_handles = [
        mpatches.Patch(color=TWO_BAR_PALETTE["baseline"],
                       label="baseline (plot-defined defaults)"),
        mpatches.Patch(color=TWO_BAR_PALETTE["best"],
                       label=f"best single-axis variation "
                             f"(≥{int(BEST_VS_BASELINE_THRESHOLD*100)}% "
                             f"faster than baseline)"),
    ]
    fig.suptitle(f"TPC-H per-query end-to-end GPU time  ·  "
                 f"baseline / best  ·  SF={scale}  ·  {machine_display}",
                 fontsize=FONT_SIZES["suptitle"],
                 fontweight="bold", y=0.99)
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.005),
               fontsize=FONT_SIZES["legend"], frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    jpg_path = out_path.with_suffix(".jpg")
    fig.savefig(jpg_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def plot_breakdown(df: pd.DataFrame, out_path: Path, scale: int,
                   machine_display: str):
    """22 subfigures, 2 stacked bars per query:

      * baseline      — per-step baseline times (each step pulled from
                        the cell where its matching axis = plot
                        baseline value). For step kinds not on a bench
                        axis, the all-default cell's time is used.
      * best-per-step — composite. Each step contributes either its
                        gated best (the fastest alternative impl, if
                        ≥5% faster than baseline) or the per-step
                        baseline (if no alternative clears the gate)."""
    fig, axes = plt.subplots(2, 11, figsize=(28, 9), sharey=False)
    axes = axes.flatten()

    used_kinds = set()

    BAR_LABELS = ["baseline", "best"]

    for i, q in enumerate(range(1, 23)):
        ax = axes[i]
        sub = df[df["query"] == q].copy()
        if sub.empty:
            ax.set_title(f"Q{q}\n(no data)",
                         fontsize=FONT_SIZES["subplot_title"])
            ax.axis("off")
            continue

        ref_rows = find_all_default_step_rows(sub)
        if len(ref_rows) == 0:
            ref_rows = sub
        step_indices = sorted(ref_rows["step_index"].unique())
        step_kinds = {
            si: ref_rows[ref_rows["step_index"] == si]["step_kind"].iloc[0]
            for si in step_indices
        }
        used_kinds.update(step_kinds.values())

        bars = {label: [] for label in BAR_LABELS}
        for si in step_indices:
            kind = step_kinds[si]
            axis = _KIND_TO_AXIS.get(kind)
            step_row = ref_rows[ref_rows["step_index"] == si]
            fallback_ms = (float(step_row["mean_ms"].iloc[0])
                           if len(step_row) else 0.0)
            if axis is None:
                bars["baseline"].append((kind, fallback_ms))
                bars["best"].append((kind, fallback_ms))
                continue
            base_ms, best_ms, _ = gated_best_per_step(sub, axis, si)
            if base_ms is None:
                base_ms = fallback_ms
            if best_ms is None:
                best_ms = base_ms
            bars["baseline"].append((kind, base_ms))
            bars["best"].append((kind, best_ms))

        x_positions = list(range(len(BAR_LABELS)))
        for x, label in zip(x_positions, BAR_LABELS):
            bottom = 0.0
            for kind, ms in bars[label]:
                ax.bar([x], [ms], bottom=[bottom], width=0.6,
                       color=STEP_COLORS.get(kind, "#aaaaaa"),
                       edgecolor="white", linewidth=0.4)
                bottom += ms

        ax.set_xticks(x_positions)
        ax.set_xticklabels(BAR_LABELS, fontsize=FONT_SIZES["xticks"])
        ax.set_title(f"Q{q}",
                     fontsize=FONT_SIZES["subplot_title"],
                     fontweight="bold")
        ax.tick_params(axis="y", labelsize=FONT_SIZES["yticks"])
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
                color="#bbbbbb", zorder=0)
        ax.set_axisbelow(True)
        if i % 11 == 0:
            ax.set_ylabel("GPU time (ms)",
                          fontsize=FONT_SIZES["ylabel"])

    used_sorted = [k for k in STEP_COLORS if k in used_kinds]
    used_sorted += sorted(used_kinds - set(STEP_COLORS))
    legend_handles = [
        mpatches.Patch(color=STEP_COLORS.get(k, "#aaaaaa"), label=k)
        for k in used_sorted
    ]
    fig.suptitle(
        f"TPC-H per-step breakdown  ·  baseline / best-per-step  ·  "
        f"SF={scale}  ·  {machine_display}",
        fontsize=FONT_SIZES["suptitle"], fontweight="bold", y=0.99)
    n_legend = len(legend_handles)
    legend_ncol = (n_legend + 1) // 2
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=legend_ncol,
               bbox_to_anchor=(0.5, -0.01),
               fontsize=FONT_SIZES["legend"], frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    jpg_path = out_path.with_suffix(".jpg")
    fig.savefig(jpg_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def plot_kind_share(df: pd.DataFrame, out_path: Path, scale: int,
                    machine_display: str):
    """One bar per StepKind that appears in the workload. Bar height
    is the % contribution of that kind to the gated best-per-step
    total — the same per-step values used by the breakdown plot's
    "best" bar."""
    gb = compute_gated_best_per_step(df)
    by_kind = (gb.groupby("kind")["best_ms"]
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
                ha="center", va="bottom",
                fontsize=FONT_SIZES["annotation"], color="#222222",
                linespacing=1.15)

    ax.set_xticks(range(len(by_kind)))
    ax.set_xticklabels(by_kind["kind"], rotation=30,
                       fontsize=FONT_SIZES["xticks"],
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("% of total best-per-step time across 22 queries",
                  fontsize=FONT_SIZES["ylabel"])
    ax.tick_params(axis="y", labelsize=FONT_SIZES["yticks"])
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
            color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax * 1.32)

    ax.set_title(f"Time share per operator kind (gated best per step)  "
                 f"·  SF={scale}  ·  {machine_display}",
                 fontsize=FONT_SIZES["section_title"],
                 fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    jpg_path = out_path.with_suffix(".jpg")
    fig.savefig(jpg_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")
    print(f"  {len(by_kind)} kinds · gated best-per-step total = "
          f"{total:.2f} ms · top-3 share = "
          f"{float(by_kind['pct'].head(3).sum()):.1f}%")


def plot_top_steps(df: pd.DataFrame, out_path: Path, scale: int,
                   machine_display: str, top_n: int = 8):
    """Top-N most expensive single steps under the gated best-per-step
    workload. Bar colour encodes StepKind. Annotation shows the cfg
    picked for that step ("baseline" when no alternative beat
    baseline by ≥5%; "axis=value" otherwise)."""
    gb = compute_gated_best_per_step(df)
    gb = gb.sort_values(["query", "step_index"]).reset_index(drop=True)
    gb["kind_rank"] = gb.groupby(["query", "kind"]).cumcount() + 1
    gb["kind_count"] = gb.groupby(["query", "kind"])["kind"].transform("count")

    def label(r):
        q, k, r1, n = int(r["query"]), r["kind"], int(r["kind_rank"]), int(r["kind_count"])
        return f"Q{q}·{k}" if n == 1 else f"Q{q}·{k}#{r1}"
    gb["label"] = gb.apply(label, axis=1)

    def cfg_label(r):
        axis = r["axis"]
        if axis is None:
            return "baseline"
        bv = r["best_value"]
        if bv == PLOT_BASELINE.get(axis):
            return "baseline"
        v = VALUE_SHORT.get(bv, bv)
        return f"{AXIS_SHORT[axis]}={v}"
    gb["cfg"] = gb.apply(cfg_label, axis=1)

    total = float(gb["best_ms"].sum())
    gb["pct"] = gb["best_ms"] / total * 100.0
    top = gb.nlargest(top_n, "pct").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [STEP_COLORS.get(k, "#aaaaaa") for k in top["kind"]]
    bars = ax.bar(range(len(top)), top["pct"], color=colors,
                  edgecolor="white", linewidth=0.6, width=0.72)

    ymax = float(top["pct"].max())
    for b, pct, ms, cfg in zip(bars, top["pct"], top["best_ms"], top["cfg"]):
        ax.text(b.get_x() + b.get_width()/2, pct + ymax*0.02,
                f"{pct:.1f}%\n{ms:.1f} ms\n[{cfg}]",
                ha="center", va="bottom",
                fontsize=FONT_SIZES["annotation"], color="#222222",
                linespacing=1.15)

    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["label"], rotation=30,
                       fontsize=FONT_SIZES["xticks"],
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("% of total best-per-step time across 22 queries",
                  fontsize=FONT_SIZES["ylabel"])
    ax.tick_params(axis="y", labelsize=FONT_SIZES["yticks"])
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
            color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax * 1.32)

    ax.set_title(f"Top-{top_n} most expensive steps "
                 f"(gated best per step)  "
                 f"·  SF={scale}  ·  {machine_display}",
                 fontsize=FONT_SIZES["section_title"],
                 fontweight="bold", pad=12)

    kinds_in_plot = list(dict.fromkeys(top["kind"].tolist()))
    legend_handles = [
        mpatches.Patch(color=STEP_COLORS.get(k, "#aaaaaa"), label=k)
        for k in kinds_in_plot
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=FONT_SIZES["legend"], frameon=False, ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    jpg_path = out_path.with_suffix(".jpg")
    fig.savefig(jpg_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")
    print(f"  workload best total = {total:.2f} ms across "
          f"{len(gb)} steps; top-{top_n} sum = "
          f"{float(top['best_ms'].sum()):.2f} ms "
          f"({float(top['pct'].sum()):.1f}%)")


def best_per_step_for_axis(df_steps_q: pd.DataFrame, axis: str,
                           kinds: list) -> str:
    """For one query's df_steps slice, emit "Step N: <alt>" for each
    step of the matching kind where an alternative beat baseline by
    ≥5%. Empty for axes where no step had a winner — keeps the CSV
    column readable (only meaningful wins land in it)."""
    if df_steps_q.empty:
        return ""
    of_kind = df_steps_q[df_steps_q["step_kind"].isin(kinds)]
    if of_kind.empty:
        return ""
    parts = []
    for si in sorted(of_kind["step_index"].unique()):
        base_ms, best_ms, best_val = gated_best_per_step(
            df_steps_q, axis, si)
        if base_ms is None:
            continue
        if best_val == PLOT_BASELINE[axis]:
            continue   # baseline holds — no winner to report
        parts.append(f"Step {int(si)}: {best_val}")
    return "; ".join(parts)


def write_per_query_summary(df: pd.DataFrame, df_steps: pd.DataFrame,
                            out_path: Path):
    """Per-query summary CSV. One row per query 1..22; queries with
    no data are skipped.

    Columns:
      query, baseline-cfg, best-cfg, best-vs-baseline-speedup,
      filter, project, sort, group_by, distinct, eq_join

    - baseline-cfg = synth e2e baseline (sum of per-step baselines).
    - best-cfg = gated min e2e (the fastest cell, only if it beats
      baseline by ≥5%; else equals baseline).
    - best-vs-baseline-speedup = baseline / best (= 1.0 when no
      winner clears the gate).
    - Per-axis columns list the gated best value for each step of
      the matching kind ("Step N: <alt>"), or empty if no winner."""
    rows = []
    for q in range(1, 23):
        sub = df[df["query"] == q]
        if sub.empty:
            continue
        sub_steps = df_steps[df_steps["query"] == q]
        base_ms = e2e_baseline_synth_ms(sub_steps)
        best_cell_ms = float(sub["gpu_ms"].min())
        if best_cell_ms < base_ms * (1.0 - BEST_VS_BASELINE_THRESHOLD):
            best_ms = best_cell_ms
        else:
            best_ms = base_ms

        per_axis = {axis: best_per_step_for_axis(sub_steps, axis, kinds)
                    for axis, kinds in AXIS_TO_KINDS.items()}

        rows.append({
            "query":                    q,
            "baseline-cfg":             round(base_ms, 3),
            "best-cfg":                 round(best_ms, 3),
            "best-vs-baseline-speedup": (round(base_ms / best_ms, 3)
                                         if best_ms > 0 else ""),
            **per_axis,
        })
    out = pd.DataFrame(rows, columns=[
        "query", "baseline-cfg", "best-cfg",
        "best-vs-baseline-speedup",
        "filter", "project", "sort", "group_by", "distinct", "eq_join"])
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


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
    plot_e2e(df_e2e, df_steps,
             out_dir / f"tpch_e2e_sf{scale}.pdf", scale, display)
    plot_breakdown(df_steps,
                   out_dir / f"tpch_breakdown_sf{scale}.pdf",
                   scale, display)
    plot_top_steps(df_steps,
                   out_dir / f"tpch_top_steps_sf{scale}.pdf",
                   scale, display)
    plot_kind_share(df_steps,
                    out_dir / f"tpch_kind_share_sf{scale}.pdf",
                    scale, display)
    # Per-query summary CSV — same numbers as the e2e plot, in a
    # paste-into-Sheets shape. Lands in the data dir (alongside the
    # bench CSVs) rather than the PDF dir; it's a CSV.
    write_per_query_summary(
        df_e2e, df_steps,
        csv_dir / f"per_query_summary_sf{scale}.csv")


if __name__ == "__main__":
    main()
