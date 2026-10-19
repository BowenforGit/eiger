#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Render the four required concurrency heatmaps from benchmark summary CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


METRICS = {
    "h_ideal": (
        "Concurrent makespan / ideal isolated overlap",
        r"$H_{ideal}=M_{concurrent}/\max(T_A,T_B)$",
        True,
    ),
    "h_sequential": (
        "Concurrent makespan / isolated sequential sum",
        r"$H_{sequential}=M_{concurrent}/(T_A+T_B)$",
        False,
    ),
    "slowdown_a": (
        "Directional slowdown of operation A",
        r"$S_A=C_A/T_A$",
        True,
    ),
    "slowdown_b": (
        "Directional slowdown of operation B",
        r"$S_B=C_B/T_B$",
        True,
    ),
}

AXIS_LABELS = {
    "expr.two_col_add.nofuse": "expression num_items",
    "misc.gather.thrust": "gather num_inputs = gather_map_size",
    "containers.multiset.insert": "multiset unique_keys",
    "containers.multiset.retrieve": "multiset probe_keys (unique_keys = probe_keys / 4)",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def render(rows: list[dict[str, str]], metric: str, output_dir: Path) -> None:
    title, definition, center_one = METRICS[metric]
    a_exponents = sorted({int(row["a_exponent"]) for row in rows})
    b_exponents = sorted({int(row["b_exponent"]) for row in rows})
    a_index = {value: index for index, value in enumerate(a_exponents)}
    b_index = {value: index for index, value in enumerate(b_exponents)}
    data = np.full((len(a_exponents), len(b_exponents)), np.nan)
    states: dict[tuple[int, int], tuple[str, bool]] = {}

    for row in rows:
        y = a_index[int(row["a_exponent"])]
        x = b_index[int(row["b_exponent"])]
        status = row.get("status", "")
        stable = row.get("stable", "0") == "1"
        states[(y, x)] = (status, stable)
        if status == "success" and row.get(metric):
            data[y, x] = float(row[metric])

    valid = finite(data)
    if valid.size:
        lo, hi = float(valid.min()), float(valid.max())
        if center_one:
            plot_data = np.log2(data)
            valid_log = np.log2(valid)
            extent = max(abs(float(valid_log.min())), abs(float(valid_log.max())), 0.02)
            norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
            cmap = plt.get_cmap("coolwarm").copy()
            colorbar_label = "log2(ratio), centered at 1x"
        else:
            plot_data = data
            padding = max((hi - lo) * 0.05, 0.01)
            norm = plt.Normalize(lo - padding, hi + padding)
            cmap = plt.get_cmap("viridis").copy()
            colorbar_label = "ratio (x)"
    else:
        plot_data = data
        norm = plt.Normalize(0.0, 1.0)
        cmap = plt.get_cmap("coolwarm").copy()
        colorbar_label = "ratio (x)"
    cmap.set_bad("#bdbdbd")

    width = max(9.5, 0.95 * len(b_exponents) + 3.5)
    height = max(8.0, 0.8 * len(a_exponents) + 3.5)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(np.ma.masked_invalid(plot_data), origin="lower", cmap=cmap, norm=norm,
                      aspect="auto")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    ax.set_xticks(range(len(b_exponents)), [f"2^{value}" for value in b_exponents])
    ax.set_yticks(range(len(a_exponents)), [f"2^{value}" for value in a_exponents])
    first = rows[0]
    op_a = first["op_a"]
    op_b = first["op_b"]
    ax.set_xlabel(f"Operation B size: {AXIS_LABELS.get(op_b, 'size')}")
    ax.set_ylabel(f"Operation A size: {AXIS_LABELS.get(op_a, 'size')}")
    subtitle = (
        f"A: {op_a}\n"
        f"B: {op_b} | GPU: {first['gpu_name']}\n{definition}"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=14)

    for y in range(len(a_exponents)):
        for x in range(len(b_exponents)):
            value = data[y, x]
            status, stable = states.get((y, x), ("missing", False))
            if math.isfinite(value):
                color_value = math.log2(value) if center_one else value
                rgba = cmap(norm(color_value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                label = f"{value:.3f}x" + ("\n!" if not stable else "")
                ax.text(x, y, label, ha="center", va="center",
                        color="black" if luminance > 0.55 else "white", fontsize=14)
            else:
                marker = "S" if status.startswith("skipped") else "X" if status.startswith("failed") else "?"
                ax.text(x, y, marker, ha="center", va="center", color="black",
                        fontsize=14, fontweight="bold")

    ax.text(0.0, -0.14, "Missing-cell markers: S=memory preflight skip, X=failed, ?=missing; !=CI target not met",
            transform=ax.transAxes, fontsize=10, va="top")
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{metric}.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    rows = read_rows(args.summary_csv)
    if not rows:
        raise SystemExit("summary CSV contains no cells")
    for metric in METRICS:
        render(rows, metric, args.output_dir)


if __name__ == "__main__":
    main()
