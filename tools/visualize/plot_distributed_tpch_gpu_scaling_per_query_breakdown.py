#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot Baseline compute/communication scaling for every TPC-H query.

The single 4x6 figure contains Q1-Q22.  Every panel shows SF=1000 on 8, 16,
24, and 32 GPUs and SF=3000 on 24 and 32 GPUs.  Panels use independent y
scales because query runtimes differ substantially.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

import plot_distributed_tpch_gpu_scaling_aggregate_breakdown as scaling
from plot_distributed_tpch_aggregate_breakdown import (
    COLORS,
    QUERIES,
    REPO_ROOT,
    QueryBreakdown,
    plt,
    write_dataclass_csv,
)


FIGURE_NAME = "legacy_cta64_gpu_scaling_per_query_time_breakdown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "distributed_tpch",
        help="Distributed TPC-H results root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "distributed_tpch"
        / "figures"
        / "presentation",
        help="Output root; figures are written to pdf/ and png/",
    )
    return parser.parse_args()


def plot(query_rows: list[QueryBreakdown], output_dir: Path) -> None:
    lookup = {
        (row.query, row.scale_factor, row.gpus): row for row in query_rows
    }
    expected = {
        (query, scale_factor, gpus)
        for query in QUERIES
        for scale_factor, gpus in scaling.CONFIGURATIONS
    }
    if set(lookup) != expected:
        raise ValueError(
            f"unexpected query/configuration coverage: "
            f"{sorted(set(lookup) ^ expected)}"
        )

    positions = [0.0, 1.0, 2.0, 3.0, 4.7, 5.7]
    tick_labels = [
        f"{scale_factor // 1000}K/{gpus}"
        for scale_factor, gpus in scaling.CONFIGURATIONS
    ]
    fig, axes = plt.subplots(4, 6, figsize=(20.0, 13.0))
    fig.patch.set_facecolor("white")

    for ax, query in zip(axes.flat, QUERIES, strict=False):
        ax.set_facecolor("white")
        rows = [
            lookup[(query, scale_factor, gpus)]
            for scale_factor, gpus in scaling.CONFIGURATIONS
        ]
        computation = [row.computation_ms for row in rows]
        communication = [row.communication_ms for row in rows]
        totals = [row.median_e2e_rank_max_ms for row in rows]
        maximum = max(totals)
        ax.bar(
            positions,
            computation,
            width=0.74,
            color=COLORS["Computation"],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        ax.bar(
            positions,
            communication,
            width=0.74,
            bottom=computation,
            color=COLORS["Communication"],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        for position, total in zip(positions, totals, strict=True):
            ax.text(
                position,
                total + maximum * 0.025,
                f"{total:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.4,
                fontweight="bold",
                color="#333333",
            )

        ax.axvline(3.85, color="#E2E2E2", linewidth=0.8, zorder=1)
        ax.set_title(f"Q{query}", fontsize=11.5, fontweight="bold", pad=5)
        ax.set_xticks(positions, tick_labels)
        ax.tick_params(axis="x", labelsize=7.0, pad=2)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_xlim(-0.65, 6.35)
        ax.set_ylim(0.0, maximum * 1.20)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.grid(
            axis="y",
            color="#D8D8D8",
            linewidth=0.65,
            linestyle=(0, (2, 3)),
            zorder=0,
        )
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")

    for ax in axes.flat[len(QUERIES) :]:
        ax.set_visible(False)

    handles = (
        mpatches.Patch(
            facecolor=COLORS["Computation"],
            label="Computation",
        ),
        mpatches.Patch(
            facecolor=COLORS["Communication"],
            label="Communication",
        ),
    )
    fig.suptitle(
        "Distributed TPC-H Per-Query Time Breakdown",
        fontsize=20,
        y=0.985,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.953),
        ncol=2,
        frameon=False,
        fontsize=12,
        handlelength=1.4,
        columnspacing=1.8,
    )
    fig.supylabel("Execution time (ms)", fontsize=14, x=0.012)
    fig.supxlabel(
        "Configuration (scale factor / GPUs)",
        fontsize=13,
        y=0.018,
    )
    fig.subplots_adjust(
        left=0.045,
        right=0.995,
        bottom=0.065,
        top=0.905,
        wspace=0.28,
        hspace=0.42,
    )

    for suffix in ("pdf", "png"):
        directory = output_dir / suffix
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{FIGURE_NAME}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    query_rows, _ = scaling.collect(results_dir)
    write_dataclass_csv(
        output_dir / "legacy_cta64_gpu_scaling_per_query_breakdown.csv",
        query_rows,
    )
    plot(query_rows, output_dir)


if __name__ == "__main__":
    main()
