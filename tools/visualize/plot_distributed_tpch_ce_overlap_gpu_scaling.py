#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Compare Baseline, Overlap+SM, and Overlap+CE aggregate TPC-H time.

Every bar sums the clean median maximum-rank end-to-end time of Q1-Q22.
SF=1000 includes 8, 16, 24, and 32 GPUs; SF=3000 includes 24 and 32 GPUs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator

from plot_distributed_tpch_aggregate_breakdown import (
    QUERIES,
    REPO_ROOT,
    plt,
)
from plot_distributed_tpch_overlap_arm_comparison import (
    timing_summary_for_figure,
)


CONFIGURATIONS = (
    (1000, 8),
    (1000, 16),
    (1000, 24),
    (1000, 32),
    (3000, 24),
    (3000, 32),
)
ARMS = (
    ("legacy_cta64", "Baseline", "#4C78A8"),
    ("all_co_nccl_cta64", "Overlap+SM", "#54A24B"),
    ("all_co_ce_pull", "Overlap+CE", "#E45756"),
)
FIGURE_NAME = "distributed_tpch_ce_overlap_gpu_scaling_aggregate_comparison"


@dataclass(frozen=True)
class QueryTiming:
    scale_factor: int
    gpus: int
    arm: str
    query: int
    median_e2e_rank_max_ms: float
    timing_result_status: str
    timing_attempt_dir: str


@dataclass(frozen=True)
class AggregateTiming:
    scale_factor: int
    gpus: int
    arm: str
    label: str
    query_count: int
    aggregate_time_s: float
    speedup_over_baseline: float


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


def collect(
    results_dir: Path,
) -> tuple[list[QueryTiming], list[AggregateTiming]]:
    query_rows = []
    totals: dict[tuple[int, int, str], float] = {}

    for scale_factor, gpus in CONFIGURATIONS:
        for arm, _, _ in ARMS:
            selected = []
            for query in QUERIES:
                median, _, attempt, result = timing_summary_for_figure(
                    results_dir, scale_factor, gpus, query, arm=arm
                )
                row = QueryTiming(
                    scale_factor=scale_factor,
                    gpus=gpus,
                    arm=arm,
                    query=query,
                    median_e2e_rank_max_ms=median,
                    timing_result_status=str(result["status"]),
                    timing_attempt_dir=str(attempt.relative_to(results_dir)),
                )
                selected.append(row)
                query_rows.append(row)
            if {row.query for row in selected} != set(QUERIES):
                raise ValueError(
                    f"SF={scale_factor}, {gpus} GPUs, {arm}: "
                    "incomplete Q1-Q22 coverage"
                )
            totals[(scale_factor, gpus, arm)] = sum(
                row.median_e2e_rank_max_ms for row in selected
            ) / 1000.0

    aggregate_rows = []
    for scale_factor, gpus in CONFIGURATIONS:
        baseline = totals[(scale_factor, gpus, "legacy_cta64")]
        for arm, label, _ in ARMS:
            total = totals[(scale_factor, gpus, arm)]
            aggregate_rows.append(
                AggregateTiming(
                    scale_factor=scale_factor,
                    gpus=gpus,
                    arm=arm,
                    label=label,
                    query_count=len(QUERIES),
                    aggregate_time_s=total,
                    speedup_over_baseline=baseline / total,
                )
            )
    return query_rows, aggregate_rows


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)
    print(f"wrote {path}")


def plot(rows: list[AggregateTiming], output_dir: Path) -> None:
    lookup = {
        (row.scale_factor, row.gpus, row.arm): row for row in rows
    }
    expected = {
        (scale_factor, gpus, arm)
        for scale_factor, gpus in CONFIGURATIONS
        for arm, _, _ in ARMS
    }
    if set(lookup) != expected:
        raise ValueError(
            f"unexpected aggregate coverage: {sorted(set(lookup) ^ expected)}"
        )

    positions = np.array((0.0, 1.0, 2.0, 3.0, 4.8, 5.8))
    width = 0.24
    offsets = (-0.27, 0.0, 0.27)
    maximum = max(row.aggregate_time_s for row in rows)

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for offset, (arm, label, color) in zip(offsets, ARMS, strict=True):
        ordered = [
            lookup[(scale_factor, gpus, arm)]
            for scale_factor, gpus in CONFIGURATIONS
        ]
        values = [row.aggregate_time_s for row in ordered]
        bars = ax.bar(
            positions + offset,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        for bar, row in zip(bars, ordered, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + maximum * 0.022,
                f"{row.aggregate_time_s:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.0,
                fontweight="bold",
                color="#333333",
            )

    ax.axvline(3.9, color="#E0E0E0", linewidth=1.0, zorder=1)
    ax.set_xticks(
        positions,
        [f"{gpus} GPUs" for _, gpus in CONFIGURATIONS],
    )
    ax.tick_params(axis="x", labelsize=11, pad=7)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlim(-0.65, 6.45)
    ax.set_ylim(0.0, maximum * 1.25)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.set_ylabel("Aggregate execution time (s)", fontsize=13)
    ax.set_title("Distributed TPC-H Aggregate Execution Time", fontsize=18, pad=18)
    ax.grid(
        axis="y",
        color="#D5D5D5",
        linewidth=0.8,
        linestyle=(0, (2, 3)),
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=12,
        columnspacing=2.0,
    )

    transform = ax.get_xaxis_transform()
    ax.text(
        np.mean(positions[:4]),
        -0.16,
        "SF=1000",
        transform=transform,
        ha="center",
        va="top",
        fontsize=13,
        color="#444444",
    )
    ax.text(
        np.mean(positions[4:]),
        -0.16,
        "SF=3000",
        transform=transform,
        ha="center",
        va="top",
        fontsize=13,
        color="#444444",
    )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.21, top=0.87)

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
    query_rows, aggregate_rows = collect(results_dir)
    write_csv(output_dir / f"{FIGURE_NAME}_queries.csv", query_rows)
    write_csv(output_dir / f"{FIGURE_NAME}.csv", aggregate_rows)
    plot(aggregate_rows, output_dir)


if __name__ == "__main__":
    main()
