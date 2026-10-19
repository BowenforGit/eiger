#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Compare legacy_cta64 and legacy_ce_pull aggregate time breakdowns.

For each scale-matched configuration, the two arms are shown as adjacent bars;
every bar is stacked into computation and communication.  Metric reduction and
attempt acceptance are shared with plot_distributed_tpch_aggregate_breakdown.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from plot_distributed_tpch_aggregate_breakdown import (
    COLORS,
    CONFIGURATIONS,
    REPO_ROOT,
    AggregateBreakdown,
    QueryBreakdown,
    collect,
    label_segment,
    plt,
    write_dataclass_csv,
)


ARMS = ("legacy_cta64", "legacy_ce_pull")
ARM_LABELS = {
    "legacy_cta64": "NCCL",
    "legacy_ce_pull": "Copy engine",
}
FIGURE_NAME = "legacy_cta64_vs_legacy_ce_pull_aggregate_time_breakdown"


@dataclass(frozen=True)
class BackendQueryBreakdown:
    arm: str
    scale_factor: int
    gpus: int
    nodes: int
    query: int
    median_e2e_rank_max_ms: float
    median_communication_fraction: float
    computation_ms: float
    communication_ms: float
    timing_result_status: str
    breakdown_result_status: str
    timing_attempt_dir: str
    breakdown_attempt_dir: str


@dataclass(frozen=True)
class BackendAggregateBreakdown:
    arm: str
    scale_factor: int
    gpus: int
    nodes: int
    query_count: int
    computation_s: float
    communication_s: float
    total_s: float
    communication_percent: float


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


def add_arm_to_query(arm: str, row: QueryBreakdown) -> BackendQueryBreakdown:
    return BackendQueryBreakdown(arm=arm, **asdict(row))


def add_arm_to_aggregate(
    arm: str, row: AggregateBreakdown
) -> BackendAggregateBreakdown:
    return BackendAggregateBreakdown(arm=arm, **asdict(row))


def collect_arms(
    results_dir: Path,
) -> tuple[list[BackendQueryBreakdown], list[BackendAggregateBreakdown]]:
    query_rows = []
    aggregate_rows = []
    for arm in ARMS:
        arm_queries, arm_aggregates = collect(results_dir, arm=arm)
        query_rows.extend(add_arm_to_query(arm, row) for row in arm_queries)
        aggregate_rows.extend(
            add_arm_to_aggregate(arm, row) for row in arm_aggregates
        )
    return query_rows, aggregate_rows


def plot(
    aggregate_rows: list[BackendAggregateBreakdown], output_dir: Path
) -> None:
    by_key = {
        (row.scale_factor, row.arm): row for row in aggregate_rows
    }
    expected = {
        (scale, arm) for scale, _ in CONFIGURATIONS for arm in ARMS
    }
    if set(by_key) != expected:
        raise ValueError(
            f"unexpected scale/arm coverage: {sorted(set(by_key) ^ expected)}"
        )

    group_centers = [0.0, 1.45]
    arm_offsets = {"legacy_cta64": -0.23, "legacy_ce_pull": 0.23}
    width = 0.40
    positions = []
    rows = []
    tick_labels = []
    for group_index, (scale, _) in enumerate(CONFIGURATIONS):
        for arm in ARMS:
            positions.append(group_centers[group_index] + arm_offsets[arm])
            rows.append(by_key[(scale, arm)])
            tick_labels.append(ARM_LABELS[arm])

    computation = [row.computation_s for row in rows]
    communication = [row.communication_s for row in rows]
    totals = [row.total_s for row in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(
        positions,
        computation,
        width,
        color=COLORS["Computation"],
        edgecolor="white",
        linewidth=0.8,
        label="Computation",
        zorder=3,
    )
    ax.bar(
        positions,
        communication,
        width,
        bottom=computation,
        color=COLORS["Communication"],
        edgecolor="white",
        linewidth=0.8,
        label="Communication",
        zorder=3,
    )

    for index, total in enumerate(totals):
        label_segment(
            ax, positions[index], 0.0, computation[index], total
        )
        label_segment(
            ax,
            positions[index],
            computation[index],
            communication[index],
            total,
        )
        ax.text(
            positions[index],
            total + max(totals) * 0.022,
            f"{total:.2f} s",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#2F2F2F",
        )

    ax.set_xticks(positions, tick_labels)
    ax.tick_params(axis="x", labelsize=9.5, pad=5)
    for center, (scale, gpus) in zip(group_centers, CONFIGURATIONS):
        ax.text(
            center,
            -0.145,
            f"SF={scale}  •  {gpus} GPUs",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            color="#333333",
        )

    ax.set_ylabel("Aggregate execution time (s)")
    ax.set_title("Distributed TPC-H Aggregate Time Breakdown", pad=14)
    ax.set_ylim(0.0, max(totals) * 1.22)
    ax.set_xlim(group_centers[0] - 0.72, group_centers[-1] + 0.72)
    ax.grid(
        axis="y",
        color="#D8D8D8",
        linewidth=0.8,
        linestyle=(0, (2, 3)),
        zorder=0,
    )
    ax.axvline(
        sum(group_centers) / 2.0,
        color="#E8E8E8",
        linewidth=0.8,
        zorder=1,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="both", colors="#333333")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.12, right=0.99, top=0.87, bottom=0.24)

    for suffix in ("pdf", "png"):
        format_dir = output_dir / suffix
        format_dir.mkdir(parents=True, exist_ok=True)
        output = format_dir / f"{FIGURE_NAME}.{suffix}"
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        print(f"wrote {output}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    query_rows, aggregate_rows = collect_arms(args.results_dir.resolve())
    write_dataclass_csv(
        output_dir / "legacy_backend_query_breakdown.csv", query_rows
    )
    write_dataclass_csv(
        output_dir / "legacy_backend_aggregate_breakdown.csv",
        aggregate_rows,
    )
    plot(aggregate_rows, output_dir)
    for row in sorted(
        aggregate_rows,
        key=lambda item: (item.scale_factor, ARMS.index(item.arm)),
    ):
        print(
            f"SF={row.scale_factor}, {row.gpus} GPUs, {row.arm}: "
            f"compute={row.computation_s:.3f} s, "
            f"communication={row.communication_s:.3f} s, "
            f"total={row.total_s:.3f} s"
        )


if __name__ == "__main__":
    main()
