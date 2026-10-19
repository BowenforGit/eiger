#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot Baseline aggregate compute/communication time across GPU counts.

The figure includes SF=1000 on 8, 16, 24, and 32 GPUs and SF=3000 on 24 and
32 GPUs.  Each bar aggregates Q1-Q22.  Per-query clean latency is the median
maximum-rank wall time over ten timing iterations.  The communication fraction
is the median interval-union fraction from five breakdown iterations, evaluated
on each iteration's critical rank.  Components are scaled to clean latency.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from plot_distributed_tpch_aggregate_breakdown import (
    COLORS,
    QUERIES,
    REPO_ROOT,
    AggregateBreakdown,
    QueryBreakdown,
    communication_fraction_summary,
    label_segment,
    plt,
    timing_summary,
    write_dataclass_csv,
)


ARM = "legacy_cta64"
CONFIGURATIONS = (
    (1000, 8),
    (1000, 16),
    (1000, 24),
    (1000, 32),
    (3000, 24),
    (3000, 32),
)
FIGURE_NAME = "legacy_cta64_gpu_scaling_aggregate_time_breakdown"


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
) -> tuple[list[QueryBreakdown], list[AggregateBreakdown]]:
    query_rows = []
    for scale_factor, gpus in CONFIGURATIONS:
        for query in QUERIES:
            median_e2e, timing_attempt, timing_result = timing_summary(
                results_dir,
                scale_factor,
                gpus,
                query,
                arm=ARM,
            )
            communication_fraction, breakdown_attempt, breakdown_result = (
                communication_fraction_summary(
                    results_dir,
                    scale_factor,
                    gpus,
                    query,
                    arm=ARM,
                )
            )
            communication_ms = median_e2e * communication_fraction
            query_rows.append(
                QueryBreakdown(
                    scale_factor=scale_factor,
                    gpus=gpus,
                    nodes=gpus // 4,
                    query=query,
                    median_e2e_rank_max_ms=median_e2e,
                    median_communication_fraction=communication_fraction,
                    computation_ms=median_e2e - communication_ms,
                    communication_ms=communication_ms,
                    timing_result_status=str(timing_result["status"]),
                    breakdown_result_status=str(breakdown_result["status"]),
                    timing_attempt_dir=str(
                        timing_attempt.relative_to(results_dir)
                    ),
                    breakdown_attempt_dir=str(
                        breakdown_attempt.relative_to(results_dir)
                    ),
                )
            )

    aggregate_rows = []
    for scale_factor, gpus in CONFIGURATIONS:
        selected = [
            row
            for row in query_rows
            if row.scale_factor == scale_factor and row.gpus == gpus
        ]
        if {row.query for row in selected} != set(QUERIES):
            raise ValueError(
                f"SF={scale_factor}, {gpus} GPUs: incomplete Q1-Q22 coverage"
            )
        computation_ms = sum(row.computation_ms for row in selected)
        communication_ms = sum(row.communication_ms for row in selected)
        total_ms = sum(row.median_e2e_rank_max_ms for row in selected)
        if not math.isclose(
            computation_ms + communication_ms,
            total_ms,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"SF={scale_factor}, {gpus} GPUs: stack does not equal total"
            )
        aggregate_rows.append(
            AggregateBreakdown(
                scale_factor=scale_factor,
                gpus=gpus,
                nodes=gpus // 4,
                query_count=len(selected),
                computation_s=computation_ms / 1000.0,
                communication_s=communication_ms / 1000.0,
                total_s=total_ms / 1000.0,
                communication_percent=100.0 * communication_ms / total_ms,
            )
        )
    return query_rows, aggregate_rows


def plot(rows: list[AggregateBreakdown], output_dir: Path) -> None:
    lookup = {(row.scale_factor, row.gpus): row for row in rows}
    if set(lookup) != set(CONFIGURATIONS):
        raise ValueError(
            f"unexpected configuration coverage: "
            f"{sorted(set(lookup) ^ set(CONFIGURATIONS))}"
        )

    positions = [0.0, 1.0, 2.0, 3.0, 4.7, 5.7]
    ordered = [lookup[configuration] for configuration in CONFIGURATIONS]
    computation = [row.computation_s for row in ordered]
    communication = [row.communication_s for row in ordered]
    totals = [row.total_s for row in ordered]
    maximum = max(totals)

    fig, ax = plt.subplots(figsize=(10.3, 5.3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    width = 0.72
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

    for position, compute, communicate, total in zip(
        positions,
        computation,
        communication,
        totals,
        strict=True,
    ):
        label_segment(ax, position, 0.0, compute, total)
        label_segment(ax, position, compute, communicate, total)
        ax.text(
            position,
            total + maximum * 0.025,
            f"{total:.2f} s",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#2F2F2F",
        )

    ax.set_xticks(
        positions,
        [f"{gpus} GPUs" for _, gpus in CONFIGURATIONS],
    )
    ax.text(
        1.5,
        -0.16,
        "SF=1000",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=13.5,
        color="#444444",
    )
    ax.text(
        5.2,
        -0.16,
        "SF=3000",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=13.5,
        color="#444444",
    )
    ax.axvline(3.85, color="#E2E2E2", linewidth=1.1, zorder=1)
    ax.set_ylabel("Aggregate execution time (s)")
    ax.set_title("Distributed TPC-H Aggregate Time Breakdown", pad=16)
    ax.set_ylim(0.0, maximum * 1.22)
    ax.set_xlim(-0.65, 6.35)
    ax.grid(
        axis="y",
        color="#D8D8D8",
        linewidth=0.8,
        linestyle=(0, (2, 3)),
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="both", colors="#333333")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.24)

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
    write_dataclass_csv(
        output_dir / "legacy_cta64_gpu_scaling_query_breakdown.csv",
        query_rows,
    )
    write_dataclass_csv(
        output_dir / "legacy_cta64_gpu_scaling_aggregate_breakdown.csv",
        aggregate_rows,
    )
    plot(aggregate_rows, output_dir)
    for row in aggregate_rows:
        print(
            f"SF={row.scale_factor}, {row.gpus} GPUs: "
            f"compute={row.computation_s:.3f} s, "
            f"communication={row.communication_s:.3f} s, "
            f"total={row.total_s:.3f} s"
        )


if __name__ == "__main__":
    main()
