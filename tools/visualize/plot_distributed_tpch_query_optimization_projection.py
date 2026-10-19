#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot per-query optimization stages and additive-benefit projections.

Creates a 2x2 presentation figure for Q3, Q5, Q8, and Q9.  Each panel shows
SF=1000/8 GPUs and SF=3000/24 GPUs with four bars: measured legacy_cta64
breakdown, measured synthesized_cta64 breakdown, all_co_ce_pull clean
end-to-end time, and projected end-to-end time.  The projection is evaluated
independently for every query and scale:

    projected = synthesized_total - (legacy_total - all_co_ce_pull_total).

SF=3000 Q3 is intentionally included when its failed-verification attempt
reports complete measurements, matching the aggregate presentation figures.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches

from plot_distributed_tpch_aggregate_breakdown import (
    COLORS,
    CONFIGURATIONS,
    REPO_ROOT,
    QueryBreakdown,
    collect,
    plt,
    timing_summary,
    write_dataclass_csv,
)


QUERIES = (3, 5, 8, 9)
BASELINE_ARM = "legacy_cta64"
SYNTHESIZED_ARM = "synthesized_cta64"
OVERLAP_ARM = "all_co_ce_pull"
OVERLAP_COLOR = "#59A14F"
FIGURE_NAME = "distributed_tpch_query_optimization_projection"


@dataclass(frozen=True)
class QueryProjectionSummary:
    scale_factor: int
    gpus: int
    nodes: int
    query: int
    baseline_compute_ms: float
    baseline_communication_ms: float
    baseline_total_ms: float
    synthesized_compute_ms: float
    synthesized_communication_ms: float
    synthesized_total_ms: float
    overlap_end_to_end_ms: float
    observed_overlap_benefit_ms: float
    projected_compute_ms: float
    projected_exposed_communication_ms: float
    projected_total_ms: float


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


def index_query_rows(
    rows: list[QueryBreakdown], arm: str
) -> dict[tuple[int, int], QueryBreakdown]:
    indexed = {
        (row.scale_factor, row.query): row
        for row in rows
        if row.query in QUERIES
    }
    expected = {
        (scale, query)
        for scale, _ in CONFIGURATIONS
        for query in QUERIES
    }
    if set(indexed) != expected:
        raise ValueError(
            f"{arm}: unexpected scale/query coverage "
            f"{sorted(set(indexed) ^ expected)}"
        )
    return indexed


def collect_query_projections(
    results_dir: Path,
) -> list[QueryProjectionSummary]:
    baseline_queries, _ = collect(results_dir, arm=BASELINE_ARM)
    synthesized_queries, _ = collect(results_dir, arm=SYNTHESIZED_ARM)
    baseline = index_query_rows(baseline_queries, BASELINE_ARM)
    synthesized = index_query_rows(synthesized_queries, SYNTHESIZED_ARM)

    summaries = []
    for query in QUERIES:
        for scale, gpus in CONFIGURATIONS:
            baseline_row = baseline[scale, query]
            synthesized_row = synthesized[scale, query]
            overlap_total_ms, _, _ = timing_summary(
                results_dir,
                scale,
                gpus,
                query,
                arm=OVERLAP_ARM,
            )
            overlap_benefit_ms = (
                baseline_row.median_e2e_rank_max_ms - overlap_total_ms
            )
            projected_residual_ms = (
                synthesized_row.communication_ms - overlap_benefit_ms
            )
            if overlap_benefit_ms < 0.0:
                raise ValueError(
                    f"Q{query} SF={scale}: {OVERLAP_ARM} is slower than baseline"
                )
            if projected_residual_ms < 0.0:
                raise ValueError(
                    f"Q{query} SF={scale}: projected exposed communication "
                    f"is negative ({projected_residual_ms:.6f} ms)"
                )
            projected_total_ms = (
                synthesized_row.computation_ms + projected_residual_ms
            )
            equivalent_total_ms = (
                synthesized_row.median_e2e_rank_max_ms - overlap_benefit_ms
            )
            if not math.isclose(
                projected_total_ms,
                equivalent_total_ms,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"Q{query} SF={scale}: projection identities disagree"
                )
            summaries.append(
                QueryProjectionSummary(
                    scale_factor=scale,
                    gpus=gpus,
                    nodes=gpus // 4,
                    query=query,
                    baseline_compute_ms=baseline_row.computation_ms,
                    baseline_communication_ms=baseline_row.communication_ms,
                    baseline_total_ms=baseline_row.median_e2e_rank_max_ms,
                    synthesized_compute_ms=synthesized_row.computation_ms,
                    synthesized_communication_ms=(
                        synthesized_row.communication_ms
                    ),
                    synthesized_total_ms=(
                        synthesized_row.median_e2e_rank_max_ms
                    ),
                    overlap_end_to_end_ms=overlap_total_ms,
                    observed_overlap_benefit_ms=overlap_benefit_ms,
                    projected_compute_ms=synthesized_row.computation_ms,
                    projected_exposed_communication_ms=projected_residual_ms,
                    projected_total_ms=projected_total_ms,
                )
            )
    return summaries


def label_segment(
    ax: plt.Axes,
    position: float,
    bottom: float,
    height: float,
    total: float,
) -> None:
    share = height / total
    if share < 0.07:
        return
    text = f"{height:.0f} ms"
    if share >= 0.14:
        text += f"\n({100.0 * share:.0f}%)"
    ax.text(
        position,
        bottom + height / 2.0,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=7.6,
        fontweight="bold",
        linespacing=1.05,
    )


def plot(rows: list[QueryProjectionSummary], output_dir: Path) -> None:
    by_key = {(row.query, row.scale_factor): row for row in rows}
    expected = {
        (query, scale)
        for query in QUERIES
        for scale, _ in CONFIGURATIONS
    }
    if set(by_key) != expected:
        raise ValueError(f"unexpected query/scale coverage: {set(by_key) ^ expected}")

    group_centers = [0.0, 1.72]
    offsets = (-0.58, -0.19, 0.19, 0.58)
    tick_labels = (
        "Baseline",
        "Opt. comp",
        "Comm-comp\noverlap",
        "Projected",
    )
    width = 0.34
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.5))
    fig.patch.set_facecolor("white")

    for ax, query in zip(axes.flat, QUERIES):
        ax.set_facecolor("white")
        positions = []
        labels = []
        specs = []
        for group_index, (scale, _) in enumerate(CONFIGURATIONS):
            row = by_key[query, scale]
            group_specs = (
                (
                    "measured",
                    row.baseline_compute_ms,
                    row.baseline_communication_ms,
                    row.baseline_total_ms,
                ),
                (
                    "measured",
                    row.synthesized_compute_ms,
                    row.synthesized_communication_ms,
                    row.synthesized_total_ms,
                ),
                ("end_to_end", 0.0, 0.0, row.overlap_end_to_end_ms),
                ("projected", 0.0, 0.0, row.projected_total_ms),
            )
            for offset, label, spec in zip(offsets, tick_labels, group_specs):
                positions.append(group_centers[group_index] + offset)
                labels.append(label)
                specs.append(spec)

        totals = [spec[3] for spec in specs]
        maximum = max(totals)
        for position, (kind, computation, communication, total) in zip(
            positions, specs
        ):
            if kind in ("end_to_end", "projected"):
                ax.bar(
                    position,
                    total,
                    width,
                    color=OVERLAP_COLOR,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                ax.text(
                    position,
                    total / 2.0,
                    "E2E",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.0,
                    fontweight="bold",
                )
                if kind == "projected":
                    ax.add_patch(
                        mpatches.Rectangle(
                            (position - width / 2.0, 0.0),
                            width,
                            total,
                            fill=False,
                            edgecolor="#3F3F3F",
                            linewidth=1.0,
                            linestyle=(0, (4, 2)),
                            zorder=4,
                        )
                    )
            else:
                ax.bar(
                    position,
                    computation,
                    width,
                    color=COLORS["Computation"],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                ax.bar(
                    position,
                    communication,
                    width,
                    bottom=computation,
                    color=COLORS["Communication"],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                label_segment(ax, position, 0.0, computation, total)
                label_segment(
                    ax, position, computation, communication, total
                )
            ax.text(
                position,
                total + maximum * 0.025,
                f"{total:.0f} ms",
                ha="center",
                va="bottom",
                fontsize=8.1,
                fontweight="bold",
                color="#2F2F2F",
            )

        ax.set_title(f"Q{query}", fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(positions, labels)
        ax.tick_params(axis="x", labelsize=7.8, pad=4)
        ax.tick_params(axis="y", labelsize=8.5)
        for center, (scale, gpus) in zip(group_centers, CONFIGURATIONS):
            ax.text(
                center,
                -0.22,
                f"SF={scale}  •  {gpus} GPUs",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9.0,
                color="#333333",
            )
        ax.set_ylim(0.0, maximum * 1.25)
        ax.set_xlim(group_centers[0] - 0.84, group_centers[-1] + 0.84)
        ax.grid(
            axis="y",
            color="#D8D8D8",
            linewidth=0.7,
            linestyle=(0, (2, 3)),
            zorder=0,
        )
        ax.axvline(
            sum(group_centers) / 2.0,
            color="#E8E8E8",
            linewidth=0.7,
            zorder=1,
        )
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#777777")
        ax.spines["bottom"].set_color("#777777")

    legend_handles = (
        mpatches.Patch(color=COLORS["Computation"], label="Computation"),
        mpatches.Patch(color=COLORS["Communication"], label="Communication"),
        mpatches.Patch(color=OVERLAP_COLOR, label="End-to-end"),
    )
    fig.suptitle(
        "Distributed TPC-H Query Time Breakdown",
        fontsize=16,
        y=0.985,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handlelength=1.4,
        columnspacing=1.5,
    )
    fig.supylabel("Query execution time (ms)", x=0.018, fontsize=12)
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.89,
        bottom=0.10,
        hspace=0.50,
        wspace=0.17,
    )

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
    summaries = collect_query_projections(args.results_dir.resolve())
    write_dataclass_csv(
        output_dir / "query_optimization_projection_summary.csv", summaries
    )
    plot(summaries, output_dir)
    for row in summaries:
        print(
            f"Q{row.query} SF={row.scale_factor}: "
            f"baseline={row.baseline_total_ms:.1f} ms, "
            f"synthesized={row.synthesized_total_ms:.1f} ms, "
            f"overlap={row.overlap_end_to_end_ms:.1f} ms, "
            f"projected={row.projected_total_ms:.1f} ms"
        )


if __name__ == "__main__":
    main()
