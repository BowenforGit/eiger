#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot measured optimization stages and their additive-benefit projection.

For SF=1000/8 GPUs and SF=3000/24 GPUs, the figure shows:

* legacy_cta64: measured computation + communication (Baseline),
* synthesized_cta64: measured computation + communication (Opt. comp),
* all_co_ce_pull: measured clean end-to-end time only (Comm-comp overlap), and
* a projection combining synthesized computation with the overlap benefit,
  displayed as end-to-end time rather than as an additive breakdown.

The projection is

    projected_total = synthesized_total
                      - (legacy_total - all_co_ce_pull_total)

or, equivalently,

    projected_total = synthesized_compute
                      + synthesized_communication
                      - observed_overlap_benefit.

SF=3000 Q3 and Q18 are intentionally included when their failed-verification
attempts report complete measurements, matching the other presentation plots.
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
    QUERIES,
    REPO_ROOT,
    AggregateBreakdown,
    collect,
    label_segment,
    plt,
    timing_summary,
    write_dataclass_csv,
)


BASELINE_ARM = "legacy_cta64"
SYNTHESIZED_ARM = "synthesized_cta64"
OVERLAP_ARM = "all_co_ce_pull"
OVERLAP_COLOR = "#59A14F"
FIGURE_NAME = "distributed_tpch_optimization_projection_breakdown"


@dataclass(frozen=True)
class ProjectionSummary:
    scale_factor: int
    gpus: int
    nodes: int
    query_count: int
    baseline_compute_s: float
    baseline_communication_s: float
    baseline_total_s: float
    synthesized_compute_s: float
    synthesized_communication_s: float
    synthesized_total_s: float
    overlap_end_to_end_s: float
    observed_overlap_benefit_s: float
    projected_compute_s: float
    projected_exposed_communication_s: float
    projected_total_s: float


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


def by_scale(rows: list[AggregateBreakdown]) -> dict[int, AggregateBreakdown]:
    result = {row.scale_factor: row for row in rows}
    expected = {scale for scale, _ in CONFIGURATIONS}
    if set(result) != expected:
        raise ValueError(f"unexpected scale coverage: {sorted(set(result) ^ expected)}")
    return result


def collect_overlap_totals(results_dir: Path) -> dict[int, float]:
    totals = {}
    for scale, gpus in CONFIGURATIONS:
        medians = []
        for query in QUERIES:
            median_e2e, _, _ = timing_summary(
                results_dir,
                scale,
                gpus,
                query,
                arm=OVERLAP_ARM,
            )
            medians.append(median_e2e)
        if len(medians) != len(QUERIES):
            raise ValueError(f"SF={scale}: incomplete {OVERLAP_ARM} query set")
        totals[scale] = sum(medians) / 1000.0
    return totals


def collect_projection(results_dir: Path) -> list[ProjectionSummary]:
    _, baseline_rows = collect(results_dir, arm=BASELINE_ARM)
    _, synthesized_rows = collect(results_dir, arm=SYNTHESIZED_ARM)
    baseline = by_scale(baseline_rows)
    synthesized = by_scale(synthesized_rows)
    overlap_totals = collect_overlap_totals(results_dir)

    summaries = []
    for scale, gpus in CONFIGURATIONS:
        baseline_row = baseline[scale]
        synthesized_row = synthesized[scale]
        overlap_total = overlap_totals[scale]
        overlap_benefit = baseline_row.total_s - overlap_total
        projected_residual = (
            synthesized_row.communication_s - overlap_benefit
        )
        if overlap_benefit < 0.0:
            raise ValueError(
                f"SF={scale}: {OVERLAP_ARM} is slower than baseline; "
                "the approved benefit projection is not applicable"
            )
        if projected_residual < 0.0:
            raise ValueError(
                f"SF={scale}: projected exposed communication is negative "
                f"({projected_residual:.6f} s)"
            )
        projected_total = synthesized_row.computation_s + projected_residual
        equivalent_total = synthesized_row.total_s - overlap_benefit
        if not math.isclose(
            projected_total, equivalent_total, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"SF={scale}: projection identities disagree")
        summaries.append(
            ProjectionSummary(
                scale_factor=scale,
                gpus=gpus,
                nodes=gpus // 4,
                query_count=len(QUERIES),
                baseline_compute_s=baseline_row.computation_s,
                baseline_communication_s=baseline_row.communication_s,
                baseline_total_s=baseline_row.total_s,
                synthesized_compute_s=synthesized_row.computation_s,
                synthesized_communication_s=synthesized_row.communication_s,
                synthesized_total_s=synthesized_row.total_s,
                overlap_end_to_end_s=overlap_total,
                observed_overlap_benefit_s=overlap_benefit,
                projected_compute_s=synthesized_row.computation_s,
                projected_exposed_communication_s=projected_residual,
                projected_total_s=projected_total,
            )
        )
    return summaries


def plot(rows: list[ProjectionSummary], output_dir: Path) -> None:
    by_sf = {row.scale_factor: row for row in rows}
    group_centers = [0.0, 2.05]
    offsets = (-0.69, -0.23, 0.23, 0.69)
    tick_labels = (
        "Baseline",
        "Opt. comp",
        "Comm-comp\noverlap",
        "Projected",
    )
    width = 0.40
    positions = []
    labels = []
    bar_specs = []
    for group_index, (scale, _) in enumerate(CONFIGURATIONS):
        row = by_sf[scale]
        scale_specs = (
            (
                "measured",
                row.baseline_compute_s,
                row.baseline_communication_s,
                row.baseline_total_s,
            ),
            (
                "measured",
                row.synthesized_compute_s,
                row.synthesized_communication_s,
                row.synthesized_total_s,
            ),
            ("end_to_end", 0.0, 0.0, row.overlap_end_to_end_s),
            (
                "projected",
                row.projected_compute_s,
                row.projected_exposed_communication_s,
                row.projected_total_s,
            ),
        )
        for offset, label, spec in zip(offsets, tick_labels, scale_specs):
            positions.append(group_centers[group_index] + offset)
            labels.append(label)
            bar_specs.append(spec)

    totals = [spec[3] for spec in bar_specs]
    maximum = max(totals)
    fig, ax = plt.subplots(figsize=(10.3, 5.05))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for position, (kind, computation, communication, total) in zip(
        positions, bar_specs
    ):
        if kind in ("end_to_end", "projected"):
            ax.bar(
                position,
                total,
                width,
                color=OVERLAP_COLOR,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.text(
                position,
                total / 2.0,
                "End-to-end",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )
            if kind == "projected":
                outline = mpatches.Rectangle(
                    (position - width / 2.0, 0.0),
                    width,
                    total,
                    fill=False,
                    edgecolor="#3F3F3F",
                    linewidth=1.1,
                    linestyle=(0, (4, 2)),
                    zorder=4,
                )
                ax.add_patch(outline)
        else:
            ax.bar(
                position,
                computation,
                width,
                color=COLORS["Computation"],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.bar(
                position,
                communication,
                width,
                bottom=computation,
                color=COLORS["Communication"],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            label_segment(ax, position, 0.0, computation, total)
            label_segment(
                ax, position, computation, communication, total
            )
        ax.text(
            position,
            total + maximum * 0.022,
            f"{total:.2f} s",
            ha="center",
            va="bottom",
            fontsize=10.2,
            fontweight="bold",
            color="#2F2F2F",
        )

    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", labelsize=9.5, pad=5)
    for center, (scale, gpus) in zip(group_centers, CONFIGURATIONS):
        ax.text(
            center,
            -0.16,
            f"SF={scale}  •  {gpus} GPUs",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            color="#333333",
        )

    ax.set_ylabel("Aggregate execution time (s)")
    ax.set_title("Distributed TPC-H Aggregate Time Breakdown", pad=14)
    ax.set_ylim(0.0, maximum * 1.23)
    ax.set_xlim(group_centers[0] - 1.04, group_centers[-1] + 1.04)
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
    legend_handles = (
        mpatches.Patch(color=COLORS["Computation"], label="Computation"),
        mpatches.Patch(color=COLORS["Communication"], label="Communication"),
        mpatches.Patch(color=OVERLAP_COLOR, label="End-to-end"),
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.10, right=0.995, top=0.87, bottom=0.25)

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
    summaries = collect_projection(args.results_dir.resolve())
    write_dataclass_csv(
        output_dir / "optimization_projection_summary.csv", summaries
    )
    plot(summaries, output_dir)
    for row in summaries:
        print(
            f"SF={row.scale_factor}, {row.gpus} GPUs: "
            f"baseline={row.baseline_total_s:.3f} s, "
            f"synthesized={row.synthesized_total_s:.3f} s, "
            f"overlap={row.overlap_end_to_end_s:.3f} s, "
            f"benefit={row.observed_overlap_benefit_s:.3f} s, "
            f"projected={row.projected_total_s:.3f} s"
        )


if __name__ == "__main__":
    main()
