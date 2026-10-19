#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot per-query observed communication-hiding effectiveness.

For SF=1000/8 GPUs and SF=3000/24 GPUs, each sample is a query containing at
least one measured ``DistributedEqJoin(...;shuffle;...)`` step:

    100 * (legacy_cta64 end-to-end time - all_co_ce_pull end-to-end time)
          / legacy_cta64 communication time.

Baseline communication time is the attributed value used by the aggregate
optimization-projection breakdown.  The overlap arm contributes only its
clean median end-to-end latency.  Values are not clipped: negative values mean
that the overlap arm is slower than Baseline, while values above 100% would
mean that its improvement exceeds all Baseline-attributed communication.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.lines as mlines
import numpy as np

import plot_distributed_tpch_eqjoin_overlap_arm_comparison as eqjoin_plot
from plot_distributed_tpch_aggregate_breakdown import (
    CONFIGURATIONS,
    QUERIES,
    REPO_ROOT,
    QueryBreakdown,
    collect,
    plt,
    timing_summary,
    write_dataclass_csv,
)


BASELINE_ARM = "legacy_cta64"
OVERLAP_ARM = "all_co_ce_pull"
FIGURE_NAME = "distributed_tpch_communication_hiding_boxplot"
SETTING_COLORS = {
    1000: "#4C78A8",
    3000: "#54A24B",
}


@dataclass(frozen=True)
class QueryCommunicationHiding:
    scale_factor: int
    gpus: int
    nodes: int
    query: int
    shuffle_eqjoin_step_count: int
    baseline_compute_ms: float
    baseline_communication_ms: float
    baseline_total_ms: float
    overlap_total_ms: float
    hidden_communication_ms: float
    overlapped_communication_percent: float
    baseline_timing_status: str
    baseline_breakdown_status: str
    overlap_timing_status: str
    overlap_timing_failure_kind: str
    baseline_timing_attempt_dir: str
    baseline_breakdown_attempt_dir: str
    overlap_timing_attempt_dir: str


@dataclass(frozen=True)
class ConfigurationHidingSummary:
    scale_factor: int
    gpus: int
    nodes: int
    query_count: int
    negative_query_count: int
    above_100_query_count: int
    minimum_percent: float
    first_quartile_percent: float
    median_percent: float
    third_quartile_percent: float
    maximum_percent: float
    arithmetic_mean_percent: float


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


def index_rows(
    rows: list[QueryBreakdown], arm: str
) -> dict[tuple[int, int], QueryBreakdown]:
    indexed = {(row.scale_factor, row.query): row for row in rows}
    expected = {
        (scale_factor, query)
        for scale_factor, _ in CONFIGURATIONS
        for query in QUERIES
    }
    if set(indexed) != expected:
        raise ValueError(
            f"{arm}: unexpected scale/query coverage: "
            f"{sorted(set(indexed) ^ expected)}"
        )
    return indexed


def collect_hiding(rows_dir: Path) -> list[QueryCommunicationHiding]:
    baseline_rows, _ = collect(rows_dir, arm=BASELINE_ARM)
    baseline = index_rows(baseline_rows, BASELINE_ARM)

    rows = []
    for scale_factor, gpus in CONFIGURATIONS:
        for query in QUERIES:
            eqjoin_row = eqjoin_plot.collect_query(
                rows_dir,
                scale_factor,
                gpus,
                BASELINE_ARM,
                query,
            )
            if eqjoin_row.shuffle_eqjoin_step_count == 0:
                continue
            baseline_row = baseline[scale_factor, query]
            if baseline_row.communication_ms <= 0.0:
                raise ValueError(
                    f"Q{query} SF={scale_factor}: non-positive Baseline "
                    f"communication time {baseline_row.communication_ms}"
                )
            attributed_total = (
                baseline_row.computation_ms + baseline_row.communication_ms
            )
            if not math.isclose(
                attributed_total,
                baseline_row.median_e2e_rank_max_ms,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"Q{query} SF={scale_factor}: Baseline attribution does "
                    "not add to clean end-to-end time"
                )

            overlap_total_ms, overlap_attempt, overlap_result = timing_summary(
                rows_dir,
                scale_factor,
                gpus,
                query,
                arm=OVERLAP_ARM,
            )
            hidden_ms = (
                baseline_row.median_e2e_rank_max_ms - overlap_total_ms
            )
            hidden_percent = 100.0 * hidden_ms / baseline_row.communication_ms
            if not math.isfinite(hidden_percent):
                raise ValueError(
                    f"Q{query} SF={scale_factor}: invalid hiding percentage "
                    f"{hidden_percent}"
                )
            rows.append(
                QueryCommunicationHiding(
                    scale_factor=scale_factor,
                    gpus=gpus,
                    nodes=gpus // 4,
                    query=query,
                    shuffle_eqjoin_step_count=(
                        eqjoin_row.shuffle_eqjoin_step_count
                    ),
                    baseline_compute_ms=baseline_row.computation_ms,
                    baseline_communication_ms=baseline_row.communication_ms,
                    baseline_total_ms=baseline_row.median_e2e_rank_max_ms,
                    overlap_total_ms=overlap_total_ms,
                    hidden_communication_ms=hidden_ms,
                    overlapped_communication_percent=hidden_percent,
                    baseline_timing_status=baseline_row.timing_result_status,
                    baseline_breakdown_status=(
                        baseline_row.breakdown_result_status
                    ),
                    overlap_timing_status=str(overlap_result.get("status", "")),
                    overlap_timing_failure_kind=str(
                        overlap_result.get("failure_kind", "")
                    ),
                    baseline_timing_attempt_dir=(
                        baseline_row.timing_attempt_dir
                    ),
                    baseline_breakdown_attempt_dir=(
                        baseline_row.breakdown_attempt_dir
                    ),
                    overlap_timing_attempt_dir=str(
                        overlap_attempt.relative_to(rows_dir)
                    ),
                )
            )
    return rows


def summarize(
    rows: list[QueryCommunicationHiding],
) -> list[ConfigurationHidingSummary]:
    summaries = []
    query_sets = {
        scale_factor: {
            row.query for row in rows if row.scale_factor == scale_factor
        }
        for scale_factor, _ in CONFIGURATIONS
    }
    if len({frozenset(queries) for queries in query_sets.values()}) != 1:
        raise ValueError(
            f"shuffle-query sets differ across configurations: {query_sets}"
        )
    for scale_factor, gpus in CONFIGURATIONS:
        values = np.asarray(
            [
                row.overlapped_communication_percent
                for row in rows
                if row.scale_factor == scale_factor
            ],
            dtype=float,
        )
        expected_count = len(query_sets[scale_factor])
        if expected_count == 0 or len(values) != expected_count:
            raise ValueError(
                f"SF={scale_factor}: expected {expected_count} samples, "
                f"found {len(values)}"
            )
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        summaries.append(
            ConfigurationHidingSummary(
                scale_factor=scale_factor,
                gpus=gpus,
                nodes=gpus // 4,
                query_count=len(values),
                negative_query_count=int(np.count_nonzero(values < 0.0)),
                above_100_query_count=int(np.count_nonzero(values > 100.0)),
                minimum_percent=float(np.min(values)),
                first_quartile_percent=float(q1),
                median_percent=float(median),
                third_quartile_percent=float(q3),
                maximum_percent=float(np.max(values)),
                arithmetic_mean_percent=float(np.mean(values)),
            )
        )
    return summaries


def plot(rows: list[QueryCommunicationHiding], output_dir: Path) -> None:
    configurations = list(CONFIGURATIONS)
    positions = [2.0, 1.0]
    data = [
        np.asarray(
            [
                row.overlapped_communication_percent
                for row in rows
                if row.scale_factor == scale_factor
            ],
            dtype=float,
        )
        for scale_factor, _ in configurations
    ]
    query_sets = [
        {
            row.query
            for row in rows
            if row.scale_factor == scale_factor
        }
        for scale_factor, _ in configurations
    ]
    if (
        any(not queries for queries in query_sets)
        or len({frozenset(queries) for queries in query_sets}) != 1
        or any(len(values) != len(queries) for values, queries in zip(data, query_sets))
    ):
        raise ValueError("incomplete per-query communication-hiding samples")

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    boxplot = ax.boxplot(
        data,
        orientation="horizontal",
        positions=positions,
        widths=0.48,
        whis=1.5,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
        manage_ticks=False,
        medianprops={"color": "#303030", "linewidth": 2.0},
        whiskerprops={"color": "#555555", "linewidth": 1.3},
        capprops={"color": "#555555", "linewidth": 1.3},
    )
    for box, (scale_factor, _) in zip(boxplot["boxes"], configurations):
        box.set_facecolor(SETTING_COLORS[scale_factor])
        box.set_edgecolor("#555555")
        box.set_linewidth(1.0)
        box.set_alpha(0.42)

    random = np.random.default_rng(20260819)
    for position, values, (scale_factor, _) in zip(
        positions, data, configurations
    ):
        jitter = random.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(
            values,
            position + jitter,
            s=37,
            color=SETTING_COLORS[scale_factor],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.92,
            zorder=4,
        )
        median = float(np.median(values))
        ax.annotate(
            f"Median: {median:.1f}%",
            xy=(median, position + 0.24),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color="#303030",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
            },
            zorder=6,
        )

    all_values = np.concatenate(data)
    value_span = max(float(np.max(all_values)) - float(np.min(all_values)), 20.0)
    left = min(float(np.min(all_values)) - 0.09 * value_span, -8.0)
    right = max(float(np.max(all_values)) + 0.09 * value_span, 108.0)
    ax.set_xlim(left, right)
    ax.set_ylim(0.55, 2.45)
    for value, label, alignment in (
        (0.0, "0%\nNone hidden", "center"),
        (100.0, "100%\nFully hidden", "center"),
    ):
        ax.axvline(
            value,
            color="#777777",
            linewidth=1.25,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
        ax.text(
            value,
            0.985,
            label,
            transform=ax.get_xaxis_transform(),
            ha=alignment,
            va="top",
            fontsize=8.4,
            linespacing=1.0,
            color="#666666",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )

    ax.set_yticks(
        positions,
        [
            f"SF={scale_factor}\n{gpus} GPUs"
            for scale_factor, gpus in configurations
        ],
    )
    ax.set_xlabel("Baseline communication hidden by overlap (%)")
    ax.set_title(
        "TPC-H Communication-Hiding Efficiency — Shuffle-Join Queries",
        pad=16,
    )
    ax.grid(
        axis="x",
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
    point_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=6,
        markerfacecolor="#666666",
        markeredgecolor="white",
        label="Query",
    )
    median_handle = mlines.Line2D(
        [],
        [],
        color="#303030",
        linewidth=2.0,
        label="Median",
    )
    ax.legend(
        handles=(point_handle, median_handle),
        loc="lower left",
        frameon=False,
        ncol=2,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.23, top=0.80)

    for suffix in ("pdf", "png"):
        directory = output_dir / suffix
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{FIGURE_NAME}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    rows = collect_hiding(rows_dir)
    summaries = summarize(rows)
    write_dataclass_csv(
        output_dir / f"{FIGURE_NAME}_by_query.csv",
        rows,
    )
    write_dataclass_csv(
        output_dir / f"{FIGURE_NAME}_summary.csv",
        summaries,
    )
    plot(rows, output_dir)


if __name__ == "__main__":
    main()
