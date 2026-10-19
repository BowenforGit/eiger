#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot per-query compute-time speedup from optimized computation.

For SF=1000/8 GPUs and SF=3000/24 GPUs, each sample is one TPC-H query's
compute-time speedup:

    legacy_cta64 computation_ms / synthesized_cta64 computation_ms.

The computation times are the same values used for the blue segments in the
aggregate optimization-projection breakdown: clean median end-to-end latency
multiplied by the measured non-communication fraction.  Q3 and Q18 are kept
when their verification-failed attempts contain complete measurements.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.lines as mlines
import numpy as np

from plot_distributed_tpch_aggregate_breakdown import (
    CONFIGURATIONS,
    QUERIES,
    REPO_ROOT,
    QueryBreakdown,
    collect,
    plt,
    write_dataclass_csv,
)


BASELINE_ARM = "legacy_cta64"
OPTIMIZED_ARM = "synthesized_cta64"
FIGURE_NAME = "distributed_tpch_opt_comp_compute_speedup_boxplot"
SETTING_COLORS = {
    1000: "#7F3C8D",
    3000: "#11A579",
}


@dataclass(frozen=True)
class QueryComputeSpeedup:
    scale_factor: int
    gpus: int
    nodes: int
    query: int
    baseline_compute_ms: float
    optimized_compute_ms: float
    compute_speedup: float
    baseline_timing_status: str
    baseline_breakdown_status: str
    optimized_timing_status: str
    optimized_breakdown_status: str
    baseline_timing_attempt_dir: str
    baseline_breakdown_attempt_dir: str
    optimized_timing_attempt_dir: str
    optimized_breakdown_attempt_dir: str


@dataclass(frozen=True)
class ConfigurationSpeedupSummary:
    scale_factor: int
    gpus: int
    nodes: int
    query_count: int
    minimum: float
    first_quartile: float
    median: float
    third_quartile: float
    maximum: float
    arithmetic_mean: float
    geometric_mean: float


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


def collect_speedups(results_dir: Path) -> list[QueryComputeSpeedup]:
    baseline_rows, _ = collect(results_dir, arm=BASELINE_ARM)
    optimized_rows, _ = collect(results_dir, arm=OPTIMIZED_ARM)
    baseline = index_rows(baseline_rows, BASELINE_ARM)
    optimized = index_rows(optimized_rows, OPTIMIZED_ARM)

    rows = []
    for scale_factor, gpus in CONFIGURATIONS:
        for query in QUERIES:
            baseline_row = baseline[scale_factor, query]
            optimized_row = optimized[scale_factor, query]
            if baseline_row.computation_ms <= 0.0:
                raise ValueError(
                    f"Q{query} SF={scale_factor}: non-positive baseline "
                    f"compute time {baseline_row.computation_ms}"
                )
            if optimized_row.computation_ms <= 0.0:
                raise ValueError(
                    f"Q{query} SF={scale_factor}: non-positive optimized "
                    f"compute time {optimized_row.computation_ms}"
                )
            speedup = baseline_row.computation_ms / optimized_row.computation_ms
            if not math.isfinite(speedup) or speedup <= 0.0:
                raise ValueError(
                    f"Q{query} SF={scale_factor}: invalid speedup {speedup}"
                )
            rows.append(
                QueryComputeSpeedup(
                    scale_factor=scale_factor,
                    gpus=gpus,
                    nodes=gpus // 4,
                    query=query,
                    baseline_compute_ms=baseline_row.computation_ms,
                    optimized_compute_ms=optimized_row.computation_ms,
                    compute_speedup=speedup,
                    baseline_timing_status=baseline_row.timing_result_status,
                    baseline_breakdown_status=baseline_row.breakdown_result_status,
                    optimized_timing_status=optimized_row.timing_result_status,
                    optimized_breakdown_status=optimized_row.breakdown_result_status,
                    baseline_timing_attempt_dir=baseline_row.timing_attempt_dir,
                    baseline_breakdown_attempt_dir=baseline_row.breakdown_attempt_dir,
                    optimized_timing_attempt_dir=optimized_row.timing_attempt_dir,
                    optimized_breakdown_attempt_dir=optimized_row.breakdown_attempt_dir,
                )
            )
    return rows


def summarize(
    rows: list[QueryComputeSpeedup],
) -> list[ConfigurationSpeedupSummary]:
    summaries = []
    for scale_factor, gpus in CONFIGURATIONS:
        values = np.asarray(
            [
                row.compute_speedup
                for row in rows
                if row.scale_factor == scale_factor
            ],
            dtype=float,
        )
        if len(values) != len(QUERIES):
            raise ValueError(
                f"SF={scale_factor}: expected {len(QUERIES)} samples, "
                f"found {len(values)}"
            )
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        summaries.append(
            ConfigurationSpeedupSummary(
                scale_factor=scale_factor,
                gpus=gpus,
                nodes=gpus // 4,
                query_count=len(values),
                minimum=float(np.min(values)),
                first_quartile=float(q1),
                median=float(median),
                third_quartile=float(q3),
                maximum=float(np.max(values)),
                arithmetic_mean=float(np.mean(values)),
                geometric_mean=float(np.exp(np.mean(np.log(values)))),
            )
        )
    return summaries


def plot(rows: list[QueryComputeSpeedup], output_dir: Path) -> None:
    configurations = list(CONFIGURATIONS)
    positions = [2.0, 1.0]
    data = [
        np.asarray(
            [
                row.compute_speedup
                for row in rows
                if row.scale_factor == scale_factor
            ],
            dtype=float,
        )
        for scale_factor, _ in configurations
    ]
    if any(len(values) != len(QUERIES) for values in data):
        raise ValueError("incomplete per-query speedup samples")

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

    all_values = np.concatenate(data)
    span = max(float(np.max(all_values)) - float(np.min(all_values)), 0.2)
    margin = 0.12 * span
    left = min(float(np.min(all_values)) - margin, 0.96)
    right = float(np.max(all_values)) + margin
    ax.set_xlim(left, right)
    ax.set_ylim(0.55, 2.45)
    for position, values in zip(positions, data):
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        upper_fence = q3 + 1.5 * (q3 - q1)
        upper_whisker = float(np.max(values[values <= upper_fence]))
        ax.annotate(
            f"Median: {median:.2f}×",
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
        ax.annotate(
            f"{upper_whisker:.2f}×",
            xy=(upper_whisker, position),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.2,
            color="#444444",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
            },
            zorder=6,
        )
    ax.axvline(
        1.0,
        color="#777777",
        linewidth=1.25,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    ax.text(
        1.0,
        0.985,
        "No speedup",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9.5,
        color="#666666",
        bbox={
            "boxstyle": "round,pad=0.15",
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
    ax.set_xlabel("Compute-time speedup over Baseline (×)")
    ax.set_title(
        "TPC-H Compute-Time Speedup: Baseline vs. Opt. Comp",
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
        loc="lower right",
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
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    rows = collect_speedups(results_dir)
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
