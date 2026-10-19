#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot a per-step time breakdown for every distributed TPC-H query.

The single 4x6 figure contains Q1-Q22.  Each panel compares SF=1000 on 8,
16, 24, and 32 GPUs with SF=3000 on 24 and 32 GPUs.  Every physical-plan step
is retained as a separate stack segment; color identifies the step's operator
kind.  Step fractions are measured on each breakdown sample's critical rank,
median-aggregated, normalized, and scaled to the clean timing median.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import statistics

import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

import plot_distributed_tpch_gpu_scaling_aggregate_breakdown as scaling
from plot_distributed_tpch_aggregate_breakdown import (
    BREAKDOWN_SAMPLES,
    QUERIES,
    REPO_ROOT,
    QueryBreakdown,
    plt,
)


FIGURE_NAME = "legacy_cta64_gpu_scaling_per_query_step_breakdown"
CONFIGURATIONS = scaling.CONFIGURATIONS

STEP_COLORS = {
    "Filter": "#4C78A8",
    "Project": "#F58518",
    "Reorder": "#A7A7A7",
    "DistributedEqJoin": "#54A24B",
    "DistributedGroupBy": "#E45756",
    "DistributedSort": "#7A6FBA",
    "DistributedTopK": "#9D755D",
    "DistributedScalarAgg": "#D37295",
    "DistributedSemiJoin": "#4CB6AC",
    "DistributedAntiJoin": "#B5A642",
    "DistributedCrossJoin": "#EECA3B",
    "DistributedDistinct": "#2F4B7C",
    "Concat": "#76B7D2",
}

STEP_LABELS = {
    "Filter": "Filter",
    "Project": "Project",
    "Reorder": "Reorder",
    "DistributedEqJoin": "EqJoin",
    "DistributedGroupBy": "GroupBy",
    "DistributedSort": "Sort",
    "DistributedTopK": "TopK",
    "DistributedScalarAgg": "ScalarAgg",
    "DistributedSemiJoin": "SemiJoin",
    "DistributedAntiJoin": "AntiJoin",
    "DistributedCrossJoin": "CrossJoin",
    "DistributedDistinct": "Distinct",
    "Concat": "Concat",
}


@dataclass(frozen=True)
class StepBreakdown:
    scale_factor: int
    gpus: int
    query: int
    step_index: int
    step_kind: str
    step_label: str
    median_raw_step_ms: float
    normalized_step_fraction: float
    attributed_step_ms: float
    median_e2e_rank_max_ms: float
    breakdown_attempt_dir: str


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_one(
    query_row: QueryBreakdown, results_dir: Path
) -> list[StepBreakdown]:
    attempt = results_dir / query_row.breakdown_attempt_dir
    path = attempt / "steps.csv"
    rows = read_csv(path)

    durations: dict[tuple[str, int, int, int], float] = {}
    rank_totals: dict[tuple[str, int, int], float] = defaultdict(float)
    metadata: dict[int, tuple[str, str]] = {}
    samples: set[tuple[str, int]] = set()

    for row in rows:
        if row["record_type"] != "rank":
            continue
        scale_factor = int(row["scale"])
        gpus = int(row["ranks"])
        if (
            scale_factor != query_row.scale_factor
            or gpus != query_row.gpus
        ):
            raise ValueError(f"{path}: scale/rank mismatch")
        timestamp = row["timestamp_utc"]
        iteration = int(row["iteration"])
        rank = int(row["rank"])
        step_index = int(row["step_index"])
        key = (timestamp, iteration, rank, step_index)
        if key in durations:
            raise ValueError(f"{path}: duplicate rank step {key}")
        duration = float(row["gpu_ms"])
        if duration < 0.0:
            raise ValueError(f"{path}: negative step duration")
        durations[key] = duration
        rank_totals[key[:3]] += duration
        samples.add((timestamp, iteration))

        current = (row["step_kind"], row["step_label"])
        previous = metadata.setdefault(step_index, current)
        if previous != current:
            raise ValueError(
                f"{path}: inconsistent metadata for step {step_index}"
            )

    if len(samples) != BREAKDOWN_SAMPLES:
        raise ValueError(
            f"{path}: {len(samples)} samples, expected {BREAKDOWN_SAMPLES}"
        )
    if not metadata:
        raise ValueError(f"{path}: no rank step rows")
    critical: dict[tuple[str, int], tuple[int, float]] = {}
    for (timestamp, iteration, rank), total in rank_totals.items():
        sample = (timestamp, iteration)
        current = critical.get(sample)
        if current is None or total > current[1] or (
            total == current[1] and rank < current[0]
        ):
            critical[sample] = (rank, total)

    fractions: dict[int, list[float]] = defaultdict(list)
    raw_durations: dict[int, list[float]] = defaultdict(list)
    for sample, (rank, total) in critical.items():
        if total <= 0.0:
            raise ValueError(f"{path}: non-positive critical-rank total")
        timestamp, iteration = sample
        for step_index in sorted(metadata):
            key = (timestamp, iteration, rank, step_index)
            if key not in durations:
                raise ValueError(f"{path}: missing critical-rank step {key}")
            duration = durations[key]
            raw_durations[step_index].append(duration)
            fractions[step_index].append(duration / total)

    median_fractions = {
        step_index: statistics.median(values)
        for step_index, values in fractions.items()
    }
    fraction_sum = sum(median_fractions.values())
    if fraction_sum <= 0.0:
        raise ValueError(f"{path}: invalid median step fractions")

    result = []
    for step_index in sorted(metadata):
        step_kind, step_label = metadata[step_index]
        if step_kind not in STEP_COLORS:
            raise ValueError(f"{path}: no color for {step_kind}")
        normalized_fraction = median_fractions[step_index] / fraction_sum
        result.append(
            StepBreakdown(
                scale_factor=query_row.scale_factor,
                gpus=query_row.gpus,
                query=query_row.query,
                step_index=step_index,
                step_kind=step_kind,
                step_label=step_label,
                median_raw_step_ms=statistics.median(
                    raw_durations[step_index]
                ),
                normalized_step_fraction=normalized_fraction,
                attributed_step_ms=(
                    query_row.median_e2e_rank_max_ms * normalized_fraction
                ),
                median_e2e_rank_max_ms=query_row.median_e2e_rank_max_ms,
                breakdown_attempt_dir=query_row.breakdown_attempt_dir,
            )
        )
    return result


def collect(
    query_rows: list[QueryBreakdown], results_dir: Path
) -> list[StepBreakdown]:
    expected = {
        (query, scale_factor, gpus)
        for query in QUERIES
        for scale_factor, gpus in CONFIGURATIONS
    }
    actual = {
        (row.query, row.scale_factor, row.gpus) for row in query_rows
    }
    if actual != expected:
        raise ValueError(
            f"unexpected query/configuration coverage: "
            f"{sorted(actual ^ expected)}"
        )

    result = []
    for row in query_rows:
        result.extend(collect_one(row, results_dir))
    return result


def write_rows(rows: list[StepBreakdown], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{FIGURE_NAME}.csv"
    fieldnames = list(asdict(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    print(f"wrote {path}")
    return path


def plot(rows: list[StepBreakdown], output_dir: Path) -> None:
    grouped: dict[tuple[int, int, int], list[StepBreakdown]] = defaultdict(list)
    for row in rows:
        grouped[(row.query, row.scale_factor, row.gpus)].append(row)

    positions = (0.0, 1.0, 2.0, 3.0, 4.7, 5.7)
    tick_labels = tuple(
        f"{scale_factor // 1000}K/{gpus}"
        for scale_factor, gpus in CONFIGURATIONS
    )
    fig, axes = plt.subplots(4, 6, figsize=(20.0, 13.0))
    fig.patch.set_facecolor("white")

    for ax, query in zip(axes.flat, QUERIES, strict=False):
        ax.set_facecolor("white")
        totals = []
        query_step_sequences = []
        for position, (scale_factor, gpus) in zip(
            positions, CONFIGURATIONS, strict=True
        ):
            step_rows = sorted(
                grouped[(query, scale_factor, gpus)],
                key=lambda row: row.step_index,
            )
            if not step_rows:
                raise ValueError(
                    f"Q{query}, SF={scale_factor}, {gpus} GPUs: missing steps"
                )
            query_step_sequences.append(
                tuple((row.step_index, row.step_kind) for row in step_rows)
            )
            bottom = 0.0
            for row in step_rows:
                ax.bar(
                    position,
                    row.attributed_step_ms,
                    width=0.74,
                    bottom=bottom,
                    color=STEP_COLORS[row.step_kind],
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )
                bottom += row.attributed_step_ms
            totals.append(bottom)

        if len(set(query_step_sequences)) != 1:
            raise ValueError(f"Q{query}: plan-step sequence differs by config")

        maximum = max(totals)
        for position, total in zip(positions, totals, strict=True):
            label = f"{total:.0f}" if maximum >= 20.0 else f"{total:.1f}"
            ax.text(
                position,
                total + maximum * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=6.2,
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

    present_kinds = [
        kind for kind in STEP_COLORS if any(row.step_kind == kind for row in rows)
    ]
    handles = tuple(
        mpatches.Patch(
            facecolor=STEP_COLORS[kind], label=STEP_LABELS[kind]
        )
        for kind in present_kinds
    )
    fig.suptitle(
        "Distributed TPC-H Per-Query Step-Time Breakdown",
        fontsize=20,
        y=0.992,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.963),
        ncol=7,
        frameon=False,
        fontsize=10.2,
        handlelength=1.3,
        columnspacing=1.2,
        labelspacing=0.65,
    )
    fig.supylabel("Execution time (ms)", fontsize=14, x=0.012)
    fig.supxlabel(
        "Configuration (scale factor / GPUs)", fontsize=13, y=0.018
    )
    fig.subplots_adjust(
        left=0.045,
        right=0.995,
        bottom=0.065,
        top=0.885,
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
    step_rows = collect(query_rows, results_dir)
    write_rows(step_rows, output_dir)
    plot(step_rows, output_dir)


if __name__ == "__main__":
    main()
