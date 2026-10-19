#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Compare measured ColumnOverlap arms with an ideal-overlap reference.

Two figures are produced: SF=1000 on 8 GPUs and SF=3000 on 24 GPUs.  The first
four bars are clean aggregate Q1-Q22 end-to-end measurements.  The fifth is not
a measured arm: for every baseline query it is the larger of communication and
non-communication time, summed over Q1-Q22.  It models perfect overlap within a
query without allowing work from different queries to overlap.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.patches as mpatches

from plot_distributed_tpch_aggregate_breakdown import (
    REPO_ROOT,
    TIMING_SAMPLES,
    campaign_root,
    communication_fraction_summary,
    plt,
    read_csv,
    require_iterations,
    select_attempt,
)


CONFIGURATIONS = ((1000, 8), (3000, 24))
QUERIES = tuple(range(1, 23))
BASELINE_ARM = "legacy_cta64"
ARMS = (
    ("legacy_cta64", "Baseline", "#4C78A8"),
    ("all_co_nccl", "CO NCCL", "#F58518"),
    ("all_co_nccl_cta64", "CO NCCL\nCTA64", "#54A24B"),
    ("all_co_ce_pull", "CO CE-pull", "#E45756"),
)
REFERENCE_KEY = "legacy_cta64_ideal_overlap_reference"
REFERENCE_LABEL = "Baseline ideal\noverlap"
REFERENCE_COLOR = "#9D9D9D"


@dataclass(frozen=True)
class QueryTiming:
    scale_factor: int
    gpus: int
    arm: str
    query: int
    median_e2e_rank_max_ms: float
    warmup_status: str
    timing_result_status: str
    timing_attempt_dir: str


@dataclass(frozen=True)
class BaselineReferenceQuery:
    scale_factor: int
    gpus: int
    query: int
    median_e2e_rank_max_ms: float
    median_communication_fraction: float
    communication_ms: float
    noncommunication_ms: float
    ideal_overlap_ms: float
    breakdown_result_status: str
    breakdown_attempt_dir: str


@dataclass(frozen=True)
class AggregateBar:
    scale_factor: int
    gpus: int
    bar_key: str
    label: str
    kind: str
    aggregate_time_s: float
    query_count: int


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


def timing_summary_for_figure(
    results_dir: Path,
    scale: int,
    gpus: int,
    query: int,
    arm: str,
) -> tuple[float, str, Path, dict[str, object]]:
    """Read clean timing, accepting one known complete max-warmup result."""
    root = campaign_root(results_dir, scale, gpus, arm) / "timing" / f"q{query:02d}"
    attempt, result = select_attempt(root, ("e2e.csv",), scale, query)
    path = attempt / "e2e.csv"
    rows = read_csv(path)
    require_iterations(rows, TIMING_SAMPLES, path)
    if len(rows) != TIMING_SAMPLES:
        raise ValueError(f"{path}: {len(rows)} rows, expected {TIMING_SAMPLES}")
    statuses = {row["warmup_status"] for row in rows}
    if len(statuses) != 1:
        raise ValueError(f"{path}: inconsistent warmup statuses {statuses}")
    warmup_status = next(iter(statuses))
    known_maximum_reached = (
        scale == 1000
        and gpus == 8
        and query == 9
        and arm == "all_co_nccl_cta64"
        and warmup_status == "maximum_reached"
        and result.get("status") == "pass"
        and result.get("measurements_complete") is True
    )
    if warmup_status != "stable" and not known_maximum_reached:
        raise ValueError(f"{path}: warmup status is {warmup_status!r}")
    values = []
    for row in rows:
        if int(row["scale"]) != scale or int(row["ranks"]) != gpus:
            raise ValueError(f"{path}: scale/rank mismatch")
        value = float(row["rank_max_ms"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{path}: invalid rank_max_ms {value}")
        values.append(value)
    return statistics.median(values), warmup_status, attempt, result


def collect(
    results_dir: Path,
) -> tuple[list[QueryTiming], list[BaselineReferenceQuery], list[AggregateBar]]:
    timing_rows = []
    reference_rows = []
    aggregate_rows = []

    for scale_factor, gpus in CONFIGURATIONS:
        by_arm: dict[str, list[QueryTiming]] = {}
        for arm, label, _ in ARMS:
            rows = []
            for query in QUERIES:
                median_e2e, warmup_status, attempt, result = timing_summary_for_figure(
                    results_dir,
                    scale_factor,
                    gpus,
                    query,
                    arm,
                )
                row = QueryTiming(
                    scale_factor=scale_factor,
                    gpus=gpus,
                    arm=arm,
                    query=query,
                    median_e2e_rank_max_ms=median_e2e,
                    warmup_status=warmup_status,
                    timing_result_status=str(result["status"]),
                    timing_attempt_dir=str(attempt.relative_to(results_dir)),
                )
                rows.append(row)
                timing_rows.append(row)
            if {row.query for row in rows} != set(QUERIES):
                raise ValueError(f"{arm}, {gpus} GPUs: incomplete Q1-Q22 timing")
            by_arm[arm] = rows
            aggregate_rows.append(
                AggregateBar(
                    scale_factor=scale_factor,
                    gpus=gpus,
                    bar_key=arm,
                    label=label.replace("\n", " "),
                    kind="measured_e2e",
                    aggregate_time_s=sum(
                        row.median_e2e_rank_max_ms for row in rows
                    )
                    / 1000.0,
                    query_count=len(rows),
                )
            )

        baseline_by_query = {
            row.query: row for row in by_arm[BASELINE_ARM]
        }
        for query in QUERIES:
            communication_fraction, attempt, result = (
                communication_fraction_summary(
                    results_dir,
                    scale_factor,
                    gpus,
                    query,
                    BASELINE_ARM,
                )
            )
            e2e = baseline_by_query[query].median_e2e_rank_max_ms
            communication = e2e * communication_fraction
            noncommunication = e2e - communication
            reference_rows.append(
                BaselineReferenceQuery(
                    scale_factor=scale_factor,
                    gpus=gpus,
                    query=query,
                    median_e2e_rank_max_ms=e2e,
                    median_communication_fraction=communication_fraction,
                    communication_ms=communication,
                    noncommunication_ms=noncommunication,
                    ideal_overlap_ms=max(noncommunication, communication),
                    breakdown_result_status=str(result["status"]),
                    breakdown_attempt_dir=str(
                        attempt.relative_to(results_dir)
                    ),
                )
            )
        selected_reference = [
            row
            for row in reference_rows
            if row.scale_factor == scale_factor and row.gpus == gpus
        ]
        if {row.query for row in selected_reference} != set(QUERIES):
            raise ValueError(
                f"baseline reference, {gpus} GPUs: incomplete Q1-Q22 coverage"
            )
        aggregate_rows.append(
            AggregateBar(
                scale_factor=scale_factor,
                gpus=gpus,
                bar_key=REFERENCE_KEY,
                label=REFERENCE_LABEL.replace("\n", " "),
                kind="derived_ideal_overlap_reference",
                aggregate_time_s=sum(
                    row.ideal_overlap_ms for row in selected_reference
                )
                / 1000.0,
                query_count=len(selected_reference),
            )
        )

    return timing_rows, reference_rows, aggregate_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot_configuration(
    rows: list[AggregateBar], scale_factor: int, gpus: int, output_dir: Path
) -> None:
    selected = [
        row
        for row in rows
        if row.scale_factor == scale_factor and row.gpus == gpus
    ]
    bar_order = tuple(arm for arm, _, _ in ARMS) + (REFERENCE_KEY,)
    lookup = {row.bar_key: row for row in selected}
    if set(lookup) != set(bar_order):
        raise ValueError(f"{gpus} GPUs: unexpected bar coverage")

    positions = list(range(len(bar_order)))
    values = [lookup[key].aggregate_time_s for key in bar_order]
    labels = [label for _, label, _ in ARMS] + [REFERENCE_LABEL]
    colors = [color for _, _, color in ARMS] + [REFERENCE_COLOR]
    maximum = max(values)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.bar(
        positions,
        values,
        width=0.70,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    reference_bar = bars[-1]
    reference_bar.set_hatch("///")
    reference_bar.set_edgecolor("#666666")
    reference_bar.set_linewidth(1.0)

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + maximum * 0.022,
            f"{value:.2f} s",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", labelsize=10.5, pad=6)
    ax.set_ylabel("Aggregate execution time (s)")
    ax.set_title(
        f"TPC-H Aggregate Time — SF={scale_factor}, {gpus} GPUs", pad=16
    )
    ax.set_ylim(0.0, maximum * 1.17)
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

    reference_handle = mpatches.Patch(
        facecolor=REFERENCE_COLOR,
        edgecolor="#666666",
        hatch="///",
        label="Derived reference (not measured)",
    )
    ax.legend(
        handles=[reference_handle],
        loc="upper right",
        frameon=False,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.22, top=0.89)

    stem = (
        "distributed_tpch_overlap_arm_comparison_"
        f"sf{scale_factor}_{gpus}gpu"
    )
    for suffix in ("pdf", "png"):
        directory = output_dir / suffix
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    timing_rows, reference_rows, aggregate_rows = collect(results_dir)

    stem = "distributed_tpch_overlap_arm_comparison"
    write_csv(
        output_dir / f"{stem}_by_query.csv",
        [asdict(row) for row in timing_rows],
    )
    write_csv(
        output_dir / f"{stem}_baseline_reference_by_query.csv",
        [asdict(row) for row in reference_rows],
    )
    write_csv(
        output_dir / f"{stem}_summary.csv",
        [asdict(row) for row in aggregate_rows],
    )
    for scale_factor, gpus in CONFIGURATIONS:
        plot_configuration(aggregate_rows, scale_factor, gpus, output_dir)

    for row in aggregate_rows:
        print(
            f"SF={row.scale_factor}, {row.gpus} GPUs, {row.bar_key}: "
            f"{row.aggregate_time_s:.3f} s"
        )


if __name__ == "__main__":
    main()
