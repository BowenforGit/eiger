#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Compare shuffle DistributedEqJoin time across communication-overlap arms.

The two figures cover SF=1000 on 8 GPUs and SF=3000 on 24 GPUs.  The first
four bars are full shuffle-strategy DistributedEqJoin physical-step time
measured in the breakdown runs.  Broadcast-left and broadcast-right joins are
excluded.  The fifth bar is the directly measured compute-phase time inside
legacy_cta64 shuffle DistributedEqJoin steps.  For each join instance, that
reference is residual time plus max(compute time, communication time), so only
its compute and communication stages are allowed to overlap.  No values are
scaled to clean end-to-end latency.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.patches as mpatches

import plot_distributed_tpch_overlap_arm_comparison as total_plot
from plot_distributed_tpch_aggregate_breakdown import (
    BREAKDOWN_SAMPLES,
    REPO_ROOT,
    campaign_root,
    plt,
    read_csv,
    require_iterations,
    select_attempt,
)


QUERIES = tuple(range(1, 23))
TARGET_OPERATOR = "DistributedEqJoin"
FIGURE_PREFIX = "distributed_tpch_eqjoin_overlap_arm_comparison"


@dataclass(frozen=True)
class QueryEqJoinTime:
    scale_factor: int
    gpus: int
    arm: str
    query: int
    selected_timestamp_utc: str
    selected_iteration: int
    critical_rank: int
    critical_rank_total_step_ms: float
    shuffle_eqjoin_step_count: int
    shuffle_eqjoin_total_ms: float
    shuffle_eqjoin_compute_ms: float | None
    shuffle_eqjoin_communication_ms: float | None
    shuffle_eqjoin_residual_ms: float | None
    shuffle_eqjoin_ideal_overlap_ms: float | None
    breakdown_result_status: str
    failure_kind: str
    steps_csv: str
    phases_csv: str


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


def interval_union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def collect_query(
    results_dir: Path,
    scale_factor: int,
    gpus: int,
    arm: str,
    query: int,
) -> QueryEqJoinTime:
    root = (
        campaign_root(results_dir, scale_factor, gpus, arm)
        / "breakdown"
        / f"q{query:02d}"
    )
    attempt, result = select_attempt(
        root, ("steps.csv", "phases.csv"), scale_factor, query
    )
    steps_path = attempt / "steps.csv"
    phases_path = attempt / "phases.csv"
    steps = read_csv(steps_path)
    phases = read_csv(phases_path)
    require_iterations(steps, BREAKDOWN_SAMPLES, steps_path)
    require_iterations(phases, BREAKDOWN_SAMPLES, phases_path)

    rank_totals: dict[tuple[str, int, int], float] = defaultdict(float)
    step_duration: dict[tuple[str, int, int, int], float] = {}
    step_kind: dict[tuple[str, int, int, int], str] = {}
    step_label: dict[tuple[str, int, int, int], str] = {}
    for row in steps:
        if row["record_type"] != "rank":
            continue
        if int(row["scale"]) != scale_factor or int(row["ranks"]) != gpus:
            raise ValueError(f"{steps_path}: scale/rank mismatch")
        key = (
            row["timestamp_utc"],
            int(row["iteration"]),
            int(row["rank"]),
            int(row["step_index"]),
        )
        if key in step_duration:
            raise ValueError(f"{steps_path}: duplicate rank step {key}")
        duration = float(row["gpu_ms"])
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError(f"{steps_path}: invalid gpu_ms {duration}")
        step_duration[key] = duration
        step_kind[key] = row["step_kind"]
        step_label[key] = row["step_label"]
        rank_totals[key[:3]] += duration

    critical: dict[tuple[str, int], tuple[int, float]] = {}
    for (timestamp, iteration, rank), total in rank_totals.items():
        sample_key = (timestamp, iteration)
        current = critical.get(sample_key)
        if current is None or total > current[1] or (
            total == current[1] and rank < current[0]
        ):
            critical[sample_key] = (rank, total)
    if len(critical) != BREAKDOWN_SAMPLES:
        raise ValueError(
            f"{steps_path}: found {len(critical)} critical-rank samples, "
            f"expected {BREAKDOWN_SAMPLES}"
        )

    candidates = sorted(
        (total, timestamp, iteration, rank)
        for (timestamp, iteration), (rank, total) in critical.items()
    )
    selected_total, timestamp, iteration, rank = candidates[len(candidates) // 2]
    selected_prefix = (timestamp, iteration, rank)
    selected_eqjoin_steps = {
        key: duration
        for key, duration in step_duration.items()
        if key[:3] == selected_prefix
        and step_kind[key] == TARGET_OPERATOR
        and ";shuffle" in step_label[key]
    }
    eqjoin_total = sum(selected_eqjoin_steps.values())

    eqjoin_compute: float | None = None
    eqjoin_communication: float | None = None
    eqjoin_residual: float | None = None
    eqjoin_ideal_overlap: float | None = None
    if arm == total_plot.BASELINE_ARM:
        compute_intervals: dict[
            tuple[str, int, int, int], list[tuple[float, float]]
        ] = defaultdict(list)
        communication_intervals: dict[
            tuple[str, int, int, int], list[tuple[float, float]]
        ] = defaultdict(list)
        for row in phases:
            if (
                row["record_type"] != "rank"
                or row["quality"] != "direct"
                or row["category"] not in {"compute", "communication"}
                or row["step_kind"] != TARGET_OPERATOR
                or ";shuffle" not in row["step_label"]
                or row["timestamp_utc"] != timestamp
                or int(row["iteration"]) != iteration
                or int(row["rank"]) != rank
            ):
                continue
            key = (timestamp, iteration, rank, int(row["step_index"]))
            if key not in selected_eqjoin_steps:
                raise ValueError(
                    f"{phases_path}: compute phase refers to missing EqJoin step {key}"
                )
            duration = selected_eqjoin_steps[key]
            start = max(0.0, float(row["start_ms"]))
            end = min(duration, float(row["end_ms"]))
            if start < end:
                target = (
                    compute_intervals
                    if row["category"] == "compute"
                    else communication_intervals
                )
                target[key].append((start, end))

        eqjoin_compute = 0.0
        eqjoin_communication = 0.0
        eqjoin_residual = 0.0
        eqjoin_ideal_overlap = 0.0
        for key, duration in selected_eqjoin_steps.items():
            compute = interval_union(compute_intervals[key])
            communication = interval_union(communication_intervals[key])
            covered = interval_union(
                compute_intervals[key] + communication_intervals[key]
            )
            tolerance = max(0.05, duration * 0.01)
            if covered > duration + tolerance:
                raise ValueError(
                    f"{phases_path}: covered phase time {covered:.6f} ms "
                    f"exceeds EqJoin step {key} duration {duration:.6f} ms"
                )
            residual = max(0.0, duration - min(duration, covered))
            ideal_overlap = residual + max(compute, communication)
            eqjoin_compute += compute
            eqjoin_communication += communication
            eqjoin_residual += residual
            eqjoin_ideal_overlap += ideal_overlap

        tolerance = max(0.05, eqjoin_total * 0.01)
        if eqjoin_ideal_overlap > eqjoin_total + tolerance:
            raise ValueError(
                f"{phases_path}: EqJoin ideal overlap "
                f"{eqjoin_ideal_overlap:.6f} ms "
                f"exceeds EqJoin total {eqjoin_total:.6f} ms"
            )

    return QueryEqJoinTime(
        scale_factor=scale_factor,
        gpus=gpus,
        arm=arm,
        query=query,
        selected_timestamp_utc=timestamp,
        selected_iteration=iteration,
        critical_rank=rank,
        critical_rank_total_step_ms=selected_total,
        shuffle_eqjoin_step_count=len(selected_eqjoin_steps),
        shuffle_eqjoin_total_ms=eqjoin_total,
        shuffle_eqjoin_compute_ms=eqjoin_compute,
        shuffle_eqjoin_communication_ms=eqjoin_communication,
        shuffle_eqjoin_residual_ms=eqjoin_residual,
        shuffle_eqjoin_ideal_overlap_ms=eqjoin_ideal_overlap,
        breakdown_result_status=str(result["status"]),
        failure_kind=str(result.get("failure_kind") or ""),
        steps_csv=str(steps_path.relative_to(results_dir)),
        phases_csv=str(phases_path.relative_to(results_dir)),
    )


def collect(
    results_dir: Path,
) -> tuple[list[QueryEqJoinTime], list[AggregateBar]]:
    query_rows = [
        collect_query(results_dir, scale_factor, gpus, arm, query)
        for scale_factor, gpus in total_plot.CONFIGURATIONS
        for arm, _, _ in total_plot.ARMS
        for query in QUERIES
    ]
    aggregate_rows = []
    for scale_factor, gpus in total_plot.CONFIGURATIONS:
        for arm, label, _ in total_plot.ARMS:
            selected = [
                row
                for row in query_rows
                if row.scale_factor == scale_factor
                and row.gpus == gpus
                and row.arm == arm
            ]
            if {row.query for row in selected} != set(QUERIES):
                raise ValueError(
                    f"SF={scale_factor}, {gpus} GPUs, {arm}: "
                    "incomplete Q1-Q22 coverage"
                )
            aggregate_rows.append(
                AggregateBar(
                    scale_factor=scale_factor,
                    gpus=gpus,
                    bar_key=arm,
                    label=label.replace("\n", " "),
                    kind="measured_shuffle_eqjoin_step_time",
                    aggregate_time_s=sum(
                        row.shuffle_eqjoin_total_ms for row in selected
                    )
                    / 1000.0,
                    query_count=len(selected),
                )
            )

        baseline = [
            row
            for row in query_rows
            if row.scale_factor == scale_factor
            and row.gpus == gpus
            and row.arm == total_plot.BASELINE_ARM
        ]
        if any(row.shuffle_eqjoin_ideal_overlap_ms is None for row in baseline):
            raise ValueError("baseline shuffle EqJoin ideal-overlap time is missing")
        aggregate_rows.append(
            AggregateBar(
                scale_factor=scale_factor,
                gpus=gpus,
                bar_key=total_plot.REFERENCE_KEY,
                label="Baseline shuffle EqJoin ideal overlap",
                kind="derived_baseline_shuffle_eqjoin_ideal_overlap_reference",
                aggregate_time_s=sum(
                    row.shuffle_eqjoin_ideal_overlap_ms or 0.0
                    for row in baseline
                )
                / 1000.0,
                query_count=len(baseline),
            )
        )
    return query_rows, aggregate_rows


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
    rows: list[AggregateBar],
    scale_factor: int,
    gpus: int,
    output_dir: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row.scale_factor == scale_factor and row.gpus == gpus
    ]
    bar_order = tuple(arm for arm, _, _ in total_plot.ARMS) + (
        total_plot.REFERENCE_KEY,
    )
    lookup = {row.bar_key: row for row in selected}
    if set(lookup) != set(bar_order):
        raise ValueError(f"SF={scale_factor}, {gpus} GPUs: unexpected bars")

    positions = list(range(len(bar_order)))
    values = [lookup[key].aggregate_time_s for key in bar_order]
    labels = [label for _, label, _ in total_plot.ARMS] + [
        "Baseline shuffle\nEqJoin ideal overlap"
    ]
    colors = [color for _, _, color in total_plot.ARMS] + [
        total_plot.REFERENCE_COLOR
    ]
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
    ax.tick_params(axis="x", labelsize=10.0, pad=6)
    ax.set_ylabel("Aggregate shuffle EqJoin time (s)")
    ax.set_title(
        f"TPC-H Shuffle EqJoin Time — SF={scale_factor}, {gpus} GPUs",
        pad=16,
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
        facecolor=total_plot.REFERENCE_COLOR,
        edgecolor="#666666",
        hatch="///",
        label="Ideal-overlap reference",
    )
    ax.legend(
        handles=[reference_handle],
        loc="upper right",
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.23, top=0.89)

    stem = f"{FIGURE_PREFIX}_sf{scale_factor}_{gpus}gpu"
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
    query_rows, aggregate_rows = collect(results_dir)
    write_csv(
        output_dir / f"{FIGURE_PREFIX}_by_query.csv",
        [asdict(row) for row in query_rows],
    )
    write_csv(
        output_dir / f"{FIGURE_PREFIX}_summary.csv",
        [asdict(row) for row in aggregate_rows],
    )
    for scale_factor, gpus in total_plot.CONFIGURATIONS:
        plot_configuration(aggregate_rows, scale_factor, gpus, output_dir)
    for row in aggregate_rows:
        print(
            f"SF={row.scale_factor}, {row.gpus} GPUs, {row.bar_key}: "
            f"{row.aggregate_time_s:.3f} s"
        )


if __name__ == "__main__":
    main()
