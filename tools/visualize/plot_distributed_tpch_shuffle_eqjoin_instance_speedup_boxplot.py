#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot speedup distributions across all shuffle-EqJoin instances.

The figure covers SF=3000 on 24 GPUs.  Every sample corresponds to one
``DistributedEqJoin(...;shuffle;...)`` physical step found across TPC-H
Q1-Q22.  Instances are matched across arms by query, step index, and exact
step label.  The instance time for an arm is the median, over five breakdown
repetitions, of that step's rank-max duration.  Speedup is Baseline time divided
by arm time.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plot_distributed_tpch_aggregate_breakdown import (
    BREAKDOWN_SAMPLES,
    QUERIES,
    REPO_ROOT,
    campaign_root,
    plt,
    read_csv,
    require_iterations,
    select_attempt,
    write_dataclass_csv,
)


SCALE_FACTOR = 3000
GPUS = 24
BASELINE_ARM = "legacy_cta64"
ARMS = (
    ("all_co_nccl_cta64", "Overlap+SM", "#54A24B"),
    ("all_co_ce_pull", "Overlap+CE", "#E45756"),
)
TARGET_OPERATOR = "DistributedEqJoin"
FIGURE_NAME = "distributed_tpch_shuffle_eqjoin_instance_speedup_boxplot"


@dataclass(frozen=True)
class InstanceTiming:
    scale_factor: int
    gpus: int
    query: int
    step_index: int
    step_label: str
    arm: str
    sample_count: int
    minimum_rank_max_ms: float
    median_rank_max_ms: float
    maximum_rank_max_ms: float
    result_status: str
    failure_kind: str
    steps_csv: str


@dataclass(frozen=True)
class InstanceSpeedup:
    scale_factor: int
    gpus: int
    query: int
    step_index: int
    step_label: str
    arm: str
    arm_label: str
    baseline_median_rank_max_ms: float
    arm_median_rank_max_ms: float
    speedup: float
    baseline_steps_csv: str
    arm_steps_csv: str


@dataclass(frozen=True)
class ArmSpeedupSummary:
    scale_factor: int
    gpus: int
    arm: str
    arm_label: str
    instance_count: int
    query_count: int
    minimum_speedup: float
    first_quartile_speedup: float
    median_speedup: float
    third_quartile_speedup: float
    maximum_speedup: float
    arithmetic_mean_speedup: float
    geometric_mean_speedup: float
    slowdown_instance_count: int


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


def extract_instance_timings(
    results_dir: Path,
    query: int,
    arm: str,
) -> list[InstanceTiming]:
    root = (
        campaign_root(results_dir, SCALE_FACTOR, GPUS, arm)
        / "breakdown"
        / f"q{query:02d}"
    )
    attempt, result = select_attempt(
        root,
        ("steps.csv",),
        SCALE_FACTOR,
        query,
    )
    steps_path = attempt / "steps.csv"
    steps = read_csv(steps_path)
    require_iterations(steps, BREAKDOWN_SAMPLES, steps_path)

    samples: dict[
        tuple[int, str],
        dict[tuple[str, int], list[tuple[float, int]]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in steps:
        if (
            row["record_type"] != "rank"
            or row["step_kind"] != TARGET_OPERATOR
            or ";shuffle" not in row["step_label"]
        ):
            continue
        duration = float(row["gpu_ms"])
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"{steps_path}: invalid gpu_ms {duration}")
        instance_key = (int(row["step_index"]), row["step_label"])
        repetition_key = (row["timestamp_utc"], int(row["iteration"]))
        samples[instance_key][repetition_key].append(
            (duration, int(row["rank"]))
        )

    rows = []
    for (step_index, step_label), repetitions in sorted(samples.items()):
        if len(repetitions) != BREAKDOWN_SAMPLES:
            raise ValueError(
                f"{steps_path}: Q{query} step {step_index} has "
                f"{len(repetitions)} repetitions, expected {BREAKDOWN_SAMPLES}"
            )
        rank_max_values = [
            max(values, key=lambda item: (item[0], -item[1]))[0]
            for values in repetitions.values()
        ]
        rows.append(
            InstanceTiming(
                scale_factor=SCALE_FACTOR,
                gpus=GPUS,
                query=query,
                step_index=step_index,
                step_label=step_label,
                arm=arm,
                sample_count=len(rank_max_values),
                minimum_rank_max_ms=min(rank_max_values),
                median_rank_max_ms=float(np.median(rank_max_values)),
                maximum_rank_max_ms=max(rank_max_values),
                result_status=str(result["status"]),
                failure_kind=str(result.get("failure_kind") or ""),
                steps_csv=str(steps_path.relative_to(results_dir)),
            )
        )
    return rows


def collect_speedups(
    results_dir: Path,
) -> tuple[list[InstanceTiming], list[InstanceSpeedup]]:
    timing_rows = [
        row
        for arm in (BASELINE_ARM,) + tuple(arm for arm, _, _ in ARMS)
        for query in QUERIES
        for row in extract_instance_timings(results_dir, query, arm)
    ]
    timing_lookup = {
        (row.arm, row.query, row.step_index, row.step_label): row
        for row in timing_rows
    }
    if len(timing_lookup) != len(timing_rows):
        raise ValueError("duplicate arm/query/step instance timing")

    baseline_keys = {
        (row.query, row.step_index, row.step_label)
        for row in timing_rows
        if row.arm == BASELINE_ARM
    }
    if len(baseline_keys) != 25:
        raise ValueError(
            f"expected 25 Baseline shuffle-join instances, "
            f"found {len(baseline_keys)}"
        )
    speedup_rows = []
    for arm, arm_label, _ in ARMS:
        arm_keys = {
            (row.query, row.step_index, row.step_label)
            for row in timing_rows
            if row.arm == arm
        }
        if arm_keys != baseline_keys:
            raise ValueError(
                f"{arm}: instance set differs from Baseline: "
                f"{sorted(arm_keys ^ baseline_keys)}"
            )
        for query, step_index, step_label in sorted(baseline_keys):
            baseline = timing_lookup[
                (BASELINE_ARM, query, step_index, step_label)
            ]
            measured = timing_lookup[(arm, query, step_index, step_label)]
            speedup = baseline.median_rank_max_ms / measured.median_rank_max_ms
            if not math.isfinite(speedup) or speedup <= 0.0:
                raise ValueError(
                    f"Q{query} step {step_index}, {arm}: invalid speedup {speedup}"
                )
            speedup_rows.append(
                InstanceSpeedup(
                    scale_factor=SCALE_FACTOR,
                    gpus=GPUS,
                    query=query,
                    step_index=step_index,
                    step_label=step_label,
                    arm=arm,
                    arm_label=arm_label,
                    baseline_median_rank_max_ms=(
                        baseline.median_rank_max_ms
                    ),
                    arm_median_rank_max_ms=measured.median_rank_max_ms,
                    speedup=speedup,
                    baseline_steps_csv=baseline.steps_csv,
                    arm_steps_csv=measured.steps_csv,
                )
            )
    return timing_rows, speedup_rows


def summarize(rows: list[InstanceSpeedup]) -> list[ArmSpeedupSummary]:
    summaries = []
    for arm, arm_label, _ in ARMS:
        selected = [row for row in rows if row.arm == arm]
        values = np.asarray([row.speedup for row in selected], dtype=float)
        if len(values) != 25:
            raise ValueError(f"{arm}: expected 25 speedups, found {len(values)}")
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        summaries.append(
            ArmSpeedupSummary(
                scale_factor=SCALE_FACTOR,
                gpus=GPUS,
                arm=arm,
                arm_label=arm_label,
                instance_count=len(values),
                query_count=len({row.query for row in selected}),
                minimum_speedup=float(np.min(values)),
                first_quartile_speedup=float(q1),
                median_speedup=float(median),
                third_quartile_speedup=float(q3),
                maximum_speedup=float(np.max(values)),
                arithmetic_mean_speedup=float(np.mean(values)),
                geometric_mean_speedup=float(np.exp(np.mean(np.log(values)))),
                slowdown_instance_count=int(np.count_nonzero(values < 1.0)),
            )
        )
    return summaries


def plot(
    rows: list[InstanceSpeedup],
    summaries: list[ArmSpeedupSummary],
    output_dir: Path,
) -> None:
    positions = np.arange(len(ARMS), dtype=float)
    data = [
        np.asarray([row.speedup for row in rows if row.arm == arm], dtype=float)
        for arm, _, _ in ARMS
    ]
    summary_lookup = {row.arm: row for row in summaries}

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    boxplot = ax.boxplot(
        data,
        orientation="vertical",
        positions=positions,
        widths=0.52,
        whis=1.5,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
        manage_ticks=False,
        medianprops={"color": "#303030", "linewidth": 2.0},
        whiskerprops={"color": "#555555", "linewidth": 1.3},
        capprops={"color": "#555555", "linewidth": 1.3},
    )
    for box, (_, _, color) in zip(boxplot["boxes"], ARMS):
        box.set_facecolor(color)
        box.set_edgecolor("#555555")
        box.set_linewidth(1.0)
        box.set_alpha(0.42)

    random = np.random.default_rng(20260823)
    for position, values, (_, _, color) in zip(positions, data, ARMS):
        jitter = random.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            s=37,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            alpha=0.92,
            zorder=4,
        )

    all_values = np.concatenate(data)
    span = max(float(np.max(all_values)) - float(np.min(all_values)), 0.2)
    lower = min(float(np.min(all_values)) - 0.10 * span, 0.94)
    upper = float(np.max(all_values)) + 0.15 * span
    ax.set_ylim(lower, upper)
    ax.axhline(
        1.0,
        color="#777777",
        linewidth=1.25,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    ax.text(
        0.015,
        1.0,
        "No speedup",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#666666",
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )
    for position, (arm, _, _) in zip(positions, ARMS):
        median = summary_lookup[arm].median_speedup
        ax.annotate(
            f"Median: {median:.2f}×",
            xy=(position, median),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.0,
            color="#303030",
            bbox={
                "boxstyle": "round,pad=0.17",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
            },
            zorder=6,
        )

    ax.set_xticks(positions, [label for _, label, _ in ARMS])
    ax.set_ylabel("Shuffle-join instance speedup over Baseline (×)")
    ax.set_title(
        f"TPC-H Shuffle-Join Instance Speedup — "
        f"SF={SCALE_FACTOR}, {GPUS} GPUs",
        pad=18,
    )
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
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.87)

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
    timing_rows, speedup_rows = collect_speedups(results_dir)
    summaries = summarize(speedup_rows)
    write_dataclass_csv(
        output_dir / f"{FIGURE_NAME}_timings.csv",
        timing_rows,
    )
    write_dataclass_csv(
        output_dir / f"{FIGURE_NAME}_samples.csv",
        speedup_rows,
    )
    write_dataclass_csv(
        output_dir / f"{FIGURE_NAME}_summary.csv",
        summaries,
    )
    plot(speedup_rows, summaries, output_dir)
    for row in summaries:
        print(
            f"{row.arm_label}: n={row.instance_count}, "
            f"median={row.median_speedup:.3f}x, "
            f"range={row.minimum_speedup:.3f}-{row.maximum_speedup:.3f}x"
        )


if __name__ == "__main__":
    main()
