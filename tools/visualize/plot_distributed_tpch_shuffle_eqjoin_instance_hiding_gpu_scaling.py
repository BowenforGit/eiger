#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot communication hiding for individual shuffle-EqJoin occurrences.

For every matched ``DistributedEqJoin(...;shuffle;...)`` physical-plan step:

    hiding (%) = 100 * (Baseline step time - Overlap+CE step time)
                       / Baseline step communication time

Step time is the median rank-max duration over five breakdown repetitions.
Baseline communication is the median direct communication interval-union on
the rank-max rank of that step in each repetition.  Values are not clipped.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.lines as mlines
import numpy as np

import plot_distributed_tpch_gpu_scaling_aggregate_breakdown as scaling
from plot_distributed_tpch_aggregate_breakdown import (
    BREAKDOWN_SAMPLES,
    QUERIES,
    REPO_ROOT,
    campaign_root,
    plt,
    read_csv,
    require_iterations,
    select_attempt,
    union_length,
    write_dataclass_csv,
)


CONFIGURATIONS = scaling.CONFIGURATIONS
BASELINE_ARM = "legacy_cta64"
OVERLAP_ARM = "all_co_ce_pull"
TARGET_OPERATOR = "DistributedEqJoin"
FIGURE_NAME = (
    "distributed_tpch_shuffle_eqjoin_instance_hiding_gpu_scaling_boxplot"
)
SETTING_COLORS = {
    (1000, 8): "#3F6FA3",
    (1000, 16): "#5A83B2",
    (1000, 24): "#7698C1",
    (1000, 32): "#91ACCF",
    (3000, 24): "#3F8F44",
    (3000, 32): "#69A964",
}


@dataclass(frozen=True)
class InstanceMeasurement:
    scale_factor: int
    gpus: int
    query: int
    step_index: int
    step_label: str
    arm: str
    sample_count: int
    median_rank_max_step_ms: float
    median_rank_max_communication_ms: float | None
    result_status: str
    failure_kind: str
    steps_csv: str
    phases_csv: str


@dataclass(frozen=True)
class InstanceHiding:
    scale_factor: int
    gpus: int
    nodes: int
    query: int
    step_index: int
    step_label: str
    baseline_step_ms: float
    overlap_step_ms: float
    baseline_communication_ms: float
    hidden_communication_ms: float
    overlapped_communication_percent: float
    baseline_steps_csv: str
    baseline_phases_csv: str
    overlap_steps_csv: str


@dataclass(frozen=True)
class ConfigurationSummary:
    scale_factor: int
    gpus: int
    nodes: int
    instance_count: int
    query_count: int
    negative_instance_count: int
    above_100_instance_count: int
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


def extract_instances(
    results_dir: Path,
    scale_factor: int,
    gpus: int,
    query: int,
    arm: str,
) -> list[InstanceMeasurement]:
    root = (
        campaign_root(results_dir, scale_factor, gpus, arm)
        / "breakdown"
        / f"q{query:02d}"
    )
    required = ("steps.csv", "phases.csv") if arm == BASELINE_ARM else ("steps.csv",)
    attempt, result = select_attempt(root, required, scale_factor, query)
    steps_path = attempt / "steps.csv"
    steps = read_csv(steps_path)
    require_iterations(steps, BREAKDOWN_SAMPLES, steps_path)

    step_samples: dict[
        tuple[int, str], dict[tuple[str, int], list[tuple[float, int]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in steps:
        if (
            row["record_type"] != "rank"
            or row["step_kind"] != TARGET_OPERATOR
            or ";shuffle" not in row["step_label"]
        ):
            continue
        if (
            int(row["scale"]) != scale_factor
            or int(row["ranks"]) != gpus
        ):
            raise ValueError(f"{steps_path}: scale/rank mismatch")
        duration = float(row["gpu_ms"])
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"{steps_path}: invalid gpu_ms {duration}")
        instance = (int(row["step_index"]), row["step_label"])
        repetition = (row["timestamp_utc"], int(row["iteration"]))
        step_samples[instance][repetition].append(
            (duration, int(row["rank"]))
        )

    phases_path = attempt / "phases.csv"
    phases = []
    if arm == BASELINE_ARM:
        phases = read_csv(phases_path)
        require_iterations(phases, BREAKDOWN_SAMPLES, phases_path)

    measurements = []
    for (step_index, step_label), repetitions in sorted(
        step_samples.items()
    ):
        if len(repetitions) != BREAKDOWN_SAMPLES:
            raise ValueError(
                f"{steps_path}: Q{query} step {step_index} has "
                f"{len(repetitions)} repetitions"
            )
        rank_max: dict[tuple[str, int], tuple[float, int]] = {}
        for repetition, values in repetitions.items():
            rank_max[repetition] = max(
                values, key=lambda item: (item[0], -item[1])
            )
        step_values = [duration for duration, _ in rank_max.values()]

        communication_median: float | None = None
        if arm == BASELINE_ARM:
            communication_values = []
            for (timestamp, iteration), (duration, rank) in rank_max.items():
                intervals = []
                for row in phases:
                    if (
                        row["record_type"] != "rank"
                        or row["quality"] != "direct"
                        or row["category"] != "communication"
                        or row["timestamp_utc"] != timestamp
                        or int(row["iteration"]) != iteration
                        or int(row["rank"]) != rank
                        or int(row["step_index"]) != step_index
                        or row["step_label"] != step_label
                    ):
                        continue
                    start = max(0.0, float(row["start_ms"]))
                    end = min(duration, float(row["end_ms"]))
                    if start < end:
                        intervals.append((start, end))
                communication = union_length(intervals)
                if communication <= 0.0:
                    raise ValueError(
                        f"{phases_path}: Q{query} step {step_index} has "
                        "no direct communication on its rank-max rank"
                    )
                communication_values.append(communication)
            communication_median = float(np.median(communication_values))

        measurements.append(
            InstanceMeasurement(
                scale_factor=scale_factor,
                gpus=gpus,
                query=query,
                step_index=step_index,
                step_label=step_label,
                arm=arm,
                sample_count=len(step_values),
                median_rank_max_step_ms=float(np.median(step_values)),
                median_rank_max_communication_ms=communication_median,
                result_status=str(result["status"]),
                failure_kind=str(result.get("failure_kind") or ""),
                steps_csv=str(steps_path.relative_to(results_dir)),
                phases_csv=(
                    str(phases_path.relative_to(results_dir))
                    if arm == BASELINE_ARM
                    else ""
                ),
            )
        )
    return measurements


def collect(
    results_dir: Path,
) -> tuple[list[InstanceMeasurement], list[InstanceHiding]]:
    measurements = [
        row
        for scale_factor, gpus in CONFIGURATIONS
        for arm in (BASELINE_ARM, OVERLAP_ARM)
        for query in QUERIES
        for row in extract_instances(
            results_dir, scale_factor, gpus, query, arm
        )
    ]
    lookup = {
        (
            row.scale_factor,
            row.gpus,
            row.arm,
            row.query,
            row.step_index,
            row.step_label,
        ): row
        for row in measurements
    }
    if len(lookup) != len(measurements):
        raise ValueError("duplicate shuffle-EqJoin instance measurement")

    hiding_rows = []
    configuration_instance_sets = []
    for scale_factor, gpus in CONFIGURATIONS:
        baseline_keys = {
            (row.query, row.step_index, row.step_label)
            for row in measurements
            if row.scale_factor == scale_factor
            and row.gpus == gpus
            and row.arm == BASELINE_ARM
        }
        overlap_keys = {
            (row.query, row.step_index, row.step_label)
            for row in measurements
            if row.scale_factor == scale_factor
            and row.gpus == gpus
            and row.arm == OVERLAP_ARM
        }
        if not baseline_keys or overlap_keys != baseline_keys:
            raise ValueError(
                f"SF={scale_factor}/{gpus}: arm instance-set mismatch: "
                f"{sorted(baseline_keys ^ overlap_keys)}"
            )
        configuration_instance_sets.append(baseline_keys)

        for query, step_index, step_label in sorted(baseline_keys):
            baseline = lookup[
                (
                    scale_factor,
                    gpus,
                    BASELINE_ARM,
                    query,
                    step_index,
                    step_label,
                )
            ]
            overlap = lookup[
                (
                    scale_factor,
                    gpus,
                    OVERLAP_ARM,
                    query,
                    step_index,
                    step_label,
                )
            ]
            communication = baseline.median_rank_max_communication_ms
            if communication is None or communication <= 0.0:
                raise ValueError(
                    f"Q{query} step {step_index}: invalid Baseline communication"
                )
            hidden = (
                baseline.median_rank_max_step_ms
                - overlap.median_rank_max_step_ms
            )
            efficiency = 100.0 * hidden / communication
            if not math.isfinite(efficiency):
                raise ValueError(
                    f"Q{query} step {step_index}: invalid efficiency {efficiency}"
                )
            hiding_rows.append(
                InstanceHiding(
                    scale_factor=scale_factor,
                    gpus=gpus,
                    nodes=gpus // 4,
                    query=query,
                    step_index=step_index,
                    step_label=step_label,
                    baseline_step_ms=baseline.median_rank_max_step_ms,
                    overlap_step_ms=overlap.median_rank_max_step_ms,
                    baseline_communication_ms=communication,
                    hidden_communication_ms=hidden,
                    overlapped_communication_percent=efficiency,
                    baseline_steps_csv=baseline.steps_csv,
                    baseline_phases_csv=baseline.phases_csv,
                    overlap_steps_csv=overlap.steps_csv,
                )
            )

    if len({frozenset(keys) for keys in configuration_instance_sets}) != 1:
        raise ValueError("shuffle-EqJoin occurrence set differs by configuration")
    return measurements, hiding_rows


def summarize(rows: list[InstanceHiding]) -> list[ConfigurationSummary]:
    summaries = []
    for scale_factor, gpus in CONFIGURATIONS:
        selected = [
            row
            for row in rows
            if row.scale_factor == scale_factor and row.gpus == gpus
        ]
        values = np.asarray(
            [row.overlapped_communication_percent for row in selected],
            dtype=float,
        )
        if len(values) == 0:
            raise ValueError(f"SF={scale_factor}/{gpus}: no instances")
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        summaries.append(
            ConfigurationSummary(
                scale_factor=scale_factor,
                gpus=gpus,
                nodes=gpus // 4,
                instance_count=len(values),
                query_count=len({row.query for row in selected}),
                negative_instance_count=int(np.count_nonzero(values < 0.0)),
                above_100_instance_count=int(np.count_nonzero(values > 100.0)),
                minimum_percent=float(np.min(values)),
                first_quartile_percent=float(q1),
                median_percent=float(median),
                third_quartile_percent=float(q3),
                maximum_percent=float(np.max(values)),
                arithmetic_mean_percent=float(np.mean(values)),
            )
        )
    return summaries


def plot(rows: list[InstanceHiding], output_dir: Path) -> None:
    configurations = list(CONFIGURATIONS)
    positions = [6.0, 5.0, 4.0, 3.0, 1.7, 0.7]
    data = [
        np.asarray(
            [
                row.overlapped_communication_percent
                for row in rows
                if row.scale_factor == scale_factor and row.gpus == gpus
            ],
            dtype=float,
        )
        for scale_factor, gpus in configurations
    ]

    fig, (ax_left, ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 6.4),
        sharey=True,
        gridspec_kw={"width_ratios": (1.25, 5.75), "wspace": 0.045},
    )
    fig.patch.set_facecolor("white")
    ax_left.set_facecolor("white")
    ax.set_facecolor("white")
    boxplot = ax.boxplot(
        data,
        orientation="horizontal",
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
    for box, configuration in zip(
        boxplot["boxes"], configurations, strict=True
    ):
        box.set_facecolor(SETTING_COLORS[configuration])
        box.set_edgecolor("#555555")
        box.set_linewidth(1.0)
        box.set_alpha(0.50)

    random = np.random.default_rng(20260828)
    for position, values, configuration in zip(
        positions, data, configurations, strict=True
    ):
        jitter = random.uniform(-0.15, 0.15, size=len(values))
        far_mask = values < -180.0
        ax_left.scatter(
            values[far_mask],
            position + jitter[far_mask],
            s=31,
            color=SETTING_COLORS[configuration],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.92,
            zorder=4,
        )
        ax.scatter(
            values[~far_mask],
            position + jitter[~far_mask],
            s=31,
            color=SETTING_COLORS[configuration],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.92,
            zorder=4,
        )
        median = float(np.median(values))
        ax.annotate(
            f"Median: {median:.1f}%",
            xy=(median, position + 0.26),
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
    far_values = all_values[all_values < -180.0]
    main_values = all_values[all_values >= -180.0]
    if len(far_values) == 0 or len(main_values) == 0:
        raise ValueError("expected both far-left and main-range samples")
    ax_left.set_xlim(float(np.min(far_values)) - 14.0, -190.0)
    ax.set_xlim(
        min(float(np.min(main_values)) - 12.0, -8.0),
        max(float(np.max(main_values)) + 10.0, 108.0),
    )
    ax_left.set_ylim(0.25, 6.55)
    ax.set_ylim(0.25, 6.55)
    ax_left.axhline(2.35, color="#E2E2E2", linewidth=1.0, zorder=1)
    ax.axhline(2.35, color="#E2E2E2", linewidth=1.0, zorder=1)
    for value, label in (
        (0.0, "0%\nNone hidden"),
        (100.0, "100%\nFully hidden"),
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
            1.015,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.7,
            linespacing=1.0,
            color="#666666",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )

    ax_left.set_yticks(
        positions,
        [
            f"SF={scale_factor}\n{gpus} GPUs"
            for scale_factor, gpus in configurations
        ],
    )
    fig.supxlabel(
        "Baseline shuffle-join communication hidden by overlap (%)",
        fontsize=12,
        y=0.035,
    )
    fig.suptitle(
        "TPC-H Communication-Hiding Efficiency — Shuffle-Join Occurrences",
        fontsize=16,
        y=0.985,
    )
    for current in (ax_left, ax):
        current.grid(
            axis="x",
            color="#D8D8D8",
            linewidth=0.8,
            linestyle=(0, (2, 3)),
            zorder=0,
        )
        current.set_axisbelow(True)
        current.spines["top"].set_visible(False)
        current.spines["bottom"].set_color("#777777")
        current.tick_params(axis="both", colors="#333333")
    ax_left.spines["left"].set_color("#777777")
    ax_left.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)

    break_size = 0.012
    break_style = {"color": "#777777", "clip_on": False, "linewidth": 1.2}
    ax_left.plot(
        (1.0 - break_size, 1.0 + break_size),
        (-break_size, +break_size),
        transform=ax_left.transAxes,
        **break_style,
    )
    ax_left.plot(
        (1.0 - break_size, 1.0 + break_size),
        (1.0 - break_size, 1.0 + break_size),
        transform=ax_left.transAxes,
        **break_style,
    )
    ax.plot(
        (-break_size, +break_size),
        (-break_size, +break_size),
        transform=ax.transAxes,
        **break_style,
    )
    ax.plot(
        (-break_size, +break_size),
        (1.0 - break_size, 1.0 + break_size),
        transform=ax.transAxes,
        **break_style,
    )

    point_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=6,
        markerfacecolor="#666666",
        markeredgecolor="white",
        label="Shuffle join",
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
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.88)

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
    measurements, hiding_rows = collect(results_dir)
    summaries = summarize(hiding_rows)
    write_dataclass_csv(
        output_dir / f"{FIGURE_NAME}_measurements.csv", measurements
    )
    write_dataclass_csv(
        output_dir / f"{FIGURE_NAME}_by_instance.csv", hiding_rows
    )
    write_dataclass_csv(
        output_dir / f"{FIGURE_NAME}_summary.csv", summaries
    )
    plot(hiding_rows, output_dir)


if __name__ == "__main__":
    main()
