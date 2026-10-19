#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot two representative SF=3000/24-GPU shuffle-EqJoin instances.

Q8 is the largest high-impact speedup and Q5 is a heavyweight instance that
lands especially close to the ideal-overlap bound.  Each measured point is the
rank-max duration of the selected physical EqJoin step in one of five
breakdown repetitions.  The ideal reference is derived independently for each
Baseline repetition as residual + max(compute, communication).
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np

from plot_distributed_tpch_aggregate_breakdown import (
    BREAKDOWN_SAMPLES,
    REPO_ROOT,
    campaign_root,
    plt,
    read_csv,
    require_iterations,
    select_attempt,
    write_dataclass_csv,
)
from plot_distributed_tpch_eqjoin_overlap_arm_comparison import interval_union


SCALE_FACTOR = 3000
GPUS = 24
BASELINE_ARM = "legacy_cta64"
SERIES = (
    (BASELINE_ARM, "Baseline", "#4C78A8"),
    ("all_co_nccl_cta64", "Overlap+SM", "#54A24B"),
    ("all_co_ce_pull", "Overlap+CE", "#E45756"),
)
IDEAL_KEY = "ideal_overlap"
IDEAL_LABEL = "Ideal overlap"
IDEAL_COLOR = "#B8B8B8"
TARGET_OPERATOR = "DistributedEqJoin"
FIGURE_PREFIX = "distributed_tpch_shuffle_eqjoin_hero"


@dataclass(frozen=True)
class Hero:
    query: int
    step_index: int
    short_name: str


HEROES = (
    Hero(8, 13, "Orders–Lineitem"),
    Hero(5, 6, "Orders–Lineitem"),
)


@dataclass(frozen=True)
class HeroSample:
    scale_factor: int
    gpus: int
    query: int
    step_index: int
    step_label: str
    series_key: str
    series_label: str
    kind: str
    timestamp_utc: str
    iteration: int
    critical_rank: int
    time_ms: float
    compute_ms: float | None
    communication_ms: float | None
    residual_ms: float | None
    result_status: str
    failure_kind: str
    steps_csv: str
    phases_csv: str


@dataclass(frozen=True)
class HeroSummary:
    scale_factor: int
    gpus: int
    query: int
    step_index: int
    step_label: str
    series_key: str
    series_label: str
    kind: str
    sample_count: int
    minimum_ms: float
    median_ms: float
    maximum_ms: float
    speedup_over_baseline: float
    baseline_compute_median_ms: float | None
    baseline_communication_median_ms: float | None
    communication_hidden_percent: float | None
    gap_above_ideal_percent: float | None


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


def selected_rank_max_steps(
    steps: list[dict[str, str]],
    hero: Hero,
    steps_path: Path,
) -> tuple[str, list[tuple[str, int, int, float]]]:
    candidates: dict[tuple[str, int], list[tuple[float, int, str]]] = (
        defaultdict(list)
    )
    for row in steps:
        if (
            row["record_type"] != "rank"
            or row["step_kind"] != TARGET_OPERATOR
            or int(row["step_index"]) != hero.step_index
            or ";shuffle" not in row["step_label"]
        ):
            continue
        duration = float(row["gpu_ms"])
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"{steps_path}: invalid gpu_ms {duration}")
        candidates[(row["timestamp_utc"], int(row["iteration"]))].append(
            (duration, int(row["rank"]), row["step_label"])
        )
    if len(candidates) != BREAKDOWN_SAMPLES:
        raise ValueError(
            f"{steps_path}: Q{hero.query} step {hero.step_index} has "
            f"{len(candidates)} repetitions, expected {BREAKDOWN_SAMPLES}"
        )

    selected = []
    labels = set()
    for (timestamp, iteration), values in sorted(candidates.items()):
        duration, rank, label = max(values, key=lambda item: (item[0], -item[1]))
        labels.add(label)
        selected.append((timestamp, iteration, rank, duration))
    if len(labels) != 1:
        raise ValueError(f"{steps_path}: inconsistent step labels {labels}")
    return labels.pop(), selected


def collect_arm(
    results_dir: Path,
    hero: Hero,
    arm: str,
    series_label: str,
) -> tuple[list[HeroSample], str]:
    root = (
        campaign_root(results_dir, SCALE_FACTOR, GPUS, arm)
        / "breakdown"
        / f"q{hero.query:02d}"
    )
    attempt, result = select_attempt(
        root,
        ("steps.csv", "phases.csv"),
        SCALE_FACTOR,
        hero.query,
    )
    steps_path = attempt / "steps.csv"
    phases_path = attempt / "phases.csv"
    steps = read_csv(steps_path)
    require_iterations(steps, BREAKDOWN_SAMPLES, steps_path)
    step_label, selected = selected_rank_max_steps(steps, hero, steps_path)

    rows = [
        HeroSample(
            scale_factor=SCALE_FACTOR,
            gpus=GPUS,
            query=hero.query,
            step_index=hero.step_index,
            step_label=step_label,
            series_key=arm,
            series_label=series_label,
            kind="measured_rank_max_step_time",
            timestamp_utc=timestamp,
            iteration=iteration,
            critical_rank=rank,
            time_ms=duration,
            compute_ms=None,
            communication_ms=None,
            residual_ms=None,
            result_status=str(result["status"]),
            failure_kind=str(result.get("failure_kind") or ""),
            steps_csv=str(steps_path.relative_to(results_dir)),
            phases_csv=str(phases_path.relative_to(results_dir)),
        )
        for timestamp, iteration, rank, duration in selected
    ]
    return rows, step_label


def collect_ideal(
    results_dir: Path,
    hero: Hero,
    baseline_rows: list[HeroSample],
) -> list[HeroSample]:
    phases_path = results_dir / baseline_rows[0].phases_csv
    phases = read_csv(phases_path)
    require_iterations(phases, BREAKDOWN_SAMPLES, phases_path)
    output = []
    for baseline in baseline_rows:
        intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in phases:
            if (
                row["record_type"] != "rank"
                or row["quality"] != "direct"
                or row["timestamp_utc"] != baseline.timestamp_utc
                or int(row["iteration"]) != baseline.iteration
                or int(row["rank"]) != baseline.critical_rank
                or int(row["step_index"]) != hero.step_index
                or row["step_kind"] != TARGET_OPERATOR
                or row["category"] not in {"compute", "communication"}
            ):
                continue
            start = max(0.0, float(row["start_ms"]))
            end = min(baseline.time_ms, float(row["end_ms"]))
            if start < end:
                intervals[row["category"]].append((start, end))

        compute = interval_union(intervals["compute"])
        communication = interval_union(intervals["communication"])
        covered = interval_union(
            intervals["compute"] + intervals["communication"]
        )
        residual = max(0.0, baseline.time_ms - min(baseline.time_ms, covered))
        ideal = residual + max(compute, communication)
        tolerance = max(0.05, baseline.time_ms * 0.01)
        if ideal > baseline.time_ms + tolerance:
            raise ValueError(
                f"Q{hero.query} iteration {baseline.iteration}: ideal "
                f"{ideal:.6f} ms exceeds step {baseline.time_ms:.6f} ms"
            )
        output.append(
            HeroSample(
                scale_factor=SCALE_FACTOR,
                gpus=GPUS,
                query=hero.query,
                step_index=hero.step_index,
                step_label=baseline.step_label,
                series_key=IDEAL_KEY,
                series_label=IDEAL_LABEL,
                kind="derived_ideal_overlap",
                timestamp_utc=baseline.timestamp_utc,
                iteration=baseline.iteration,
                critical_rank=baseline.critical_rank,
                time_ms=ideal,
                compute_ms=compute,
                communication_ms=communication,
                residual_ms=residual,
                result_status=baseline.result_status,
                failure_kind=baseline.failure_kind,
                steps_csv=baseline.steps_csv,
                phases_csv=baseline.phases_csv,
            )
        )
    return output


def collect_hero(results_dir: Path, hero: Hero) -> list[HeroSample]:
    rows = []
    labels = set()
    baseline_rows = []
    for arm, series_label, _ in SERIES:
        arm_rows, step_label = collect_arm(
            results_dir,
            hero,
            arm,
            series_label,
        )
        rows.extend(arm_rows)
        labels.add(step_label)
        if arm == BASELINE_ARM:
            baseline_rows = arm_rows
    if len(labels) != 1:
        raise ValueError(f"Q{hero.query}: arms have different labels {labels}")
    rows.extend(collect_ideal(results_dir, hero, baseline_rows))
    return rows


def summarize(rows: list[HeroSample]) -> list[HeroSummary]:
    summaries = []
    for hero in HEROES:
        selected = [row for row in rows if row.query == hero.query]
        lookup = {
            key: [row for row in selected if row.series_key == key]
            for key in tuple(arm for arm, _, _ in SERIES) + (IDEAL_KEY,)
        }
        if any(len(values) != BREAKDOWN_SAMPLES for values in lookup.values()):
            raise ValueError(f"Q{hero.query}: incomplete series")
        medians = {
            key: float(np.median([row.time_ms for row in values]))
            for key, values in lookup.items()
        }
        baseline_median = medians[BASELINE_ARM]
        ideal_rows = lookup[IDEAL_KEY]
        compute_median = float(
            np.median([row.compute_ms for row in ideal_rows])
        )
        communication_median = float(
            np.median([row.communication_ms for row in ideal_rows])
        )
        ce_median = medians["all_co_ce_pull"]
        hidden_percent = (
            100.0
            * (baseline_median - ce_median)
            / communication_median
        )
        gap_percent = 100.0 * (ce_median / medians[IDEAL_KEY] - 1.0)
        for key, values in lookup.items():
            times = np.asarray([row.time_ms for row in values], dtype=float)
            label = values[0].series_label
            summaries.append(
                HeroSummary(
                    scale_factor=SCALE_FACTOR,
                    gpus=GPUS,
                    query=hero.query,
                    step_index=hero.step_index,
                    step_label=values[0].step_label,
                    series_key=key,
                    series_label=label,
                    kind=values[0].kind,
                    sample_count=len(times),
                    minimum_ms=float(np.min(times)),
                    median_ms=float(np.median(times)),
                    maximum_ms=float(np.max(times)),
                    speedup_over_baseline=baseline_median / float(np.median(times)),
                    baseline_compute_median_ms=(
                        compute_median if key == IDEAL_KEY else None
                    ),
                    baseline_communication_median_ms=(
                        communication_median if key == IDEAL_KEY else None
                    ),
                    communication_hidden_percent=(
                        hidden_percent if key == "all_co_ce_pull" else None
                    ),
                    gap_above_ideal_percent=(
                        gap_percent if key == "all_co_ce_pull" else None
                    ),
                )
            )
    return summaries


def plot_hero(
    hero: Hero,
    summaries: list[HeroSummary],
    output_dir: Path,
) -> None:
    order = tuple(arm for arm, _, _ in SERIES) + (IDEAL_KEY,)
    labels = [label for _, label, _ in SERIES] + [IDEAL_LABEL]
    colors = [color for _, _, color in SERIES] + [IDEAL_COLOR]
    selected_summaries = {
        row.series_key: row for row in summaries if row.query == hero.query
    }
    values = [selected_summaries[key].median_ms for key in order]
    maximum = max(values)
    positions = np.arange(len(order), dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.bar(
        positions,
        values,
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor("#666666")
    bars[-1].set_linewidth(1.0)

    for position, key, value in zip(positions, order, values):
        speedup = selected_summaries[key].speedup_over_baseline
        ax.text(
            position,
            value + maximum * 0.026,
            f"{value:.1f} ms\n{speedup:.2f}×",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#303030",
            linespacing=1.05,
        )

    ce_summary = selected_summaries["all_co_ce_pull"]
    ax.text(
        0.985,
        0.975,
        f"Overlap+CE hides {ce_summary.communication_hidden_percent:.0f}% "
        "of communication\n"
        f"{ce_summary.gap_above_ideal_percent:.1f}% above ideal overlap",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.94,
        },
        zorder=6,
    )

    ax.set_xticks(positions, labels)
    ax.set_ylabel("Rank-max shuffle EqJoin time (ms)")
    ax.set_title(
        f"Q{hero.query} {hero.short_name} Shuffle Join — "
        f"SF={SCALE_FACTOR}, {GPUS} GPUs",
        pad=16,
    )
    ax.set_ylim(0.0, maximum * 1.30)
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

    ideal_handle = mpatches.Patch(
        facecolor=IDEAL_COLOR,
        edgecolor="#666666",
        hatch="///",
        label="Derived ideal",
    )
    ax.legend(
        handles=(ideal_handle,),
        loc="upper left",
        frameon=False,
        fontsize=9.0,
    )
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.15, top=0.88)

    stem = f"{FIGURE_PREFIX}_q{hero.query:02d}_sf{SCALE_FACTOR}_{GPUS}gpu"
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
    rows = [
        row
        for hero in HEROES
        for row in collect_hero(results_dir, hero)
    ]
    summaries = summarize(rows)
    write_dataclass_csv(
        output_dir / f"{FIGURE_PREFIX}_samples.csv",
        rows,
    )
    write_dataclass_csv(
        output_dir / f"{FIGURE_PREFIX}_summary.csv",
        summaries,
    )
    for hero in HEROES:
        plot_hero(hero, summaries, output_dir)
    for row in summaries:
        print(
            f"Q{row.query} {row.series_label}: {row.median_ms:.3f} ms, "
            f"{row.speedup_over_baseline:.3f}x"
        )


if __name__ == "__main__":
    main()
