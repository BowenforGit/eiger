#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot paired SF100/SF300 distributed TPC-H benchmark results.

The end-to-end chart uses the median maximum-rank wall time from each clean
timing pass. The physical-step chart uses aggregate records from the separately
instrumented pass. The compute/communication chart selects the critical rank
for each iteration and then preserves the complete component vector from the
median-total iteration.
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    sys.exit(f"error: matplotlib is required: {error}")


QUERIES = tuple(range(1, 23))
SCALES = (100, 300)
SCALE_COLORS = {100: "#4E79A7", 300: "#F28E2B"}

# Distributed and local forms intentionally share a color family. This keeps
# the common legend compact while the actual step kind remains in its label.
STEP_COLORS = {
    "Filter": "#4E79A7",
    "Project": "#F28E2B",
    "Reorder": "#BAB0AC",
    "Concat": "#FFBE7D",
    "GroupBy": "#59A14F",
    "ScalarAgg": "#EDC948",
    "EqJoin": "#E15759",
    "SemiJoin": "#76B7B2",
    "AntiJoin": "#B07AA1",
    "CrossJoin": "#8CD17D",
    "Distinct": "#A0CBE8",
    "Sort": "#9C755F",
    "TopK": "#D37295",
}

STEP_KIND_ORDER = (
    "Filter",
    "Project",
    "Reorder",
    "DistributedEqJoin",
    "DistributedSemiJoin",
    "DistributedAntiJoin",
    "DistributedCrossJoin",
    "DistributedGroupBy",
    "DistributedScalarAgg",
    "DistributedDistinct",
    "DistributedSort",
    "DistributedTopK",
    "Concat",
)

PHASE_COMPONENT_ORDER = (
    "Compute",
    "Shuffle",
    "Broadcast",
    "Gather",
    "Reduce",
    "Sampling",
    "Other communication",
    "Overlap",
    "Overhead",
)
PHASE_COLORS = {
    "Compute": "#4E79A7",
    "Shuffle": "#E15759",
    "Broadcast": "#F28E2B",
    "Gather": "#59A14F",
    "Reduce": "#B07AA1",
    "Sampling": "#76B7B2",
    "Other communication": "#9C755F",
    "Overlap": "#D4A6C8",
    "Overhead": "#BAB0AC",
}


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    sample_count: int


@dataclass(frozen=True)
class StepSummary:
    index: int
    kind: str
    label: str
    median_ms: float


@dataclass(frozen=True)
class PhaseBreakdown:
    timestamp_utc: str
    iteration: int
    rank: int
    step_total_ms: float
    components_ms: dict[str, float]


def parse_query(value: str) -> int:
    normalized = value.strip().upper()
    if normalized.startswith("Q"):
        normalized = normalized[1:]
    try:
        query = int(normalized)
    except ValueError as error:
        raise ValueError(f"invalid query value: {value!r}") from error
    if query not in QUERIES:
        raise ValueError(f"query must be in Q1-Q22, got {value!r}")
    return query


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing CSV: {path}")
    with path.open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def require_columns(
    rows: list[dict[str, str]], required: set[str], path: Path
) -> None:
    missing = required - set(rows[0])
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing columns: {names}")


def validate_scale(rows: list[dict[str, str]], scale: int, path: Path) -> None:
    scales = {int(row["scale"]) for row in rows}
    if scales != {scale}:
        raise ValueError(f"{path} has scale values {sorted(scales)}, expected {scale}")


def summarize_e2e(path: Path, scale: int) -> tuple[dict[int, TimingSummary], dict[str, str]]:
    rows = read_csv(path)
    require_columns(
        rows,
        {"query", "scale", "rank_max_ms", "ranks", "nodes", "gpu_name"},
        path,
    )
    validate_scale(rows, scale, path)

    samples: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        samples[parse_query(row["query"])].append(float(row["rank_max_ms"]))
    missing = set(QUERIES) - set(samples)
    if missing:
        raise ValueError(f"{path} is missing queries: {sorted(missing)}")

    summaries = {
        query: TimingSummary(
            median_ms=statistics.median(values),
            minimum_ms=min(values),
            maximum_ms=max(values),
            sample_count=len(values),
        )
        for query, values in samples.items()
    }
    metadata = {}
    for column in ("ranks", "nodes", "gpu_name"):
        values = {row[column] for row in rows}
        if len(values) != 1:
            raise ValueError(f"{path} has inconsistent {column}: {sorted(values)}")
        metadata[column] = values.pop()
    return summaries, metadata


def summarize_steps(path: Path, scale: int) -> dict[int, list[StepSummary]]:
    rows = read_csv(path)
    require_columns(
        rows,
        {
            "query",
            "scale",
            "record_type",
            "step_index",
            "step_kind",
            "step_label",
            "rank_max_ms",
        },
        path,
    )
    validate_scale(rows, scale, path)
    aggregate_rows = [row for row in rows if row["record_type"] == "aggregate"]
    if not aggregate_rows:
        raise ValueError(f"{path} has no aggregate step records")

    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in aggregate_rows:
        grouped[(parse_query(row["query"]), int(row["step_index"]))].append(row)

    by_query: dict[int, list[StepSummary]] = defaultdict(list)
    for (query, step_index), step_rows in grouped.items():
        kinds = {row["step_kind"] for row in step_rows}
        labels = {row["step_label"] for row in step_rows}
        if len(kinds) != 1 or len(labels) != 1:
            raise ValueError(
                f"{path}: Q{query} step {step_index} changed identity across samples"
            )
        values = [float(row["rank_max_ms"]) for row in step_rows]
        by_query[query].append(
            StepSummary(
                index=step_index,
                kind=kinds.pop(),
                label=labels.pop(),
                median_ms=statistics.median(values),
            )
        )

    missing = set(QUERIES) - set(by_query)
    if missing:
        raise ValueError(f"{path} is missing step data for queries: {sorted(missing)}")
    for steps in by_query.values():
        steps.sort(key=lambda step: step.index)
    return dict(by_query)


def phase_component(row: dict[str, str]) -> str:
    category = row["category"]
    if category == "compute":
        return "Compute"
    if category == "overhead":
        return "Overhead"
    if category != "communication":
        raise ValueError(f"unknown phase category: {category!r}")
    pattern = row["communication_pattern"]
    names = {
        "shuffle": "Shuffle",
        "broadcast": "Broadcast",
        "gather": "Gather",
        "reduce": "Reduce",
        "sampling": "Sampling",
        "other": "Other communication",
    }
    return names.get(pattern, "Other communication")


def attribute_phase_intervals(
    rows: list[dict[str, str]], step_ms: float
) -> dict[str, float]:
    """Partition one step's elapsed union into exclusive work and overlap."""
    intervals = []
    for row in rows:
        start = max(0.0, float(row["start_ms"]))
        end = min(step_ms, float(row["end_ms"]))
        if start < end:
            intervals.append((start, end, phase_component(row)))
    points = sorted({point for start, end, _ in intervals for point in (start, end)})
    attributed: dict[str, float] = defaultdict(float)
    for start, end in zip(points, points[1:]):
        active = {
            component
            for interval_start, interval_end, component in intervals
            if interval_start < end and interval_end > start
        }
        if not active:
            continue
        component = next(iter(active)) if len(active) == 1 else "Overlap"
        attributed[component] += end - start
    return dict(attributed)


def select_critical_median_samples(
    step_totals: dict[tuple[int, str, int, int], float]
) -> dict[int, tuple[str, int, int, float]]:
    critical_by_iteration: dict[tuple[int, str, int], tuple[int, float]] = {}
    for (query, timestamp_utc, iteration, rank), total in step_totals.items():
        key = (query, timestamp_utc, iteration)
        current = critical_by_iteration.get(key)
        if current is None or total > current[1] or (
            total == current[1] and rank < current[0]
        ):
            critical_by_iteration[key] = (rank, total)

    iterations_by_query: dict[int, list[tuple[str, int, int, float]]] = defaultdict(list)
    for (query, timestamp_utc, iteration), (rank, total) in (
        critical_by_iteration.items()
    ):
        iterations_by_query[query].append(
            (timestamp_utc, iteration, rank, total)
        )

    selected = {}
    for query, samples in iterations_by_query.items():
        samples.sort(key=lambda sample: (sample[3], sample[0], sample[1]))
        selected[query] = samples[(len(samples) - 1) // 2]
    return selected


def summarize_phases(
    steps_path: Path, phases_path: Path, scale: int
) -> dict[int, PhaseBreakdown]:
    step_rows = read_csv(steps_path)
    phase_rows = read_csv(phases_path)
    require_columns(
        step_rows,
        {
            "timestamp_utc",
            "query",
            "scale",
            "record_type",
            "rank",
            "iteration",
            "gpu_ms",
        },
        steps_path,
    )
    require_columns(
        phase_rows,
        {
            "query",
            "scale",
            "timestamp_utc",
            "record_type",
            "rank",
            "iteration",
            "category",
            "communication_pattern",
            "gpu_ms",
        },
        phases_path,
    )
    validate_scale(step_rows, scale, steps_path)
    validate_scale(phase_rows, scale, phases_path)
    interval_schema = bool(phase_rows) and {
        "step_index", "start_ms", "end_ms", "quality"
    }.issubset(phase_rows[0])

    step_totals: dict[tuple[int, str, int, int], float] = defaultdict(float)
    for row in step_rows:
        if row["record_type"] != "rank":
            continue
        key = (
            parse_query(row["query"]),
            row["timestamp_utc"],
            int(row["iteration"]),
            int(row["rank"]),
        )
        step_totals[key] += float(row["gpu_ms"])

    selected = select_critical_median_samples(step_totals)
    missing = set(QUERIES) - set(selected)
    if missing:
        raise ValueError(
            f"{steps_path} is missing raw rank data for queries: {sorted(missing)}"
        )
    components: dict[int, dict[str, float]] = {
        query: defaultdict(float) for query in QUERIES
    }
    selected_phase_rows: dict[int, list[dict[str, str]]] = {
        query: [] for query in QUERIES
    }
    for row in phase_rows:
        if row["record_type"] != "rank":
            continue
        query = parse_query(row["query"])
        timestamp_utc, iteration, rank, _ = selected[query]
        if (
            row["timestamp_utc"] != timestamp_utc
            or int(row["iteration"]) != iteration
            or int(row["rank"]) != rank
        ):
            continue
        selected_phase_rows[query].append(row)

    if interval_schema:
        step_durations = {}
        for row in step_rows:
            if row["record_type"] != "rank":
                continue
            query = parse_query(row["query"])
            timestamp_utc, iteration, rank, _ = selected[query]
            if (
                row["timestamp_utc"] == timestamp_utc
                and int(row["iteration"]) == iteration
                and int(row["rank"]) == rank
            ):
                step_durations[query, int(row["step_index"])] = float(row["gpu_ms"])

        for query, rows in selected_phase_rows.items():
            direct_by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                if row["category"] == "overhead":
                    components[query]["Overhead"] += float(row["gpu_ms"])
                elif row["quality"] == "direct":
                    direct_by_step[int(row["step_index"])].append(row)
            for step_index, direct_rows in direct_by_step.items():
                duration = step_durations[query, step_index]
                for component, value in attribute_phase_intervals(
                    direct_rows, duration
                ).items():
                    components[query][component] += value
    else:
        # Compatibility with historical single-stream phase CSVs, whose
        # exclusive direct durations were additive and lacked intervals.
        for query, rows in selected_phase_rows.items():
            for row in rows:
                components[query][phase_component(row)] += float(row["gpu_ms"])

    result = {}
    for query in QUERIES:
        timestamp_utc, iteration, rank, total = selected[query]
        values = {name: components[query].get(name, 0.0) for name in PHASE_COMPONENT_ORDER}
        phase_total = sum(values.values())
        tolerance = max(0.05, total * 0.01)
        if abs(phase_total - total) > tolerance:
            raise ValueError(
                f"{phases_path}: Q{query} selected phase total {phase_total:.6f} "
                f"does not match outer-step total {total:.6f}"
            )
        result[query] = PhaseBreakdown(
            timestamp_utc=timestamp_utc,
            iteration=iteration,
            rank=rank,
            step_total_ms=total,
            components_ms=values,
        )
    return result


def base_kind(kind: str) -> str:
    return kind.removeprefix("Distributed")


def step_color(kind: str) -> str:
    return STEP_COLORS.get(base_kind(kind), "#6B6B6B")


def display_kind(kind: str) -> str:
    if kind.startswith("Distributed"):
        return "Dist. " + kind.removeprefix("Distributed")
    return kind


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(colors="#333333")
    axis.set_axisbelow(True)


def save_figure(figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output_base.with_suffix(suffix)
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(figure)


def e2e_caption(summaries: dict[int, dict[int, TimingSummary]]) -> str:
    counts = [
        summary.sample_count
        for scale_summaries in summaries.values()
        for summary in scale_summaries.values()
    ]
    minimum = min(counts)
    maximum = max(counts)
    if minimum == maximum:
        sample_text = f"across {minimum} runs per query"
    else:
        sample_text = f"with {minimum}-{maximum} runs per query"
    return (
        f"Bars show the median maximum-rank wall time {sample_text}; "
        "whiskers show min-max."
    )


def plot_e2e(
    summaries: dict[int, dict[int, TimingSummary]],
    metadata: dict[str, str],
    output_base: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(18, 6.2))
    positions = list(range(len(QUERIES)))
    width = 0.38

    for scale, offset in ((100, -width / 2), (300, width / 2)):
        medians = [summaries[scale][query].median_ms for query in QUERIES]
        lower = [
            summaries[scale][query].median_ms - summaries[scale][query].minimum_ms
            for query in QUERIES
        ]
        upper = [
            summaries[scale][query].maximum_ms - summaries[scale][query].median_ms
            for query in QUERIES
        ]
        bars = axis.bar(
            [position + offset for position in positions],
            medians,
            width=width,
            color=SCALE_COLORS[scale],
            edgecolor="white",
            linewidth=0.5,
            label=f"SF={scale}",
            yerr=[lower, upper],
            error_kw={"ecolor": "#333333", "elinewidth": 0.7, "capsize": 2},
            zorder=3,
        )
        for bar, value in zip(bars, medians):
            axis.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
                color="#222222",
            )

    axis.set_xticks(positions, [f"Q{query}" for query in QUERIES])
    axis.set_ylabel("Median end-to-end time (ms)", fontsize=12)
    axis.set_xlabel("TPC-H query", fontsize=12)
    axis.set_title(
        "Distributed TPC-H query time after data load\n"
        f"{metadata['ranks']} GPUs on {metadata['nodes']} node "
        f"({metadata['gpu_name']})",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    axis.grid(axis="y", linestyle=":", linewidth=0.7, color="#B8B8B8")
    axis.legend(frameon=False, ncol=2, fontsize=11, loc="upper left")
    axis.margins(x=0.01)
    axis.set_ylim(bottom=0)
    style_axis(axis)
    figure.text(
        0.5,
        0.005,
        e2e_caption(summaries),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(figure, output_base)


def plot_breakdown(
    summaries: dict[int, dict[int, list[StepSummary]]],
    metadata: dict[str, str],
    output_base: Path,
) -> None:
    figure, axes = plt.subplots(2, 11, figsize=(30, 10.5), sharey=False)
    used_kinds: set[str] = set()

    for plot_index, query in enumerate(QUERIES):
        axis = axes.flat[plot_index]
        totals = []
        for x_position, scale in enumerate(SCALES):
            bottom = 0.0
            for step in summaries[scale][query]:
                used_kinds.add(step.kind)
                axis.bar(
                    x_position,
                    step.median_ms,
                    bottom=bottom,
                    width=0.62,
                    color=step_color(step.kind),
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                )
                bottom += step.median_ms
            totals.append(bottom)

        axis.set_xticks((0, 1), ("100", "300"), fontsize=9)
        axis.set_title(f"Q{query}", fontsize=12, fontweight="bold", pad=5)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(axis="y", linestyle=":", linewidth=0.55, color="#C0C0C0")
        axis.set_ylim(0, max(totals) * 1.18 if max(totals) > 0 else 1.0)
        style_axis(axis)
        if plot_index % 11 == 0:
            axis.set_ylabel("Step time (ms)", fontsize=10)
        for x_position, total in enumerate(totals):
            axis.annotate(
                f"{total:.1f}",
                (x_position, total),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#222222",
            )

    legend_order = [kind for kind in STEP_KIND_ORDER if kind in used_kinds]
    legend_order.extend(sorted(used_kinds - set(legend_order)))
    handles = [
        mpatches.Patch(color=step_color(kind), label=display_kind(kind))
        for kind in legend_order
    ]
    figure.suptitle(
        "Distributed TPC-H physical-step breakdown\n"
        f"{metadata['ranks']} GPUs on {metadata['nodes']} node "
        f"({metadata['gpu_name']})",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.035),
    )
    figure.text(
        0.5,
        0.012,
        "Each segment is one physical step in execution order. Values are median maximum-rank "
        "GPU times from the separate instrumented step pass.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.015, 0.095, 0.995, 0.945), w_pad=1.0, h_pad=1.8)
    save_figure(figure, output_base)


def plot_phase_breakdown(
    summaries: dict[int, dict[int, PhaseBreakdown]],
    metadata: dict[str, str],
    output_base: Path,
) -> None:
    figure, axes = plt.subplots(2, 11, figsize=(30, 10.5), sharey=False)
    used_components: set[str] = set()

    for plot_index, query in enumerate(QUERIES):
        axis = axes.flat[plot_index]
        totals = []
        for x_position, scale in enumerate(SCALES):
            sample = summaries[scale][query]
            bottom = 0.0
            for component in PHASE_COMPONENT_ORDER:
                value = sample.components_ms[component]
                if value <= 0:
                    continue
                used_components.add(component)
                axis.bar(
                    x_position,
                    value,
                    bottom=bottom,
                    width=0.62,
                    color=PHASE_COLORS[component],
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                )
                bottom += value
            totals.append(bottom)

        axis.set_xticks((0, 1), ("100", "300"), fontsize=9)
        axis.set_title(f"Q{query}", fontsize=12, fontweight="bold", pad=5)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(axis="y", linestyle=":", linewidth=0.55, color="#C0C0C0")
        axis.set_ylim(0, max(totals) * 1.18 if max(totals) > 0 else 1.0)
        style_axis(axis)
        if plot_index % 11 == 0:
            axis.set_ylabel("Phase time (ms)", fontsize=10)
        for x_position, total in enumerate(totals):
            axis.annotate(
                f"{total:.1f}",
                (x_position, total),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#222222",
            )

    handles = [
        mpatches.Patch(color=PHASE_COLORS[name], label=name)
        for name in PHASE_COMPONENT_ORDER
        if name in used_components
    ]
    figure.suptitle(
        "Distributed TPC-H compute/communication breakdown\n"
        f"{metadata['ranks']} GPUs on {metadata['nodes']} node "
        f"({metadata['gpu_name']})",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.035),
    )
    figure.text(
        0.5,
        0.012,
        "For each iteration, phases come from the rank with the largest summed step time. "
        "The chart preserves the complete phase vector from the median-total iteration; "
        "Overlap is elapsed time with multiple active phase categories.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.015, 0.095, 0.995, 0.945), w_pad=1.0, h_pad=1.8)
    save_figure(figure, output_base)


def validate_paired_metadata(metadata: dict[int, dict[str, str]]) -> dict[str, str]:
    reference = metadata[SCALES[0]]
    for scale in SCALES[1:]:
        if metadata[scale] != reference:
            raise ValueError(
                f"benchmark metadata differs between SF100 and SF{scale}: "
                f"{reference} vs {metadata[scale]}"
            )
    return reference


def warn_on_plan_differences(
    summaries: dict[int, dict[int, list[StepSummary]]]
) -> None:
    for query in QUERIES:
        sf100_identity = [
            (step.index, step.kind, step.label) for step in summaries[100][query]
        ]
        sf300_identity = [
            (step.index, step.kind, step.label) for step in summaries[300][query]
        ]
        if sf100_identity != sf300_identity:
            print(
                f"warning: Q{query} physical steps differ between SF100 and SF300",
                file=sys.stderr,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paired SF100/SF300 distributed TPC-H benchmark CSVs."
    )
    parser.add_argument(
        "--sf100-dir",
        type=Path,
        required=True,
        help="directory containing SF100 e2e.csv and steps.csv",
    )
    parser.add_argument(
        "--sf300-dir",
        type=Path,
        required=True,
        help="directory containing SF300 e2e.csv and steps.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="output directory for PNG and PDF figures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directories = {100: args.sf100_dir, 300: args.sf300_dir}
    e2e_summaries = {}
    step_summaries = {}
    phase_summaries = {}
    metadata = {}
    have_phases = all(
        (directories[scale] / "phases.csv").is_file() for scale in SCALES
    )

    try:
        for scale in SCALES:
            e2e_summaries[scale], metadata[scale] = summarize_e2e(
                directories[scale] / "e2e.csv", scale
            )
            step_summaries[scale] = summarize_steps(
                directories[scale] / "steps.csv", scale
            )
            if have_phases:
                phase_summaries[scale] = summarize_phases(
                    directories[scale] / "steps.csv",
                    directories[scale] / "phases.csv",
                    scale,
                )
        shared_metadata = validate_paired_metadata(metadata)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    warn_on_plan_differences(step_summaries)
    plot_e2e(
        e2e_summaries,
        shared_metadata,
        args.out_dir / "tpch_sf100_sf300_e2e",
    )
    plot_breakdown(
        step_summaries,
        shared_metadata,
        args.out_dir / "tpch_sf100_sf300_step_breakdown",
    )
    if have_phases:
        plot_phase_breakdown(
            phase_summaries,
            shared_metadata,
            args.out_dir / "tpch_sf100_sf300_comp_comm_breakdown",
        )
    else:
        print(
            "warning: phases.csv is missing; skipped compute/communication plot",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
