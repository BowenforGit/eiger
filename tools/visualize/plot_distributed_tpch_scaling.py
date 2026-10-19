#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Create paper-ready aggregate scaling figures from TPC-H campaigns.

For every query, the clean end-to-end measurement is the median of the ten
maximum-rank iteration times in ``e2e.csv``.  A suite aggregate is the sum of
those 22 independent query medians.  It is not an official TPC-H score or a
directly measured sequential suite runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ModuleNotFoundError as error:
    sys.exit(f"error: matplotlib is required: {error}")


GPU_COUNTS = (8, 16, 24, 32)
QUERIES = tuple(range(1, 23))
EXPECTED_SAMPLE_COUNT = 10
EXPECTED_BREAKDOWN_SAMPLE_COUNT = 5
EXPECTED_SCALE = 1000
EXPECTED_RESERVE_BYTES = 40_000_000_000

BREAKDOWN_ARMS = (
    "legacy",
    "synthesized",
    "legacy_aligned_both",
    "legacy_ce_pull",
)
PHASE_ORDER = ("Compute", "Shuffle", "Other communication", "Overhead")
PHASE_COLORS = {
    "Compute": "#4E79A7",
    "Shuffle": "#F28E2B",
    "Other communication": "#76B7B2",
    "Overhead": "#BAB0AC",
}
PHASE_FIELDS = {
    "Compute": "compute_ms",
    "Shuffle": "shuffle_ms",
    "Other communication": "other_communication_ms",
    "Overhead": "overhead_ms",
}

# Okabe-Ito-derived, color-vision-deficiency-friendly colors. The baseline
# remains neutral in every figure so the experimental arm carries the emphasis.
ARM_STYLES = {
    "legacy": {"label": "Baseline (NCCL)", "color": "#7A8793"},
    "synthesized": {
        "label": "Optimized local operators",
        "color": "#0072B2",
    },
    "legacy_aligned_both": {
        "label": "Optimized alignment",
        "color": "#009E73",
    },
    "legacy_ce_pull": {
        "label": "Legacy + CE-pull",
        "color": "#D55E00",
    },
    "all_co_nccl": {
        "label": "ColumnOverlap + NCCL (auto)",
        "color": "#0072B2",
    },
    "all_co_ce_pull": {
        "label": "ColumnOverlap + CE-pull",
        "color": "#D55E00",
    },
    "all_co_nccl_cta16": {
        "label": "ColumnOverlap + NCCL (16 CTAs)",
        "color": "#009E73",
    },
    "all_co_nccl_cta64": {
        "label": "ColumnOverlap + NCCL (64 CTAs)",
        "color": "#CC79A7",
    },
}

FIGURE_SPECS = (
    (
        "tpch_sf1000_aggregate_plan_synthesis",
        ("legacy", "synthesized"),
        "single",
    ),
    (
        "tpch_sf1000_aggregate_legacy_alignment",
        ("legacy", "legacy_aligned_both"),
        "single",
    ),
    (
        "tpch_sf1000_aggregate_legacy_ce_pull",
        ("legacy", "legacy_ce_pull"),
        "single",
    ),
    (
        "tpch_sf1000_aggregate_column_overlap_backends",
        (
            "legacy",
            "all_co_nccl",
            "all_co_nccl_cta16",
            "all_co_nccl_cta64",
            "all_co_ce_pull",
        ),
        "double",
    ),
)

COMPONENT_FIGURE_SPECS = (
    (
        "tpch_sf1000_compute_baseline_vs_local_operators",
        ("legacy", "synthesized"),
        "compute",
    ),
    (
        "tpch_sf1000_communication_baseline_vs_ce_pull",
        ("legacy", "legacy_ce_pull"),
        "communication",
    ),
    (
        "tpch_sf1000_communication_baseline_vs_alignment",
        ("legacy", "legacy_aligned_both"),
        "communication",
    ),
)

SINGLE_COLUMN_SIZE = (3.45, 2.45)
DOUBLE_COLUMN_SIZE = (7.08, 3.05)


@dataclass(frozen=True)
class QuerySummary:
    gpus: int
    nodes: int
    arm: str
    query: int
    sample_count: int
    median_rank_max_ms: float
    min_rank_max_ms: float
    max_rank_max_ms: float
    mean_rank_max_ms: float
    stddev_rank_max_ms: float
    warmup_runs: int
    warmup_status: str
    attempt: int
    attempt_dir: str


@dataclass(frozen=True)
class SuiteSummary:
    gpus: int
    nodes: int
    arm: str
    query_count: int
    samples_per_query: int
    sum_query_medians_ms: float
    sum_query_medians_s: float
    legacy_sum_query_medians_ms: float
    speedup_vs_legacy: float
    improvement_vs_legacy_percent: float


@dataclass(frozen=True)
class BreakdownSummary:
    gpus: int
    nodes: int
    arm: str
    query_count: int
    breakdown_samples_per_query: int
    clean_total_ms: float
    raw_phase_total_ms: float
    phase_scale_to_clean: float
    compute_ms: float
    shuffle_ms: float
    other_communication_ms: float
    overhead_ms: float
    measured_overlap_ms: float


@dataclass(frozen=True)
class ComponentComparisonSummary:
    figure: str
    metric: str
    gpus: int
    nodes: int
    arm: str
    time_ms: float
    time_s: float


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repository / "results" / "distributed_tpch",
        help="directory containing the four campaign directories",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            repository
            / "results"
            / "distributed_tpch"
            / "figures"
            / "sf1000_fullset_reserve40g"
        ),
        help="output directory for summaries and figures",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing JSON file: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error


def read_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not path.is_file():
        raise ValueError(f"missing CSV file: {path}")
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
        columns = set(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows, columns


def campaign_dir(results_dir: Path, gpus: int) -> Path:
    return results_dir / f"sf1000_{gpus}gpu_fullset_reserve40g"


def attempt_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("attempt-"))
    except ValueError as error:
        raise ValueError(f"invalid attempt directory name: {path}") from error


def accepted_attempt(query_root: Path, arm: str, query: int) -> Path:
    if not query_root.is_dir():
        raise ValueError(f"missing timing cell: {query_root}")
    candidates = sorted(
        query_root.glob("attempt-*"), key=attempt_number, reverse=True
    )
    for attempt in candidates:
        result_path = attempt / "result.json"
        e2e_path = attempt / "e2e.csv"
        if (
            not result_path.is_file()
            or not e2e_path.is_file()
            or not e2e_path.stat().st_size
        ):
            continue
        result = read_json(result_path)
        if result.get("status") != "pass":
            continue
        if result.get("arm") != arm or int(result.get("query", -1)) != query:
            raise ValueError(f"result identity mismatch: {result_path}")
        if result.get("stage") != "timing":
            raise ValueError(f"expected timing result: {result_path}")
        return attempt
    raise ValueError(f"no successful timing attempt with e2e.csv: {query_root}")


def accepted_breakdown_attempt(query_root: Path, arm: str, query: int) -> Path:
    if not query_root.is_dir():
        raise ValueError(f"missing breakdown cell: {query_root}")
    candidates = sorted(
        query_root.glob("attempt-*"), key=attempt_number, reverse=True
    )
    for attempt in candidates:
        result_path = attempt / "result.json"
        steps_path = attempt / "steps.csv"
        phases_path = attempt / "phases.csv"
        if not all(
            path.is_file() and path.stat().st_size
            for path in (result_path, steps_path, phases_path)
        ):
            continue
        result = read_json(result_path)
        if result.get("status") != "pass":
            continue
        if result.get("arm") != arm or int(result.get("query", -1)) != query:
            raise ValueError(f"result identity mismatch: {result_path}")
        if result.get("stage") != "breakdown":
            raise ValueError(f"expected breakdown result: {result_path}")
        return attempt
    raise ValueError(
        f"no successful breakdown attempt with steps/phases CSVs: {query_root}"
    )


def require_single_value(rows: list[dict[str, str]], column: str, path: Path) -> str:
    values = {row[column] for row in rows}
    if len(values) != 1:
        raise ValueError(f"{path} has inconsistent {column}: {sorted(values)}")
    return values.pop()


def summarize_query(
    results_dir: Path, gpus: int, nodes: int, arm: str, query: int
) -> QuerySummary:
    root = campaign_dir(results_dir, gpus) / "arms" / arm / "timing" / f"q{query:02d}"
    attempt = accepted_attempt(root, arm, query)
    path = attempt / "e2e.csv"
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
    required = {
        "query",
        "scale",
        "ranks",
        "nodes",
        "iteration",
        "warmup_runs",
        "warmup_status",
        "rank_max_ms",
    }
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"{path} has {len(rows)} samples, expected {EXPECTED_SAMPLE_COUNT}"
        )
    if require_single_value(rows, "query", path).upper() != f"Q{query}":
        raise ValueError(f"query mismatch in {path}")
    if int(require_single_value(rows, "scale", path)) != EXPECTED_SCALE:
        raise ValueError(f"scale mismatch in {path}")
    if int(require_single_value(rows, "ranks", path)) != gpus:
        raise ValueError(f"rank count mismatch in {path}")
    if int(require_single_value(rows, "nodes", path)) != nodes:
        raise ValueError(f"node count mismatch in {path}")
    iterations = sorted(int(row["iteration"]) for row in rows)
    if iterations != list(range(EXPECTED_SAMPLE_COUNT)):
        raise ValueError(f"unexpected iterations in {path}: {iterations}")
    warmup_status = require_single_value(rows, "warmup_status", path)
    if warmup_status != "stable":
        raise ValueError(f"warmup did not stabilize in {path}: {warmup_status}")
    warmup_runs = int(require_single_value(rows, "warmup_runs", path))
    samples = [float(row["rank_max_ms"]) for row in rows]
    if not all(math.isfinite(value) and value > 0.0 for value in samples):
        raise ValueError(f"invalid rank_max_ms sample in {path}")
    return QuerySummary(
        gpus=gpus,
        nodes=nodes,
        arm=arm,
        query=query,
        sample_count=len(samples),
        median_rank_max_ms=statistics.median(samples),
        min_rank_max_ms=min(samples),
        max_rank_max_ms=max(samples),
        mean_rank_max_ms=statistics.fmean(samples),
        stddev_rank_max_ms=statistics.pstdev(samples),
        warmup_runs=warmup_runs,
        warmup_status=warmup_status,
        attempt=attempt_number(attempt),
        attempt_dir=str(attempt),
    )


def phase_component(row: dict[str, str]) -> str:
    category = row["category"]
    if category == "compute":
        return "Compute"
    if category == "communication":
        return (
            "Shuffle"
            if row["communication_pattern"] == "shuffle"
            else "Other communication"
        )
    raise ValueError(f"unsupported direct phase category: {category!r}")


def attribute_phase_intervals(
    rows: list[dict[str, str]], step_ms: float
) -> dict[str, float]:
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


def summarize_breakdown_query(
    results_dir: Path, gpus: int, nodes: int, arm: str, query: int
) -> tuple[dict[str, float], float, int]:
    root = (
        campaign_dir(results_dir, gpus)
        / "arms"
        / arm
        / "breakdown"
        / f"q{query:02d}"
    )
    attempt = accepted_breakdown_attempt(root, arm, query)
    steps_path = attempt / "steps.csv"
    phases_path = attempt / "phases.csv"
    step_rows, step_columns = read_csv(steps_path)
    phase_rows, phase_columns = read_csv(phases_path)
    required_steps = {
        "timestamp_utc",
        "query",
        "scale",
        "ranks",
        "nodes",
        "iteration",
        "record_type",
        "rank",
        "step_index",
        "gpu_ms",
    }
    required_phases = required_steps | {
        "category",
        "communication_pattern",
        "quality",
        "start_ms",
        "end_ms",
    }
    if required_steps - step_columns:
        raise ValueError(
            f"{steps_path} is missing columns: {sorted(required_steps - step_columns)}"
        )
    if required_phases - phase_columns:
        raise ValueError(
            f"{phases_path} is missing columns: "
            f"{sorted(required_phases - phase_columns)}"
        )
    for rows, path in ((step_rows, steps_path), (phase_rows, phases_path)):
        if require_single_value(rows, "query", path).upper() != f"Q{query}":
            raise ValueError(f"query mismatch in {path}")
        if int(require_single_value(rows, "scale", path)) != EXPECTED_SCALE:
            raise ValueError(f"scale mismatch in {path}")
        if int(require_single_value(rows, "ranks", path)) != gpus:
            raise ValueError(f"rank count mismatch in {path}")
        if int(require_single_value(rows, "nodes", path)) != nodes:
            raise ValueError(f"node count mismatch in {path}")

    step_totals: dict[tuple[str, int, int], float] = defaultdict(float)
    for row in step_rows:
        if row["record_type"] == "rank":
            key = (
                row["timestamp_utc"],
                int(row["iteration"]),
                int(row["rank"]),
            )
            step_totals[key] += float(row["gpu_ms"])
    critical_by_iteration: dict[tuple[str, int], tuple[int, float]] = {}
    for (timestamp, iteration, rank), total in step_totals.items():
        key = (timestamp, iteration)
        current = critical_by_iteration.get(key)
        if current is None or total > current[1] or (
            total == current[1] and rank < current[0]
        ):
            critical_by_iteration[key] = (rank, total)
    samples = [
        (total, timestamp, iteration, rank)
        for (timestamp, iteration), (rank, total) in critical_by_iteration.items()
    ]
    if len(samples) != EXPECTED_BREAKDOWN_SAMPLE_COUNT:
        raise ValueError(
            f"{steps_path} has {len(samples)} breakdown samples, "
            f"expected {EXPECTED_BREAKDOWN_SAMPLE_COUNT}"
        )
    samples.sort()
    selected_total, timestamp, iteration, rank = samples[(len(samples) - 1) // 2]
    selected_steps = {
        int(row["step_index"]): float(row["gpu_ms"])
        for row in step_rows
        if row["record_type"] == "rank"
        and row["timestamp_utc"] == timestamp
        and int(row["iteration"]) == iteration
        and int(row["rank"]) == rank
    }
    direct_by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
    components: dict[str, float] = defaultdict(float)
    for row in phase_rows:
        if (
            row["record_type"] != "rank"
            or row["timestamp_utc"] != timestamp
            or int(row["iteration"]) != iteration
            or int(row["rank"]) != rank
        ):
            continue
        if row["category"] == "overhead":
            components["Overhead"] += float(row["gpu_ms"])
        elif row["quality"] == "direct":
            direct_by_step[int(row["step_index"])].append(row)
    for step_index, rows in direct_by_step.items():
        if step_index not in selected_steps:
            raise ValueError(f"{phases_path} refers to missing step {step_index}")
        attributed = attribute_phase_intervals(rows, selected_steps[step_index])
        for component, value in attributed.items():
            components[component] += value

    phase_total = sum(components.values())
    tolerance = max(0.05, selected_total * 0.01)
    if abs(phase_total - selected_total) > tolerance:
        raise ValueError(
            f"{phases_path}: selected phase total {phase_total:.6f} does not "
            f"match step total {selected_total:.6f}"
        )
    overlap = components.get("Overlap", 0.0)
    if overlap > 1e-6:
        raise ValueError(
            f"{phases_path}: expected no compute/communication overlap, "
            f"measured {overlap:.6f} ms"
        )
    values = {component: components.get(component, 0.0) for component in PHASE_ORDER}
    return values, phase_total, len(samples)


def validate_campaigns(results_dir: Path) -> dict[str, object]:
    benchmark_hashes = set()
    campaign_fingerprints = {}
    hosts_by_gpu = {}
    expected_arms = set(ARM_STYLES)
    for gpus in GPU_COUNTS:
        directory = campaign_dir(results_dir, gpus)
        manifest = read_json(directory / "manifest" / "campaign.json")
        configuration = manifest.get("configuration", {})
        arguments = configuration.get("arguments", {})
        if int(arguments.get("scale", -1)) != EXPECTED_SCALE:
            raise ValueError(f"unexpected scale in {directory}")
        if int(arguments.get("memory_reserve_bytes", -1)) != EXPECTED_RESERVE_BYTES:
            raise ValueError(f"unexpected memory reserve in {directory}")
        if tuple(arguments.get("queries", ())) != QUERIES:
            raise ValueError(f"unexpected query set in {directory}")
        missing_arms = expected_arms - set(configuration.get("arms", ()))
        if missing_arms:
            raise ValueError(f"{directory} is missing arms: {sorted(missing_arms)}")
        hosts = arguments.get("hosts", "")
        slots = sum(int(entry.rsplit(":", 1)[1]) for entry in hosts.split(","))
        if slots != gpus:
            raise ValueError(f"{directory} host slots total {slots}, expected {gpus}")
        hosts_by_gpu[str(gpus)] = hosts
        campaign_fingerprints[str(gpus)] = manifest.get("fingerprint")
        benchmark_hashes.add(configuration.get("binary_sha256", {}).get("benchmark"))
    if len(benchmark_hashes) != 1 or None in benchmark_hashes:
        raise ValueError(f"campaign benchmark hashes differ: {sorted(benchmark_hashes)}")
    return {
        "scale": EXPECTED_SCALE,
        "queries": list(QUERIES),
        "gpu_counts": list(GPU_COUNTS),
        "memory_reserve_bytes": EXPECTED_RESERVE_BYTES,
        "benchmark_sha256": benchmark_hashes.pop(),
        "campaign_fingerprints": campaign_fingerprints,
        "hosts_by_gpu_count": hosts_by_gpu,
        "aggregate_definition": (
            "For each query, median rank_max_ms over ten clean iterations; "
            "suite value is the sum of Q1-Q22 medians."
        ),
        "breakdown_definition": (
            "For each query, select the lower-median complete critical-rank phase "
            "vector from five instrumented iterations, sum Q1-Q22 by component, "
            "and scale component shares to the clean aggregate."
        ),
        "chart_type": (
            "Pairwise grouped stacked bars and five-arm grouped bars, with clean "
            "aggregate seconds annotated above each bar"
        ),
        "component_comparison_definition": (
            "Compute-only figures use scaled compute_ms. Communication-only "
            "figures use scaled shuffle_ms + other_communication_ms."
        ),
        "figure_dimensions_inches": {
            "single_column": list(SINGLE_COLUMN_SIZE),
            "double_column": list(DOUBLE_COLUMN_SIZE),
        },
        "figure_output_layout": {"pdf": "pdf/", "png": "png/"},
        "font": "Matplotlib default family, 8 pt base size",
        "matplotlib_version": matplotlib.__version__,
    }


def collect(
    results_dir: Path,
) -> tuple[
    list[QuerySummary], list[SuiteSummary], list[BreakdownSummary], dict
]:
    provenance = validate_campaigns(results_dir)
    query_rows = []
    for gpus in GPU_COUNTS:
        nodes = gpus // 4
        for arm in ARM_STYLES:
            for query in QUERIES:
                query_rows.append(summarize_query(results_dir, gpus, nodes, arm, query))

    totals = {}
    for gpus in GPU_COUNTS:
        for arm in ARM_STYLES:
            matching = [
                row for row in query_rows if row.gpus == gpus and row.arm == arm
            ]
            if len(matching) != len(QUERIES):
                raise ValueError(f"incomplete aggregate for {gpus} GPUs, {arm}")
            totals[(gpus, arm)] = sum(row.median_rank_max_ms for row in matching)

    suite_rows = []
    for gpus in GPU_COUNTS:
        legacy_ms = totals[(gpus, "legacy")]
        for arm in ARM_STYLES:
            total_ms = totals[(gpus, arm)]
            speedup = legacy_ms / total_ms
            suite_rows.append(
                SuiteSummary(
                    gpus=gpus,
                    nodes=gpus // 4,
                    arm=arm,
                    query_count=len(QUERIES),
                    samples_per_query=EXPECTED_SAMPLE_COUNT,
                    sum_query_medians_ms=total_ms,
                    sum_query_medians_s=total_ms / 1000.0,
                    legacy_sum_query_medians_ms=legacy_ms,
                    speedup_vs_legacy=speedup,
                    improvement_vs_legacy_percent=(1.0 - 1.0 / speedup) * 100.0,
                )
            )
    breakdown_rows = []
    for gpus in GPU_COUNTS:
        nodes = gpus // 4
        for arm in BREAKDOWN_ARMS:
            raw_components: dict[str, float] = defaultdict(float)
            raw_phase_total = 0.0
            sample_counts = set()
            for query in QUERIES:
                components, phase_total, sample_count = summarize_breakdown_query(
                    results_dir, gpus, nodes, arm, query
                )
                for component, value in components.items():
                    raw_components[component] += value
                raw_phase_total += phase_total
                sample_counts.add(sample_count)
            if sample_counts != {EXPECTED_BREAKDOWN_SAMPLE_COUNT}:
                raise ValueError(
                    f"inconsistent breakdown sample counts for {gpus} GPUs, {arm}: "
                    f"{sorted(sample_counts)}"
                )
            clean_total = totals[(gpus, arm)]
            scale = clean_total / raw_phase_total
            scaled = {
                component: raw_components[component] * scale
                for component in PHASE_ORDER
            }
            breakdown_rows.append(
                BreakdownSummary(
                    gpus=gpus,
                    nodes=nodes,
                    arm=arm,
                    query_count=len(QUERIES),
                    breakdown_samples_per_query=EXPECTED_BREAKDOWN_SAMPLE_COUNT,
                    clean_total_ms=clean_total,
                    raw_phase_total_ms=raw_phase_total,
                    phase_scale_to_clean=scale,
                    compute_ms=scaled["Compute"],
                    shuffle_ms=scaled["Shuffle"],
                    other_communication_ms=scaled["Other communication"],
                    overhead_ms=scaled["Overhead"],
                    measured_overlap_ms=0.0,
                )
            )
    return query_rows, suite_rows, breakdown_rows, provenance


def write_dataclass_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def collect_component_comparisons(
    breakdown_rows: list[BreakdownSummary],
) -> list[ComponentComparisonSummary]:
    rows = []
    for figure_name, arms, metric in COMPONENT_FIGURE_SPECS:
        for arm in arms:
            arm_rows = sorted(
                (row for row in breakdown_rows if row.arm == arm),
                key=lambda row: row.gpus,
            )
            if len(arm_rows) != len(GPU_COUNTS):
                raise ValueError(
                    f"incomplete {metric} comparison data for {arm}: "
                    f"{len(arm_rows)} GPU counts"
                )
            for row in arm_rows:
                if metric == "compute":
                    time_ms = row.compute_ms
                elif metric == "communication":
                    time_ms = row.shuffle_ms + row.other_communication_ms
                else:
                    raise ValueError(f"unsupported comparison metric: {metric}")
                rows.append(
                    ComponentComparisonSummary(
                        figure=figure_name,
                        metric=metric,
                        gpus=row.gpus,
                        nodes=row.nodes,
                        arm=arm,
                        time_ms=time_ms,
                        time_s=time_ms / 1000.0,
                    )
                )
    return rows


def style_axis(
    axis, y_max: float, y_label: str = "Sum of query medians (s)"
) -> None:
    axis.set_xlabel("Number of GPUs")
    axis.set_ylabel(y_label)
    axis.set_ylim(0.0, y_max)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    axis.grid(axis="y", color="#D5D5D5", linestyle="-", linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.0, width=0.7)


def annotate_total(axis, x_position: float, total_s: float, fontsize: float) -> None:
    axis.annotate(
        f"{total_s:.2f}",
        xy=(x_position, total_s),
        xytext=(0, 2.5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="#222222",
        clip_on=False,
    )


def save_figure(figure, out_dir: Path, name: str) -> None:
    for suffix in (".pdf", ".png"):
        metadata = (
            {"CreationDate": None, "ModDate": None} if suffix == ".pdf" else None
        )
        output_path = out_dir / suffix.removeprefix(".") / f"{name}{suffix}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, facecolor="white", metadata=metadata)
    plt.close(figure)


def plot_pairwise_breakdown(
    out_dir: Path,
    name: str,
    arms: tuple[str, ...],
    breakdown_rows: list[BreakdownSummary],
    y_max: float,
) -> None:
    figure, axis = plt.subplots(figsize=SINGLE_COLUMN_SIZE)
    centers = list(range(len(GPU_COUNTS)))
    bar_width = 0.72 / len(arms)
    for arm_index, arm in enumerate(arms):
        rows = sorted(
            (row for row in breakdown_rows if row.arm == arm),
            key=lambda row: row.gpus,
        )
        if len(rows) != len(GPU_COUNTS):
            raise ValueError(f"incomplete breakdown plot data for {arm}")
        positions = [
            center + (arm_index - (len(arms) - 1) / 2.0) * bar_width
            for center in centers
        ]
        bottoms = [0.0] * len(rows)
        for component in PHASE_ORDER:
            values = [
                getattr(row, PHASE_FIELDS[component]) / 1000.0 for row in rows
            ]
            axis.bar(
                positions,
                values,
                width=bar_width * 0.92,
                bottom=bottoms,
                color=PHASE_COLORS[component],
                edgecolor="#4A4A4A",
                linewidth=0.45,
                hatch="///" if arm == "legacy" else None,
                zorder=3,
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        for position, row in zip(positions, rows):
            annotate_total(axis, position, row.clean_total_ms / 1000.0, 6.5)

    style_axis(axis, y_max)
    axis.set_xticks(centers, [str(gpus) for gpus in GPU_COUNTS])
    axis.set_xlim(-0.58, len(GPU_COUNTS) - 0.42)
    figure.subplots_adjust(left=0.175, right=0.98, bottom=0.19, top=0.97)

    phase_handles = {
        component: mpatches.Patch(
            facecolor=PHASE_COLORS[component],
            edgecolor="#4A4A4A",
            linewidth=0.45,
            label=component,
        )
        for component in PHASE_ORDER
    }
    arm_handles = {
        "legacy": mpatches.Patch(
            facecolor="white",
            edgecolor="#4A4A4A",
            linewidth=0.6,
            hatch="///",
            label=ARM_STYLES["legacy"]["label"],
        ),
        arms[1]: mpatches.Patch(
            facecolor="white",
            edgecolor="#4A4A4A",
            linewidth=0.6,
            label=ARM_STYLES[arms[1]]["label"],
        ),
    }
    # Legend columns read as: Compute/Shuffle, Other/Overhead, Baseline/variant.
    handles = [
        phase_handles["Compute"],
        phase_handles["Other communication"],
        arm_handles["legacy"],
        phase_handles["Shuffle"],
        phase_handles["Overhead"],
        arm_handles[arms[1]],
    ]
    axis.legend(
        handles=handles,
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=5.25,
        handlelength=1.45,
        handletextpad=0.4,
        columnspacing=0.65,
        borderpad=0.3,
        labelspacing=0.25,
    )
    save_figure(figure, out_dir, name)


def plot_component_comparison(
    out_dir: Path,
    name: str,
    arms: tuple[str, ...],
    metric: str,
    component_rows: list[ComponentComparisonSummary],
) -> None:
    figure, axis = plt.subplots(figsize=SINGLE_COLUMN_SIZE)
    centers = list(range(len(GPU_COUNTS)))
    bar_width = 0.72 / len(arms)
    selected_rows = [row for row in component_rows if row.figure == name]
    maximum = max(row.time_s for row in selected_rows)
    y_max = math.ceil(maximum * 1.18 * 10.0) / 10.0

    for arm_index, arm in enumerate(arms):
        rows = sorted(
            (row for row in selected_rows if row.arm == arm),
            key=lambda row: row.gpus,
        )
        positions = [
            center + (arm_index - (len(arms) - 1) / 2.0) * bar_width
            for center in centers
        ]
        bars = axis.bar(
            positions,
            [row.time_s for row in rows],
            width=bar_width * 0.92,
            label=ARM_STYLES[arm]["label"],
            color=ARM_STYLES[arm]["color"],
            edgecolor="#4A4A4A",
            linewidth=0.45,
            zorder=3,
        )
        for bar, row in zip(bars, rows):
            annotate_total(
                axis,
                bar.get_x() + bar.get_width() / 2.0,
                row.time_s,
                6.5,
            )

    y_label = (
        "Aggregate compute time (s)"
        if metric == "compute"
        else "Aggregate communication time (s)"
    )
    style_axis(axis, y_max, y_label)
    axis.set_xticks(centers, [str(gpus) for gpus in GPU_COUNTS])
    axis.set_xlim(-0.58, len(GPU_COUNTS) - 0.42)
    figure.subplots_adjust(left=0.175, right=0.98, bottom=0.19, top=0.97)
    axis.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=6.5,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.35,
    )
    save_figure(figure, out_dir, name)


def plot_one(
    out_dir: Path,
    name: str,
    arms: tuple[str, ...],
    width: str,
    suite_rows: list[SuiteSummary],
    breakdown_rows: list[BreakdownSummary],
    y_max: float,
) -> None:
    if width == "single":
        plot_pairwise_breakdown(
            out_dir, name, arms, breakdown_rows, y_max
        )
        return
    figsize = SINGLE_COLUMN_SIZE if width == "single" else DOUBLE_COLUMN_SIZE
    figure, axis = plt.subplots(figsize=figsize)
    centers = list(range(len(GPU_COUNTS)))
    group_width = 0.72 if width == "single" else 0.82
    bar_width = group_width / len(arms)
    for arm_index, arm in enumerate(arms):
        rows = sorted(
            (row for row in suite_rows if row.arm == arm), key=lambda row: row.gpus
        )
        style = ARM_STYLES[arm]
        positions = [
            center + (arm_index - (len(arms) - 1) / 2.0) * bar_width
            for center in centers
        ]
        bars = axis.bar(
            positions,
            [row.sum_query_medians_s for row in rows],
            width=bar_width * 0.92,
            label=style["label"],
            color=style["color"],
            edgecolor="#333333",
            linewidth=0.45,
            zorder=3,
        )
        for bar, row in zip(bars, rows):
            annotate_total(
                axis,
                bar.get_x() + bar.get_width() / 2.0,
                row.sum_query_medians_s,
                6.75,
            )
    style_axis(axis, y_max)
    axis.set_xticks(centers, [str(gpus) for gpus in GPU_COUNTS])
    axis.set_xlim(-0.58, len(GPU_COUNTS) - 0.42)
    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.16, top=0.97)
    handles, labels = axis.get_legend_handles_labels()
    # Matplotlib fills legend columns vertically. Reorder five entries to
    # read by row as baseline/auto, CTA16/CTA64, then CE-pull.
    order = (0, 2, 4, 1, 3)
    axis.legend(
        handles=[handles[index] for index in order],
        labels=[labels[index] for index in order],
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=6.25,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=0.9,
        borderpad=0.35,
        labelspacing=0.3,
    )
    save_figure(figure, out_dir, name)


def plot_all(
    out_dir: Path,
    suite_rows: list[SuiteSummary],
    breakdown_rows: list[BreakdownSummary],
    component_rows: list[ComponentComparisonSummary],
) -> None:
    maximum = max(row.sum_query_medians_s for row in suite_rows)
    y_max = math.ceil(maximum * 1.08 * 2.0) / 2.0
    for name, arms, width in FIGURE_SPECS:
        plot_one(
            out_dir, name, arms, width, suite_rows, breakdown_rows, y_max
        )
    for name, arms, metric in COMPONENT_FIGURE_SPECS:
        plot_component_comparison(
            out_dir, name, arms, metric, component_rows
        )


def main() -> int:
    args = parse_args()
    try:
        query_rows, suite_rows, breakdown_rows, provenance = collect(
            args.results_dir.resolve()
        )
        component_rows = collect_component_comparisons(breakdown_rows)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_dataclass_csv(args.out_dir / "query_summary.csv", query_rows)
        write_dataclass_csv(args.out_dir / "suite_summary.csv", suite_rows)
        write_dataclass_csv(args.out_dir / "breakdown_summary.csv", breakdown_rows)
        write_dataclass_csv(
            args.out_dir / "component_comparison_summary.csv", component_rows
        )
        (args.out_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        plot_all(args.out_dir, suite_rows, breakdown_rows, component_rows)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    figure_count = len(FIGURE_SPECS) + len(COMPONENT_FIGURE_SPECS)
    print(f"wrote summaries and {figure_count} PNG/PDF figure pairs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
