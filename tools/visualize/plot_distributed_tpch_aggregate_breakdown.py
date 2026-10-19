#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot the aggregate legacy_cta64 computation/communication breakdown.

The figure compares the scale-matched configurations SF=1000 on 8 GPUs and
SF=3000 on 24 GPUs (SF=125 per GPU).  For each query, clean latency is the
median maximum-rank wall time over ten timing iterations.  The communication
fraction is the median, over five breakdown iterations, of the interval-union
communication time on that iteration's critical rank.  Per-query components
are scaled to clean latency and then summed over Q1-Q22.

SF=3000 Q3 and Q18 are intentionally included when their failed-verification
attempts report complete measurements, as requested for this presentation.
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
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    sys.exit(f"error: matplotlib is required: {error}")


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGURATIONS = ((1000, 8), (3000, 24))
QUERIES = tuple(range(1, 23))
TIMING_SAMPLES = 10
BREAKDOWN_SAMPLES = 5
ALLOWED_VERIFICATION_FAILURES = frozenset(((3000, 3), (3000, 18)))
COLORS = {"Computation": "#4C78A8", "Communication": "#F58518"}
FIGURE_NAME = "legacy_cta64_aggregate_time_breakdown"


@dataclass(frozen=True)
class QueryBreakdown:
    scale_factor: int
    gpus: int
    nodes: int
    query: int
    median_e2e_rank_max_ms: float
    median_communication_fraction: float
    computation_ms: float
    communication_ms: float
    timing_result_status: str
    breakdown_result_status: str
    timing_attempt_dir: str
    breakdown_attempt_dir: str


@dataclass(frozen=True)
class AggregateBreakdown:
    scale_factor: int
    gpus: int
    nodes: int
    query_count: int
    computation_s: float
    communication_s: float
    total_s: float
    communication_percent: float


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
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_dataclass_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)
    print(f"wrote {path}")


def attempt_number(path: Path) -> int:
    return int(path.name.removeprefix("attempt-"))


def select_attempt(
    root: Path, required: tuple[str, ...], scale: int, query: int
) -> tuple[Path, dict[str, object]]:
    attempts = sorted(root.glob("attempt-*"), key=attempt_number, reverse=True)
    for attempt in attempts:
        result_path = attempt / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        accepted = result.get("status") == "pass"
        if (scale, query) in ALLOWED_VERIFICATION_FAILURES:
            accepted = accepted or (
                result.get("status") == "fail"
                and result.get("failure_kind") == "verification"
                and result.get("measurements_complete") is True
            )
        if not accepted:
            continue
        if all(
            (attempt / name).is_file() and (attempt / name).stat().st_size > 0
            for name in required
        ):
            return attempt, result
    raise ValueError(f"no usable attempt under {root}")


def campaign_root(
    results_dir: Path, scale: int, gpus: int, arm: str = "legacy_cta64"
) -> Path:
    return (
        results_dir
        / f"sf{scale}"
        / arm
        / f"{gpus}gpu_128p_timing_breakdown"
        / "arms"
        / arm
    )


def require_iterations(
    rows: list[dict[str, str]], expected_count: int, path: Path
) -> None:
    iterations = sorted({int(row["iteration"]) for row in rows})
    if iterations != list(range(expected_count)):
        raise ValueError(
            f"{path}: iterations {iterations}, expected 0..{expected_count - 1}"
        )


def timing_summary(
    results_dir: Path,
    scale: int,
    gpus: int,
    query: int,
    arm: str = "legacy_cta64",
) -> tuple[float, Path, dict[str, object]]:
    root = (
        campaign_root(results_dir, scale, gpus, arm)
        / "timing"
        / f"q{query:02d}"
    )
    attempt, result = select_attempt(root, ("e2e.csv",), scale, query)
    path = attempt / "e2e.csv"
    rows = read_csv(path)
    require_iterations(rows, TIMING_SAMPLES, path)
    if len(rows) != TIMING_SAMPLES:
        raise ValueError(f"{path}: {len(rows)} rows, expected {TIMING_SAMPLES}")
    values = []
    for row in rows:
        if int(row["scale"]) != scale or int(row["ranks"]) != gpus:
            raise ValueError(f"{path}: scale/rank mismatch")
        if row["warmup_status"] != "stable":
            raise ValueError(f"{path}: warmup status is {row['warmup_status']!r}")
        value = float(row["rank_max_ms"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{path}: invalid rank_max_ms {value}")
        values.append(value)
    return statistics.median(values), attempt, result


def union_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    current_start, current_end = ordered[0]
    total = 0.0
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def communication_fraction_summary(
    results_dir: Path,
    scale: int,
    gpus: int,
    query: int,
    arm: str = "legacy_cta64",
) -> tuple[float, Path, dict[str, object]]:
    root = (
        campaign_root(results_dir, scale, gpus, arm)
        / "breakdown"
        / f"q{query:02d}"
    )
    attempt, result = select_attempt(
        root, ("steps.csv", "phases.csv"), scale, query
    )
    steps_path = attempt / "steps.csv"
    phases_path = attempt / "phases.csv"
    step_rows = read_csv(steps_path)
    phase_rows = read_csv(phases_path)
    require_iterations(step_rows, BREAKDOWN_SAMPLES, steps_path)
    require_iterations(phase_rows, BREAKDOWN_SAMPLES, phases_path)

    step_duration: dict[tuple[str, int, int, int], float] = {}
    rank_total: dict[tuple[str, int, int], float] = defaultdict(float)
    for row in step_rows:
        if row["record_type"] != "rank":
            continue
        if int(row["scale"]) != scale or int(row["ranks"]) != gpus:
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
        step_duration[key] = duration
        rank_total[key[:3]] += duration

    critical: dict[tuple[str, int], tuple[int, float]] = {}
    for (timestamp, iteration, rank), total in rank_total.items():
        key = (timestamp, iteration)
        current = critical.get(key)
        if current is None or total > current[1] or (
            total == current[1] and rank < current[0]
        ):
            critical[key] = (rank, total)
    if len(critical) != BREAKDOWN_SAMPLES:
        raise ValueError(
            f"{steps_path}: {len(critical)} samples, expected {BREAKDOWN_SAMPLES}"
        )

    intervals_by_step: dict[
        tuple[str, int, int, int], list[tuple[float, float]]
    ] = defaultdict(list)
    for row in phase_rows:
        if (
            row["record_type"] != "rank"
            or row["quality"] != "direct"
            or row["category"] != "communication"
        ):
            continue
        timestamp = row["timestamp_utc"]
        iteration = int(row["iteration"])
        rank = int(row["rank"])
        if critical.get((timestamp, iteration), (-1, 0.0))[0] != rank:
            continue
        step_index = int(row["step_index"])
        step_key = (timestamp, iteration, rank, step_index)
        if step_key not in step_duration:
            raise ValueError(f"{phases_path}: missing step {step_key}")
        duration = step_duration[step_key]
        start = max(0.0, float(row["start_ms"]))
        end = min(duration, float(row["end_ms"]))
        if start < end:
            intervals_by_step[step_key].append((start, end))

    fractions = []
    for (timestamp, iteration), (rank, total) in critical.items():
        communication = sum(
            union_length(intervals)
            for (phase_timestamp, phase_iteration, phase_rank, _), intervals
            in intervals_by_step.items()
            if phase_timestamp == timestamp
            and phase_iteration == iteration
            and phase_rank == rank
        )
        tolerance = max(0.05, total * 0.01)
        if communication < 0.0 or communication > total + tolerance:
            raise ValueError(
                f"{phases_path}: communication {communication} exceeds {total}"
            )
        fractions.append(min(communication, total) / total)
    return statistics.median(fractions), attempt, result


def collect(
    results_dir: Path, arm: str = "legacy_cta64"
) -> tuple[list[QueryBreakdown], list[AggregateBreakdown]]:
    query_rows = []
    for scale, gpus in CONFIGURATIONS:
        for query in QUERIES:
            median_e2e, timing_attempt, timing_result = timing_summary(
                results_dir, scale, gpus, query, arm
            )
            communication_fraction, breakdown_attempt, breakdown_result = (
                communication_fraction_summary(
                    results_dir, scale, gpus, query, arm
                )
            )
            communication_ms = median_e2e * communication_fraction
            query_rows.append(
                QueryBreakdown(
                    scale_factor=scale,
                    gpus=gpus,
                    nodes=gpus // 4,
                    query=query,
                    median_e2e_rank_max_ms=median_e2e,
                    median_communication_fraction=communication_fraction,
                    computation_ms=median_e2e - communication_ms,
                    communication_ms=communication_ms,
                    timing_result_status=str(timing_result["status"]),
                    breakdown_result_status=str(breakdown_result["status"]),
                    timing_attempt_dir=str(timing_attempt.relative_to(results_dir)),
                    breakdown_attempt_dir=str(
                        breakdown_attempt.relative_to(results_dir)
                    ),
                )
            )

    aggregate_rows = []
    for scale, gpus in CONFIGURATIONS:
        selected = [row for row in query_rows if row.scale_factor == scale]
        if len(selected) != len(QUERIES):
            raise ValueError(f"SF={scale}: expected {len(QUERIES)} query rows")
        computation_ms = sum(row.computation_ms for row in selected)
        communication_ms = sum(row.communication_ms for row in selected)
        total_ms = sum(row.median_e2e_rank_max_ms for row in selected)
        if not math.isclose(
            computation_ms + communication_ms, total_ms, rel_tol=1e-12
        ):
            raise ValueError(f"SF={scale}: stack does not equal clean aggregate")
        aggregate_rows.append(
            AggregateBreakdown(
                scale_factor=scale,
                gpus=gpus,
                nodes=gpus // 4,
                query_count=len(selected),
                computation_s=computation_ms / 1000.0,
                communication_s=communication_ms / 1000.0,
                total_s=total_ms / 1000.0,
                communication_percent=100.0 * communication_ms / total_ms,
            )
        )
    return query_rows, aggregate_rows


def label_segment(
    ax: plt.Axes, x: float, bottom: float, height: float, total: float
) -> None:
    if height / total < 0.075:
        return
    ax.text(
        x,
        bottom + height / 2.0,
        f"{height:.2f} s\n({100.0 * height / total:.1f}%)",
        ha="center",
        va="center",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        linespacing=1.15,
    )


def plot(aggregate_rows: list[AggregateBreakdown], output_dir: Path) -> None:
    rows = sorted(aggregate_rows, key=lambda row: row.scale_factor)
    x = list(range(len(rows)))
    computation = [row.computation_s for row in rows]
    communication = [row.communication_s for row in rows]
    totals = [row.total_s for row in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.55))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    width = 0.58
    ax.bar(
        x,
        computation,
        width,
        color=COLORS["Computation"],
        edgecolor="white",
        linewidth=0.8,
        label="Computation",
        zorder=3,
    )
    ax.bar(
        x,
        communication,
        width,
        bottom=computation,
        color=COLORS["Communication"],
        edgecolor="white",
        linewidth=0.8,
        label="Communication",
        zorder=3,
    )

    for index, total in enumerate(totals):
        label_segment(ax, x[index], 0.0, computation[index], total)
        label_segment(
            ax, x[index], computation[index], communication[index], total
        )
        ax.text(
            x[index],
            total + max(totals) * 0.025,
            f"{total:.2f} s",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#2F2F2F",
        )

    ax.set_xticks(
        x,
        [f"SF={row.scale_factor}\n{row.gpus} GPUs" for row in rows],
    )
    ax.set_ylabel("Aggregate execution time (s)")
    ax.set_title("Distributed TPC-H Aggregate Time Breakdown", pad=14)
    ax.set_ylim(0.0, max(totals) * 1.20)
    ax.set_xlim(-0.65, len(rows) - 0.35)
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
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.87, bottom=0.18)

    for suffix in ("pdf", "png"):
        format_dir = output_dir / suffix
        format_dir.mkdir(parents=True, exist_ok=True)
        output = format_dir / f"{FIGURE_NAME}.{suffix}"
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        print(f"wrote {output}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    query_rows, aggregate_rows = collect(results_dir)
    write_dataclass_csv(output_dir / "query_breakdown.csv", query_rows)
    write_dataclass_csv(output_dir / "aggregate_breakdown.csv", aggregate_rows)
    plot(aggregate_rows, output_dir)
    for row in aggregate_rows:
        print(
            f"SF={row.scale_factor}, {row.gpus} GPUs: "
            f"compute={row.computation_s:.3f} s, "
            f"communication={row.communication_s:.3f} s, "
            f"total={row.total_s:.3f} s"
        )


if __name__ == "__main__":
    main()
