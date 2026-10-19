#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot directly measured compute-phase time by owning physical operator.

The figure compares legacy_cta64 (Baseline) and synthesized_cta64
(Opt. comp) for SF=1000 on 8 GPUs and SF=3000 on 24 GPUs.  For every query,
the script selects the median-total breakdown sample after choosing that
sample's critical rank.  It then unions direct compute-phase intervals within
each physical step and attributes that time to the step's ``step_kind``.

The resulting bars are strict phase-instrumented compute time.  They do not
include communication, phase overhead, or uninstrumented portions of a step,
and they are not rescaled to clean end-to-end latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
            "font.size": 13,
            "axes.titlesize": 20,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    sys.exit(f"error: matplotlib is required: {error}")


REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES = tuple(range(1, 23))
BREAKDOWN_SAMPLES = 5
FIGURE_NAME = "distributed_tpch_compute_operator_breakdown"

ARMS = (
    ("legacy_cta64", "Baseline"),
    ("synthesized_cta64", "Opt. comp"),
)
CONFIGURATIONS = ((1000, 8), (3000, 24))

# Preserve every step kind that owns a direct compute phase in the selected
# campaigns.  Large/common operators are placed first so their segments have
# a stable baseline; small/specialized operators appear above them.
OPERATOR_ORDER = (
    "DistributedEqJoin",
    "DistributedGroupBy",
    "DistributedSort",
    "DistributedScalarAgg",
    "DistributedDistinct",
    "Filter",
    "Project",
    "Reorder",
    "DistributedAntiJoin",
    "DistributedSemiJoin",
    "DistributedCrossJoin",
    "DistributedTopK",
    "Concat",
)

OPERATOR_LABELS = {
    "DistributedEqJoin": "Eq join",
    "DistributedGroupBy": "Group by",
    "DistributedSort": "Sort",
    "DistributedScalarAgg": "Scalar agg.",
    "DistributedDistinct": "Distinct",
    "Filter": "Filter",
    "Project": "Project",
    "Reorder": "Reorder",
    "DistributedAntiJoin": "Anti join",
    "DistributedSemiJoin": "Semi join",
    "DistributedCrossJoin": "Cross join",
    "DistributedTopK": "Top-K",
    "Concat": "Concat",
}

# A colorblind-friendly extension of the Tableau palette.  White segment
# boundaries keep adjacent small categories legible in the stacked bars.
OPERATOR_COLORS = {
    "DistributedEqJoin": "#4E79A7",
    "DistributedGroupBy": "#F28E2B",
    "DistributedSort": "#59A14F",
    "DistributedScalarAgg": "#E15759",
    "DistributedDistinct": "#D37295",
    "Filter": "#B07AA1",
    "Project": "#76B7B2",
    "Reorder": "#EDC948",
    "DistributedAntiJoin": "#9C755F",
    "DistributedSemiJoin": "#FF9DA7",
    "DistributedCrossJoin": "#86BCB6",
    "DistributedTopK": "#BAB0AC",
    "Concat": "#AF7AA1",
}


@dataclass(frozen=True)
class Setting:
    scale_factor: int
    gpus: int
    arm: str
    arm_label: str

    @property
    def key(self) -> str:
        return f"sf{self.scale_factor}_{self.gpus}gpu_{self.arm}"


SETTINGS = tuple(
    Setting(scale, gpus, arm, arm_label)
    for scale, gpus in CONFIGURATIONS
    for arm, arm_label in ARMS
)


@dataclass(frozen=True)
class QuerySelection:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    query: int
    timestamp_utc: str
    iteration: int
    critical_rank: int
    critical_step_total_ms: float
    strict_compute_ms: float
    strict_compute_fraction_of_step_time: float
    result_status: str
    failure_kind: str
    steps_csv: str
    phases_csv: str


@dataclass(frozen=True)
class OperatorTime:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    arm_label: str
    query: int
    operator_kind: str
    operator_label: str
    compute_ms: float


@dataclass(frozen=True)
class AggregateOperatorTime:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    arm_label: str
    operator_kind: str
    operator_label: str
    compute_ms: float
    compute_s: float


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


def attempt_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def campaign_root(results_dir: Path, setting: Setting) -> Path:
    return (
        results_dir
        / f"sf{setting.scale_factor}"
        / setting.arm
        / f"{setting.gpus}gpu_128p_timing_breakdown"
        / "arms"
        / setting.arm
    )


def accepted_result(result: dict[str, object], setting: Setting, query: int) -> bool:
    if result.get("status") == "pass":
        return True
    return (
        setting.scale_factor == 3000
        and query in (3, 18)
        and result.get("status") == "fail"
        and result.get("failure_kind") == "verification"
        and result.get("measurements_complete") is True
    )


def select_attempt(
    results_dir: Path, setting: Setting, query: int
) -> tuple[Path, dict[str, object]]:
    root = campaign_root(results_dir, setting) / "breakdown" / f"q{query:02d}"
    for attempt in sorted(root.glob("attempt-*"), key=attempt_number, reverse=True):
        result_path = attempt / "result.json"
        steps_path = attempt / "steps.csv"
        phases_path = attempt / "phases.csv"
        if not all(path.is_file() and path.stat().st_size for path in (
            result_path,
            steps_path,
            phases_path,
        )):
            continue
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if accepted_result(result, setting, query):
            return attempt, result
    raise ValueError(f"no usable breakdown attempt under {root}")


def require_columns(
    rows: list[dict[str, str]], required: set[str], path: Path
) -> None:
    columns = set(rows[0]) if rows else set()
    missing = required - columns
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


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
    results_dir: Path, setting: Setting, query: int
) -> tuple[QuerySelection, list[OperatorTime]]:
    attempt, result = select_attempt(results_dir, setting, query)
    steps_path = attempt / "steps.csv"
    phases_path = attempt / "phases.csv"
    steps = read_csv(steps_path)
    phases = read_csv(phases_path)
    common = {
        "timestamp_utc",
        "query",
        "scale",
        "ranks",
        "iteration",
        "record_type",
        "rank",
        "step_index",
        "step_kind",
        "gpu_ms",
    }
    require_columns(steps, common, steps_path)
    require_columns(
        phases,
        common | {"category", "quality", "start_ms", "end_ms"},
        phases_path,
    )

    for rows, path in ((steps, steps_path), (phases, phases_path)):
        iterations = sorted({int(row["iteration"]) for row in rows})
        if iterations != list(range(BREAKDOWN_SAMPLES)):
            raise ValueError(
                f"{path}: iterations {iterations}, expected "
                f"0..{BREAKDOWN_SAMPLES - 1}"
            )
        for row in rows:
            if row["query"].upper() != f"Q{query}":
                raise ValueError(f"{path}: query mismatch")
            if int(row["scale"]) != setting.scale_factor:
                raise ValueError(f"{path}: scale-factor mismatch")
            if int(row["ranks"]) != setting.gpus:
                raise ValueError(f"{path}: rank-count mismatch")

    step_duration: dict[tuple[str, int, int, int], float] = {}
    step_kind: dict[tuple[str, int, int, int], str] = {}
    rank_totals: dict[tuple[str, int, int], float] = defaultdict(float)
    for row in steps:
        if row["record_type"] != "rank":
            continue
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
        rank_totals[key[:3]] += duration

    critical: dict[tuple[str, int], tuple[int, float]] = {}
    for (timestamp, iteration, rank), total in rank_totals.items():
        key = (timestamp, iteration)
        current = critical.get(key)
        if current is None or total > current[1] or (
            total == current[1] and rank < current[0]
        ):
            critical[key] = (rank, total)
    if len(critical) != BREAKDOWN_SAMPLES:
        raise ValueError(
            f"{steps_path}: found {len(critical)} critical-rank samples, "
            f"expected {BREAKDOWN_SAMPLES}"
        )

    # Keep one coherent sample: the median critical-rank step total.  Sorting
    # the remaining fields makes ties deterministic.
    candidates = sorted(
        (total, timestamp, iteration, rank)
        for (timestamp, iteration), (rank, total) in critical.items()
    )
    selected_total, timestamp, iteration, rank = candidates[len(candidates) // 2]
    selected_steps = {
        key: duration
        for key, duration in step_duration.items()
        if key[:3] == (timestamp, iteration, rank)
    }

    compute_intervals: dict[tuple[str, int, int, int], list[tuple[float, float]]] = (
        defaultdict(list)
    )
    observed_operators: set[str] = set()
    for row in phases:
        if (
            row["record_type"] != "rank"
            or row["quality"] != "direct"
            or row["category"] != "compute"
            or row["timestamp_utc"] != timestamp
            or int(row["iteration"]) != iteration
            or int(row["rank"]) != rank
        ):
            continue
        key = (timestamp, iteration, rank, int(row["step_index"]))
        if key not in selected_steps:
            raise ValueError(f"{phases_path}: phase refers to missing step {key}")
        if row["step_kind"] != step_kind[key]:
            raise ValueError(f"{phases_path}: step-kind mismatch for {key}")
        operator = row["step_kind"]
        if operator not in OPERATOR_ORDER:
            raise ValueError(f"{phases_path}: unknown compute operator {operator!r}")
        duration = selected_steps[key]
        start = max(0.0, float(row["start_ms"]))
        end = min(duration, float(row["end_ms"]))
        if start < end:
            compute_intervals[key].append((start, end))
            observed_operators.add(operator)

    by_operator: dict[str, float] = defaultdict(float)
    for key, intervals in compute_intervals.items():
        by_operator[step_kind[key]] += interval_union(intervals)
    strict_compute = sum(by_operator.values())
    if strict_compute > selected_total + max(0.05, selected_total * 0.01):
        raise ValueError(
            f"{phases_path}: compute time {strict_compute:.6f} ms exceeds "
            f"step time {selected_total:.6f} ms"
        )
    if not observed_operators:
        raise ValueError(f"{phases_path}: selected sample has no compute phases")

    relative_steps = str(steps_path.relative_to(results_dir))
    relative_phases = str(phases_path.relative_to(results_dir))
    selection = QuerySelection(
        setting_key=setting.key,
        scale_factor=setting.scale_factor,
        gpus=setting.gpus,
        arm=setting.arm,
        query=query,
        timestamp_utc=timestamp,
        iteration=iteration,
        critical_rank=rank,
        critical_step_total_ms=selected_total,
        strict_compute_ms=strict_compute,
        strict_compute_fraction_of_step_time=(
            strict_compute / selected_total if selected_total else 0.0
        ),
        result_status=str(result["status"]),
        failure_kind=str(result.get("failure_kind") or ""),
        steps_csv=relative_steps,
        phases_csv=relative_phases,
    )
    rows = [
        OperatorTime(
            setting_key=setting.key,
            scale_factor=setting.scale_factor,
            gpus=setting.gpus,
            arm=setting.arm,
            arm_label=setting.arm_label,
            query=query,
            operator_kind=operator,
            operator_label=OPERATOR_LABELS[operator],
            compute_ms=by_operator.get(operator, 0.0),
        )
        for operator in OPERATOR_ORDER
    ]
    return selection, rows


def aggregate(rows: list[OperatorTime]) -> list[AggregateOperatorTime]:
    output = []
    for setting in SETTINGS:
        selected = [row for row in rows if row.setting_key == setting.key]
        if {row.query for row in selected} != set(QUERIES):
            raise ValueError(f"{setting.key}: incomplete Q1-Q22 coverage")
        for operator in OPERATOR_ORDER:
            value = sum(
                row.compute_ms
                for row in selected
                if row.operator_kind == operator
            )
            output.append(
                AggregateOperatorTime(
                    setting_key=setting.key,
                    scale_factor=setting.scale_factor,
                    gpus=setting.gpus,
                    arm=setting.arm,
                    arm_label=setting.arm_label,
                    operator_kind=operator,
                    operator_label=OPERATOR_LABELS[operator],
                    compute_ms=value,
                    compute_s=value / 1000.0,
                )
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot(rows: list[AggregateOperatorTime], output_dir: Path) -> None:
    values = {
        (row.scale_factor, row.gpus, row.arm, row.operator_kind): row.compute_s
        for row in rows
    }
    group_centers = {(1000, 8): 0.0, (3000, 24): 2.15}
    arm_offsets = {"legacy_cta64": -0.34, "synthesized_cta64": 0.34}
    width = 0.60
    positions = []
    tick_labels = []
    totals = []

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for scale, gpus in CONFIGURATIONS:
        center = group_centers[(scale, gpus)]
        for arm, arm_label in ARMS:
            position = center + arm_offsets[arm]
            bottom = 0.0
            for operator in OPERATOR_ORDER:
                value = values[(scale, gpus, arm, operator)]
                if value > 0.0:
                    ax.bar(
                        position,
                        value,
                        width=width,
                        bottom=bottom,
                        color=OPERATOR_COLORS[operator],
                        edgecolor="white",
                        linewidth=0.65,
                        zorder=3,
                    )
                bottom += value
            positions.append(position)
            tick_labels.append(arm_label)
            totals.append(bottom)

    maximum = max(totals)
    for position, total in zip(positions, totals):
        ax.text(
            position,
            total + maximum * 0.018,
            f"{total:.2f} s",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#333333",
        )

    ax.set_xticks(positions, tick_labels)
    ax.tick_params(axis="x", labelsize=10.5, pad=5)
    for (scale, gpus), center in group_centers.items():
        ax.text(
            center,
            -0.16,
            f"SF={scale}\n{gpus} GPUs",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12.5,
            color="#333333",
        )

    ax.set_ylabel("Aggregate compute-phase time (s)")
    ax.set_ylim(0.0, maximum * 1.15)
    ax.set_xlim(-0.85, 3.00)
    ax.grid(
        axis="y",
        color="#D8D8D8",
        linewidth=0.8,
        linestyle=(0, (2, 3)),
        zorder=0,
    )
    ax.axvline(1.075, color="#ECECEC", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="both", colors="#333333")

    legend_handles = [
        mpatches.Patch(
            facecolor=OPERATOR_COLORS[operator],
            edgecolor="none",
            label=OPERATOR_LABELS[operator],
        )
        for operator in OPERATOR_ORDER
    ]
    fig.suptitle("TPC-H Compute Time by Operator", y=0.98, fontsize=20)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=7,
        frameon=False,
        handlelength=1.25,
        columnspacing=1.0,
        labelspacing=0.65,
    )
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.24, top=0.70)

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
    selections = []
    operator_rows = []
    for setting in SETTINGS:
        for query in QUERIES:
            selection, rows = collect_query(results_dir, setting, query)
            selections.append(selection)
            operator_rows.extend(rows)
    aggregate_rows = aggregate(operator_rows)

    stem = "distributed_tpch_compute_operator_breakdown"
    write_csv(
        output_dir / f"{stem}_selected_samples.csv",
        [asdict(row) for row in selections],
    )
    write_csv(
        output_dir / f"{stem}_by_query.csv",
        [asdict(row) for row in operator_rows],
    )
    write_csv(
        output_dir / f"{stem}_summary.csv",
        [asdict(row) for row in aggregate_rows],
    )
    plot(aggregate_rows, output_dir)

    by_setting = defaultdict(float)
    for row in aggregate_rows:
        by_setting[row.setting_key] += row.compute_s
    for setting in SETTINGS:
        print(f"{setting.key}: {by_setting[setting.key]:.3f} s strict compute")


if __name__ == "__main__":
    main()
