#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Compare communication-stage utilization for NCCL and copy-engine arms.

The comparison covers SF=1000 on 8 GPUs and SF=3000 on 24 GPUs.  For each
arm and configuration, rank-0 Nsight Systems GPU metric samples are restricted
to communication activity inside completed ``phase:communication:*`` ranges
and ``tpch.profile``, then pooled across Q1-Q22.  Communication activity means
NCCL kernel execution for ``legacy_cta64`` and peer-to-peer transfers issued by
``cudaMemcpyBatchAsync`` for ``legacy_ce_pull``.

DRAM utilization is the sample-wise sum of read and write throughput.  NVLink
RX and TX are wire utilization: User + Protocol data, each summed across
Request + Response channels.  Temporary SQLite exports are consumed
immediately and deleted; reading existing reports requires no counter access.
"""

from __future__ import annotations

import argparse
import bisect
import os
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

import plot_distributed_tpch_compute_utilization as compute_util
import plot_distributed_tpch_profile_kernels as profile_kernels


REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES = tuple(range(1, 23))
PROFILE_RANGE = "tpch.profile"
COMMUNICATION_RANGE_PREFIX = "phase:communication:"

SM_ISSUE_METRIC = "SM Issue [Throughput %]"
DRAM_READ_METRIC = "DRAM Read Bandwidth [Throughput %]"
DRAM_WRITE_METRIC = "DRAM Write Bandwidth [Throughput %]"
NVLINK_RX_SOURCE_METRICS = (
    "NVLink RX Requests Protocol Data [Throughput %]",
    "NVLink RX Requests User Data [Throughput %]",
    "NVLink RX Responses Protocol Data [Throughput %]",
    "NVLink RX Responses User Data [Throughput %]",
)
NVLINK_TX_SOURCE_METRICS = (
    "NVLink TX Requests Protocol Data [Throughput %]",
    "NVLink TX Requests User Data [Throughput %]",
    "NVLink TX Responses Protocol Data [Throughput %]",
    "NVLink TX Responses User Data [Throughput %]",
)
SOURCE_METRIC_NAMES = (
    SM_ISSUE_METRIC,
    DRAM_READ_METRIC,
    DRAM_WRITE_METRIC,
    *NVLINK_RX_SOURCE_METRICS,
    *NVLINK_TX_SOURCE_METRICS,
)
DRAM_COMBINED_METRIC = "DRAM Read + Write Bandwidth [Throughput %]"
NVLINK_TX_WIRE_METRIC = "NVLink TX Wire [Throughput %]"
NVLINK_RX_WIRE_METRIC = "NVLink RX Wire [Throughput %]"
METRICS = (
    (SM_ISSUE_METRIC, "SM issue"),
    (DRAM_COMBINED_METRIC, "DRAM read + write"),
    (NVLINK_TX_WIRE_METRIC, "NVLink TX"),
    (NVLINK_RX_WIRE_METRIC, "NVLink RX"),
)
METRIC_NAMES = tuple(name for name, _ in METRICS)

ARMS = (
    ("legacy_cta64", "NCCL", "#4C78A8"),
    ("legacy_ce_pull", "Copy engine", "#F58518"),
)
SETTINGS = tuple(
    profile_kernels.Setting(
        scale_factor=scale_factor,
        gpus=gpus,
        arm=arm,
        label=label,
        color=color,
    )
    for scale_factor, gpus in ((1000, 8), (3000, 24))
    for arm, label, color in ARMS
)


@dataclass(frozen=True)
class QueryCommunicationUtilization:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    query: int
    rank: int
    profile_range_ms: float
    communication_stage_interval_count: int
    communication_stage_duration_ms: float
    activity_filter: str
    activity_event_count: int
    communication_interval_count: int
    communication_duration_ms: float
    metric_name: str
    sample_count: int
    sample_sum_percent: float
    mean_utilization_percent: float | None
    min_utilization_percent: float | None
    max_utilization_percent: float | None
    result_status: str
    failure_kind: str
    report: str


@dataclass(frozen=True)
class CommunicationUtilizationSample:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    query: int
    rank: int
    metric_name: str
    sample_index: int
    timestamp_from_profile_start_ms: float
    utilization_percent: float


@dataclass
class ExtractedQuery:
    summaries: list[QueryCommunicationUtilization]
    samples: list[CommunicationUtilizationSample]


@dataclass(frozen=True)
class CampaignCommunicationUtilization:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    metric_name: str
    query_count: int
    sample_count: int
    sample_sum_percent: float
    mean_utilization_percent: float


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
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Concurrent nsys exports (default: up to 4)",
    )
    parser.add_argument(
        "--export-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for one nsys export",
    )
    return parser.parse_args()


def intersect_interval_unions(
    first: list[tuple[int, int]], second: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    first = compute_util.merge_intervals(first)
    second = compute_util.merge_intervals(second)
    intersections = []
    first_index = 0
    second_index = 0
    while first_index < len(first) and second_index < len(second):
        first_start, first_end = first[first_index]
        second_start, second_end = second[second_index]
        start = max(first_start, second_start)
        end = min(first_end, second_end)
        if start < end:
            intersections.append((start, end))
        if first_end <= second_end:
            first_index += 1
        else:
            second_index += 1
    return compute_util.merge_intervals(intersections)


def extract_sqlite(
    selected: profile_kernels.SelectedReport,
    sqlite_path: Path,
    results_dir: Path,
) -> ExtractedQuery:
    connection = sqlite3.connect(sqlite_path)
    try:
        profile_ranges = connection.execute(
            """
            SELECT e.start, e.end
            FROM NVTX_EVENTS e
            LEFT JOIN StringIds s ON s.id = e.textId
            WHERE coalesce(e.text, s.value) = ? AND e.end IS NOT NULL
            ORDER BY e.start
            """,
            (PROFILE_RANGE,),
        ).fetchall()
        if len(profile_ranges) != 1:
            raise RuntimeError(
                f"{selected.report}: expected one completed {PROFILE_RANGE} "
                f"range, found {len(profile_ranges)}"
            )
        profile_start, profile_end = map(int, profile_ranges[0])
        if profile_start >= profile_end:
            raise RuntimeError(f"{selected.report}: invalid profile range")

        communication_ranges = connection.execute(
            """
            SELECT e.start, e.end
            FROM NVTX_EVENTS e
            LEFT JOIN StringIds s ON s.id = e.textId
            WHERE coalesce(e.text, s.value) LIKE ?
              AND e.end IS NOT NULL
              AND e.start >= ? AND e.end <= ?
            ORDER BY e.start, e.end
            """,
            (f"{COMMUNICATION_RANGE_PREFIX}%", profile_start, profile_end),
        ).fetchall()
        stage_intervals = compute_util.merge_intervals(
            [(int(start), int(end)) for start, end in communication_ranges]
        )
        if not stage_intervals:
            raise RuntimeError(
                f"{selected.report}: no completed "
                f"{COMMUNICATION_RANGE_PREFIX}* ranges inside {PROFILE_RANGE}"
            )

        if selected.setting.arm == "legacy_cta64":
            activity_filter = "NCCL kernels"
            activity_ranges = connection.execute(
                """
                SELECT k.start, k.end
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON s.id = k.shortName
                WHERE lower(s.value) LIKE '%nccl%'
                  AND k.start >= ? AND k.end <= ? AND k.end > k.start
                ORDER BY k.start, k.end
                """,
                (profile_start, profile_end),
            ).fetchall()
        elif selected.setting.arm == "legacy_ce_pull":
            activity_filter = "cudaMemcpyBatchAsync peer-to-peer copies"
            activity_ranges = connection.execute(
                """
                SELECT m.start, m.end
                FROM CUPTI_ACTIVITY_KIND_MEMCPY m
                JOIN ENUM_CUDA_MEMCPY_OPER op ON op.id = m.copyKind
                JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                  ON r.correlationId = m.correlationId
                JOIN StringIds s ON s.id = r.nameId
                WHERE op.name = 'CUDA_MEMCPY_KIND_PTOP'
                  AND s.value LIKE 'cudaMemcpyBatchAsync%'
                  AND m.start >= ? AND m.end <= ? AND m.end > m.start
                ORDER BY m.start, m.end
                """,
                (profile_start, profile_end),
            ).fetchall()
        else:
            raise RuntimeError(
                f"unsupported communication arm: {selected.setting.arm}"
            )
        activity_ranges = [
            (int(start), int(end)) for start, end in activity_ranges
        ]
        activity_event_count = sum(
            any(
                max(event_start, stage_start) < min(event_end, stage_end)
                for stage_start, stage_end in stage_intervals
            )
            for event_start, event_end in activity_ranges
        )
        intervals = intersect_interval_unions(
            stage_intervals, activity_ranges
        )
        if bool(intervals) != (activity_event_count > 0):
            raise RuntimeError(
                f"{selected.report}: inconsistent {activity_filter} event and "
                "interval coverage"
            )

        placeholders = ", ".join("?" for _ in SOURCE_METRIC_NAMES)
        definitions = connection.execute(
            f"""
            SELECT metricName, count(*)
            FROM TARGET_INFO_GPU_METRICS
            WHERE metricName IN ({placeholders})
            GROUP BY metricName
            """,
            SOURCE_METRIC_NAMES,
        ).fetchall()
        definition_counts = {str(name): int(count) for name, count in definitions}
        if definition_counts != {name: 1 for name in SOURCE_METRIC_NAMES}:
            raise RuntimeError(
                f"{selected.report}: unexpected GPU metric definitions: "
                f"{definition_counts}"
            )

        samples = connection.execute(
            f"""
            SELECT d.metricName, g.timestamp, g.value
            FROM GPU_METRICS g
            JOIN TARGET_INFO_GPU_METRICS d
              ON d.typeId = g.typeId AND d.metricId = g.metricId
            WHERE d.metricName IN ({placeholders})
              AND g.timestamp >= ? AND g.timestamp < ?
            ORDER BY g.timestamp, d.metricName
            """,
            (*SOURCE_METRIC_NAMES, profile_start, profile_end),
        ).fetchall()
    finally:
        connection.close()

    starts = [start for start, _ in intervals]
    ends = [end for _, end in intervals]
    samples_by_metric: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for metric_name, timestamp, value in samples:
        timestamp = int(timestamp)
        interval_index = bisect.bisect_right(starts, timestamp) - 1
        if interval_index < 0 or timestamp >= ends[interval_index]:
            continue
        numeric_value = float(value)
        if not 0.0 <= numeric_value <= 100.0:
            raise RuntimeError(
                f"{selected.report}: {metric_name} sample outside [0, 100]: "
                f"{numeric_value}"
            )
        samples_by_metric[str(metric_name)].append((timestamp, numeric_value))

    counts = {name: len(samples_by_metric[name]) for name in SOURCE_METRIC_NAMES}
    if len(set(counts.values())) != 1:
        raise RuntimeError(
            f"{selected.report}: unequal communication-activity sample counts: "
            f"{counts}"
        )
    reference_timestamps = [
        timestamp for timestamp, _ in samples_by_metric[SM_ISSUE_METRIC]
    ]
    for metric_name in SOURCE_METRIC_NAMES[1:]:
        timestamps = [timestamp for timestamp, _ in samples_by_metric[metric_name]]
        if timestamps != reference_timestamps:
            raise RuntimeError(
                f"{selected.report}: GPU metric timestamps are not aligned for "
                f"{metric_name}"
            )

    source_values = {
        name: [value for _, value in samples_by_metric[name]]
        for name in SOURCE_METRIC_NAMES
    }
    sample_count = len(reference_timestamps)
    output_values = {
        SM_ISSUE_METRIC: source_values[SM_ISSUE_METRIC],
        DRAM_COMBINED_METRIC: [
            source_values[DRAM_READ_METRIC][index]
            + source_values[DRAM_WRITE_METRIC][index]
            for index in range(sample_count)
        ],
        NVLINK_TX_WIRE_METRIC: [
            sum(source_values[name][index] for name in NVLINK_TX_SOURCE_METRICS)
            for index in range(sample_count)
        ],
        NVLINK_RX_WIRE_METRIC: [
            sum(source_values[name][index] for name in NVLINK_RX_SOURCE_METRICS)
            for index in range(sample_count)
        ],
    }
    for metric_name, values in output_values.items():
        if values and (min(values) < 0.0 or max(values) > 100.0):
            raise RuntimeError(
                f"{selected.report}: derived {metric_name} samples outside "
                f"[0, 100]: min={min(values)}, max={max(values)}"
            )

    setting = selected.setting
    rows = []
    sample_rows = []
    for metric_name in METRIC_NAMES:
        values = output_values[metric_name]
        sample_sum = sum(values)
        rows.append(
            QueryCommunicationUtilization(
                setting_key=setting.key,
                scale_factor=setting.scale_factor,
                gpus=setting.gpus,
                arm=setting.arm,
                query=selected.query,
                rank=0,
                profile_range_ms=(profile_end - profile_start) / 1e6,
                communication_stage_interval_count=len(stage_intervals),
                communication_stage_duration_ms=sum(
                    end - start for start, end in stage_intervals
                )
                / 1e6,
                activity_filter=activity_filter,
                activity_event_count=activity_event_count,
                communication_interval_count=len(intervals),
                communication_duration_ms=sum(
                    end - start for start, end in intervals
                )
                / 1e6,
                metric_name=metric_name,
                sample_count=len(values),
                sample_sum_percent=sample_sum,
                mean_utilization_percent=(
                    sample_sum / len(values) if values else None
                ),
                min_utilization_percent=min(values) if values else None,
                max_utilization_percent=max(values) if values else None,
                result_status=selected.result_status,
                failure_kind=selected.failure_kind,
                report=str(selected.report.relative_to(results_dir)),
            )
        )
        sample_rows.extend(
            CommunicationUtilizationSample(
                setting_key=setting.key,
                scale_factor=setting.scale_factor,
                gpus=setting.gpus,
                arm=setting.arm,
                query=selected.query,
                rank=0,
                metric_name=metric_name,
                sample_index=sample_index,
                timestamp_from_profile_start_ms=(timestamp - profile_start)
                / 1e6,
                utilization_percent=value,
            )
            for sample_index, (timestamp, value) in enumerate(
                zip(reference_timestamps, values, strict=True), start=1
            )
        )
    return ExtractedQuery(summaries=rows, samples=sample_rows)


def process_report(
    nsys: str,
    selected: profile_kernels.SelectedReport,
    timeout: float,
    results_dir: Path,
) -> ExtractedQuery:
    with tempfile.TemporaryDirectory(prefix="tpch_nsys_comm_util_") as directory:
        sqlite_path = Path(directory) / "report.sqlite"
        profile_kernels.export_sqlite(nsys, selected.report, sqlite_path, timeout)
        return extract_sqlite(selected, sqlite_path, results_dir)


def summarize_campaigns(
    query_rows: list[QueryCommunicationUtilization],
) -> list[CampaignCommunicationUtilization]:
    summaries = []
    for setting in SETTINGS:
        setting_rows = [row for row in query_rows if row.setting_key == setting.key]
        if {row.query for row in setting_rows} != set(QUERIES):
            raise ValueError(f"{setting.key}: incomplete Q1-Q22 coverage")
        for metric_name in METRIC_NAMES:
            metric_rows = [
                row for row in setting_rows if row.metric_name == metric_name
            ]
            if len(metric_rows) != len(QUERIES):
                raise ValueError(
                    f"{setting.key}, {metric_name}: expected {len(QUERIES)} "
                    f"query rows, found {len(metric_rows)}"
                )
            sample_count = sum(row.sample_count for row in metric_rows)
            sample_sum = sum(row.sample_sum_percent for row in metric_rows)
            if sample_count <= 0:
                raise ValueError(f"{setting.key}, {metric_name}: no samples")
            summaries.append(
                CampaignCommunicationUtilization(
                    setting_key=setting.key,
                    scale_factor=setting.scale_factor,
                    gpus=setting.gpus,
                    arm=setting.arm,
                    metric_name=metric_name,
                    query_count=len(metric_rows),
                    sample_count=sample_count,
                    sample_sum_percent=sample_sum,
                    mean_utilization_percent=sample_sum / sample_count,
                )
            )
    return summaries


def plot(
    rows: list[CampaignCommunicationUtilization],
    samples: list[CommunicationUtilizationSample],
    output_dir: Path,
) -> None:
    lookup = {
        (row.scale_factor, row.arm, row.metric_name): row for row in rows
    }
    group_centers = {
        (1000, metric_name): index * 1.35
        for index, (metric_name, _) in enumerate(METRICS)
    }
    group_centers.update(
        {
            (3000, metric_name): 5.75 + index * 1.35
            for index, (metric_name, _) in enumerate(METRICS)
        }
    )
    violin_width = 0.42
    offsets = (-violin_width / 2.0, violin_width / 2.0)
    sample_lookup: dict[tuple[int, str, str], list[float]] = {}
    for sample in samples:
        key = (sample.scale_factor, sample.arm, sample.metric_name)
        sample_lookup.setdefault(key, []).append(sample.utilization_percent)
    for key, row in lookup.items():
        values = sample_lookup[key]
        if len(values) != row.sample_count:
            raise ValueError(
                f"{key}: raw sample count {len(values)} != {row.sample_count}"
            )
        sample_mean = float(np.mean(values))
        if not np.isclose(sample_mean, row.mean_utilization_percent):
            raise ValueError(
                f"{key}: raw mean {sample_mean} != "
                f"{row.mean_utilization_percent}"
            )

    fig, ax = plt.subplots(figsize=(12.8, 6.3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for scale_factor in (1000, 3000):
        for metric_name, _ in METRICS:
            center = group_centers[(scale_factor, metric_name)]
            for (arm, label, color), offset in zip(ARMS, offsets, strict=True):
                row = lookup[(scale_factor, arm, metric_name)]
                position = center + offset
                values = np.asarray(
                    sample_lookup[(scale_factor, arm, metric_name)],
                    dtype=float,
                )
                parts = ax.violinplot(
                    [values],
                    positions=[position],
                    widths=violin_width * 0.88,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                    points=160,
                    bw_method="scott",
                )
                body = parts["bodies"][0]
                body.set_facecolor(color)
                body.set_edgecolor("#555555")
                body.set_linewidth(0.65)
                body.set_alpha(0.86)

                q1, median, q3 = np.percentile(values, [25, 50, 75])
                mean = row.mean_utilization_percent
                ax.vlines(
                    position,
                    q1,
                    q3,
                    color="#303030",
                    linewidth=2.2,
                    zorder=5,
                )
                ax.scatter(
                    [position],
                    [median],
                    marker="o",
                    s=27,
                    facecolor="white",
                    edgecolor="#303030",
                    linewidth=0.8,
                    zorder=6,
                )
                ax.scatter(
                    [position],
                    [mean],
                    marker="D",
                    s=24,
                    facecolor="#303030",
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=7,
                )
                ax.annotate(
                    f"{mean:.1f}",
                    xy=(position, mean),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#333333",
                    zorder=8,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.88,
                    },
                )

    centers = [
        group_centers[(scale_factor, metric_name)]
        for scale_factor in (1000, 3000)
        for metric_name, _ in METRICS
    ]
    labels = [label for _ in (1000, 3000) for _, label in METRICS]
    ax.set_xticks(centers, labels)
    ax.text(
        2.025,
        -0.13,
        "SF=1000, 8 GPUs",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=14,
        color="#444444",
    )
    ax.text(
        7.775,
        -0.13,
        "SF=3000, 24 GPUs",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=14,
        color="#444444",
    )
    ax.axvline(4.9, color="#E2E2E2", linewidth=1.1, zorder=1)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Utilization")
    ax.set_title("TPC-H Communication-Stage GPU Utilization", pad=64)
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

    handles = [Patch(facecolor=color, label=label) for _, label, color in ARMS]
    handles.extend(
        (
            Line2D(
                [],
                [],
                marker="D",
                linestyle="None",
                markersize=6,
                markerfacecolor="#303030",
                markeredgecolor="white",
                label="Mean",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor="#303030",
                label="Median",
            ),
            Line2D([], [], color="#303030", linewidth=2.2, label="IQR"),
        )
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.77)

    stem = "distributed_tpch_communication_stage_utilization_arm_comparison"
    for suffix in ("pdf", "png"):
        directory = output_dir / suffix
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError("nsys is required to export .nsys-rep files")
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()

    selected_reports = [
        profile_kernels.select_rank_zero_report(results_dir, setting, query)
        for setting in SETTINGS
        for query in QUERIES
    ]
    expected_reports = len(SETTINGS) * len(QUERIES)
    if len(selected_reports) != expected_reports:
        raise ValueError(
            f"expected {expected_reports} reports, found {len(selected_reports)}"
        )
    print(
        f"processing {len(selected_reports)} rank-0 reports with "
        f"{args.jobs} concurrent exports"
    )

    query_rows: list[QueryCommunicationUtilization] = []
    sample_rows: list[CommunicationUtilizationSample] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_report,
                nsys,
                selected,
                args.export_timeout,
                results_dir,
            ): selected
            for selected in selected_reports
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            selected = futures[future]
            try:
                extracted = future.result()
                query_rows.extend(extracted.summaries)
                sample_rows.extend(extracted.samples)
            except Exception as error:
                raise RuntimeError(
                    f"failed to process {selected.report}: {error}"
                ) from error
            if completed % 10 == 0 or completed == len(futures):
                print(f"processed {completed}/{len(futures)} reports")

    query_rows.sort(
        key=lambda row: (
            row.scale_factor,
            row.gpus,
            row.arm,
            row.query,
            METRIC_NAMES.index(row.metric_name),
        )
    )
    if len(query_rows) != expected_reports * len(METRICS):
        raise ValueError("unexpected query-metric row count")
    sample_rows.sort(
        key=lambda row: (
            row.scale_factor,
            row.gpus,
            row.arm,
            row.query,
            METRIC_NAMES.index(row.metric_name),
            row.sample_index,
        )
    )
    campaign_rows = summarize_campaigns(query_rows)
    compute_util.write_csv(
        output_dir
        / "distributed_tpch_communication_stage_utilization_arm_comparison_by_query.csv",
        [asdict(row) for row in query_rows],
    )
    compute_util.write_csv(
        output_dir
        / "distributed_tpch_communication_stage_utilization_arm_comparison_summary.csv",
        [asdict(row) for row in campaign_rows],
    )
    compute_util.write_csv(
        output_dir
        / "distributed_tpch_communication_stage_utilization_arm_comparison_samples.csv",
        [asdict(row) for row in sample_rows],
    )
    plot(campaign_rows, sample_rows, output_dir)


if __name__ == "__main__":
    main()
