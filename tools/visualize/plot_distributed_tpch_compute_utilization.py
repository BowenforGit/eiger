#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot compute-stage GPU-utilization distributions for TPC-H campaigns.

For each available ``legacy_cta64`` setting, this tool exports the rank-0
Nsight Systems report for Q1-Q22 and selects GPU metric samples that fall in
the union of completed ``phase:compute:*`` NVTX ranges inside the completed
``tpch.profile`` range.  Samples are pooled across all queries, making both
the violin density and arithmetic mean time-weighted because the reports use
a uniform 10 kHz sampling cadence.

DRAM utilization is the sample-wise sum of the read- and write-bandwidth
throughput percentages, followed by the same pooled-sample aggregation.

Hash partitioning is included because its NVTX range is named
``phase:compute:shuffle:shuffle.hash_partition``.  Temporary SQLite exports
are consumed immediately and deleted.  Existing metric samples can be read
without GPU performance-counter access.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
except ModuleNotFoundError as error:
    sys.exit(f"error: matplotlib and numpy are required: {error}")

import plot_distributed_tpch_profile_kernels as profile_kernels


REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES = tuple(range(1, 23))
PROFILE_RANGE = "tpch.profile"
COMPUTE_RANGE_PREFIX = "phase:compute:"
SM_ISSUE_METRIC = "SM Issue [Throughput %]"
DRAM_READ_METRIC = "DRAM Read Bandwidth [Throughput %]"
DRAM_WRITE_METRIC = "DRAM Write Bandwidth [Throughput %]"
DRAM_COMBINED_METRIC = "DRAM Read + Write Bandwidth [Throughput %]"
SOURCE_METRIC_NAMES = (SM_ISSUE_METRIC, DRAM_READ_METRIC, DRAM_WRITE_METRIC)
METRICS = (
    (SM_ISSUE_METRIC, "SM issue"),
    (DRAM_COMBINED_METRIC, "DRAM read + write"),
)
METRIC_NAMES = tuple(name for name, _ in METRICS)
GPU_COLORS = {
    8: "#4C78A8",
    16: "#F58518",
    24: "#54A24B",
    32: "#E45756",
}
SETTINGS = tuple(
    profile_kernels.Setting(
        scale_factor=scale_factor,
        gpus=gpus,
        arm="legacy_cta64",
        label=f"{gpus} GPUs",
        color=GPU_COLORS[gpus],
    )
    for scale_factor, gpu_counts in ((1000, (8, 16, 24, 32)), (3000, (24, 32)))
    for gpus in gpu_counts
)


@dataclass(frozen=True)
class QueryUtilization:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    query: int
    rank: int
    profile_range_ms: float
    compute_interval_count: int
    compute_duration_ms: float
    metric_name: str
    sample_count: int
    sample_sum_percent: float
    mean_utilization_percent: float
    min_utilization_percent: float
    max_utilization_percent: float
    result_status: str
    failure_kind: str
    report: str


@dataclass(frozen=True)
class UtilizationSample:
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
    summaries: list[QueryUtilization]
    samples: list[UtilizationSample]


@dataclass(frozen=True)
class CampaignUtilization:
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


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


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

        compute_ranges = connection.execute(
            """
            SELECT e.start, e.end
            FROM NVTX_EVENTS e
            LEFT JOIN StringIds s ON s.id = e.textId
            WHERE coalesce(e.text, s.value) LIKE ?
              AND e.end IS NOT NULL
              AND e.start >= ? AND e.end <= ?
            ORDER BY e.start, e.end
            """,
            (f"{COMPUTE_RANGE_PREFIX}%", profile_start, profile_end),
        ).fetchall()
        intervals = merge_intervals(
            [(int(start), int(end)) for start, end in compute_ranges]
        )
        if not intervals:
            raise RuntimeError(
                f"{selected.report}: no completed {COMPUTE_RANGE_PREFIX}* "
                f"ranges inside {PROFILE_RANGE}"
            )

        definitions = connection.execute(
            """
            SELECT metricName, count(*)
            FROM TARGET_INFO_GPU_METRICS
            WHERE metricName IN (?, ?, ?)
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
            """
            SELECT d.metricName, g.timestamp, g.value
            FROM GPU_METRICS g
            JOIN TARGET_INFO_GPU_METRICS d
              ON d.typeId = g.typeId AND d.metricId = g.metricId
            WHERE d.metricName IN (?, ?, ?)
              AND g.timestamp >= ? AND g.timestamp < ?
            ORDER BY g.timestamp, d.metricName
            """,
            (*SOURCE_METRIC_NAMES, profile_start, profile_end),
        ).fetchall()
    finally:
        connection.close()

    starts = [start for start, _ in intervals]
    ends = [end for _, end in intervals]
    values_by_metric: dict[str, list[tuple[int, float]]] = defaultdict(list)
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
        values_by_metric[str(metric_name)].append((timestamp, numeric_value))

    counts = {name: len(values_by_metric[name]) for name in SOURCE_METRIC_NAMES}
    if len(set(counts.values())) != 1 or next(iter(counts.values())) == 0:
        raise RuntimeError(
            f"{selected.report}: unequal or empty compute-stage sample counts: "
            f"{counts}"
        )

    timestamps_by_metric = {
        name: [timestamp for timestamp, _ in values_by_metric[name]]
        for name in SOURCE_METRIC_NAMES
    }
    if len({tuple(values) for values in timestamps_by_metric.values()}) != 1:
        raise RuntimeError(
            f"{selected.report}: GPU metric timestamps are not aligned"
        )
    timestamps = timestamps_by_metric[SM_ISSUE_METRIC]
    numeric_values = {
        name: [value for _, value in values_by_metric[name]]
        for name in SOURCE_METRIC_NAMES
    }

    combined_dram_values = [
        read + write
        for read, write in zip(
            numeric_values[DRAM_READ_METRIC],
            numeric_values[DRAM_WRITE_METRIC],
            strict=True,
        )
    ]
    if max(combined_dram_values) > 100.0:
        raise RuntimeError(
            f"{selected.report}: combined DRAM sample outside [0, 100]: "
            f"{max(combined_dram_values)}"
        )
    output_values = {
        SM_ISSUE_METRIC: numeric_values[SM_ISSUE_METRIC],
        DRAM_COMBINED_METRIC: combined_dram_values,
    }

    setting = selected.setting
    rows = []
    sample_rows = []
    for metric_name in METRIC_NAMES:
        values = output_values[metric_name]
        sample_sum = sum(values)
        rows.append(
            QueryUtilization(
                setting_key=setting.key,
                scale_factor=setting.scale_factor,
                gpus=setting.gpus,
                arm=setting.arm,
                query=selected.query,
                rank=0,
                profile_range_ms=(profile_end - profile_start) / 1e6,
                compute_interval_count=len(intervals),
                compute_duration_ms=sum(end - start for start, end in intervals)
                / 1e6,
                metric_name=metric_name,
                sample_count=len(values),
                sample_sum_percent=sample_sum,
                mean_utilization_percent=sample_sum / len(values),
                min_utilization_percent=min(values),
                max_utilization_percent=max(values),
                result_status=selected.result_status,
                failure_kind=selected.failure_kind,
                report=str(selected.report.relative_to(results_dir)),
            )
        )
        sample_rows.extend(
            UtilizationSample(
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
                zip(timestamps, values, strict=True), start=1
            )
        )
    return ExtractedQuery(summaries=rows, samples=sample_rows)


def process_report(
    nsys: str,
    selected: profile_kernels.SelectedReport,
    timeout: float,
    results_dir: Path,
) -> ExtractedQuery:
    with tempfile.TemporaryDirectory(prefix="tpch_nsys_compute_util_") as directory:
        sqlite_path = Path(directory) / "report.sqlite"
        profile_kernels.export_sqlite(nsys, selected.report, sqlite_path, timeout)
        return extract_sqlite(selected, sqlite_path, results_dir)


def summarize_campaigns(
    query_rows: list[QueryUtilization],
) -> list[CampaignUtilization]:
    rows = []
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
            rows.append(
                CampaignUtilization(
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
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot(
    rows: list[CampaignUtilization],
    samples: list[UtilizationSample],
    output_dir: Path,
) -> None:
    lookup = {
        (row.scale_factor, row.gpus, row.metric_name): row for row in rows
    }
    group_centers = {
        (1000, METRICS[0][0]): 0.0,
        (1000, METRICS[1][0]): 1.7,
        (3000, METRICS[0][0]): 3.5,
        (3000, METRICS[1][0]): 4.6,
    }
    gpu_counts = {1000: (8, 16, 24, 32), 3000: (24, 32)}
    violin_width = {1000: 0.36, 3000: 0.42}
    sample_lookup: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for sample in samples:
        sample_lookup[
            (sample.scale_factor, sample.gpus, sample.metric_name)
        ].append(sample.utilization_percent)
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

    fig, ax = plt.subplots(figsize=(9.2, 6.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for scale_factor in (1000, 3000):
        gpus_for_scale = gpu_counts[scale_factor]
        width = violin_width[scale_factor]
        offsets = [
            (index - (len(gpus_for_scale) - 1) / 2.0) * width
            for index in range(len(gpus_for_scale))
        ]
        for metric_name, _ in METRICS:
            center = group_centers[(scale_factor, metric_name)]
            for gpus, offset in zip(gpus_for_scale, offsets):
                row = lookup[(scale_factor, gpus, metric_name)]
                position = center + offset
                values = np.asarray(
                    sample_lookup[(scale_factor, gpus, metric_name)],
                    dtype=float,
                )
                parts = ax.violinplot(
                    [values],
                    positions=[position],
                    widths=width * 0.82,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                    points=160,
                    bw_method="scott",
                )
                body = parts["bodies"][0]
                body.set_facecolor(GPU_COLORS[gpus])
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
                    color="#303030",
                    zorder=8,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.88,
                    },
                )

    centers = [group_centers[(sf, name)] for sf in (1000, 3000) for name, _ in METRICS]
    labels = [label for _ in (1000, 3000) for _, label in METRICS]
    ax.set_xticks(centers, labels)
    ax.text(
        0.85,
        -0.13,
        "SF=1000",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=14,
        color="#444444",
    )
    ax.text(
        4.05,
        -0.13,
        "SF=3000",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=14,
        color="#444444",
    )
    ax.axvline(2.75, color="#E2E2E2", linewidth=1.1, zorder=1)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Utilization")
    ax.set_title("TPC-H Compute-Stage GPU Utilization", pad=78)
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

    gpu_handles = {
        gpus: Patch(facecolor=GPU_COLORS[gpus], label=f"{gpus} GPUs")
        for gpus in (8, 16, 24, 32)
    }
    mean_handle = Line2D(
        [],
        [],
        marker="D",
        linestyle="None",
        markersize=6,
        markerfacecolor="#303030",
        markeredgecolor="white",
        label="Mean",
    )
    median_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="#303030",
        label="Median",
    )
    iqr_handle = Line2D(
        [], [], color="#303030", linewidth=2.2, label="IQR"
    )
    # Matplotlib fills multi-row legends column-first.  Interleave the input
    # so the rendered first row is 8/16/24/32 GPUs and the second is the
    # mean/median/IQR key.
    handles = [
        gpu_handles[8],
        mean_handle,
        gpu_handles[16],
        median_handle,
        gpu_handles[24],
        iqr_handle,
        gpu_handles[32],
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.65,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.75)

    stem = "distributed_tpch_compute_stage_utilization_legacy_cta64"
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

    query_rows: list[QueryUtilization] = []
    sample_rows: list[UtilizationSample] = []
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
            row.query,
            METRIC_NAMES.index(row.metric_name),
            row.sample_index,
        )
    )
    campaign_rows = summarize_campaigns(query_rows)
    write_csv(
        output_dir / "distributed_tpch_compute_stage_utilization_by_query.csv",
        [asdict(row) for row in query_rows],
    )
    write_csv(
        output_dir / "distributed_tpch_compute_stage_utilization_summary.csv",
        [asdict(row) for row in campaign_rows],
    )
    write_csv(
        output_dir / "distributed_tpch_compute_stage_utilization_samples.csv",
        [asdict(row) for row in sample_rows],
    )
    plot(campaign_rows, sample_rows, output_dir)


if __name__ == "__main__":
    main()
