#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Compare compute-stage utilization for baseline and optimized compute arms.

The comparison is fixed to SF=1000 on 8 GPUs and SF=3000 on 24 GPUs.  It
reuses the rank-0 Nsight Systems extraction used by the GPU-count figure:
samples are restricted to the union of completed ``phase:compute:*`` ranges
inside ``tpch.profile`` and pooled across Q1-Q22.  DRAM utilization is the
sample-wise sum of read- and write-bandwidth throughput percentages.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
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
ARMS = (
    ("legacy_cta64", "Baseline", "#4C78A8"),
    ("synthesized_cta64", "Opt. comp", "#54A24B"),
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


def summarize_campaigns(
    query_rows: list[compute_util.QueryUtilization],
) -> list[compute_util.CampaignUtilization]:
    summaries = []
    for setting in SETTINGS:
        setting_rows = [row for row in query_rows if row.setting_key == setting.key]
        if {row.query for row in setting_rows} != set(QUERIES):
            raise ValueError(f"{setting.key}: incomplete Q1-Q22 coverage")
        for metric_name in compute_util.METRIC_NAMES:
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
                compute_util.CampaignUtilization(
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
    rows: list[compute_util.CampaignUtilization],
    samples: list[compute_util.UtilizationSample],
    output_dir: Path,
) -> None:
    lookup = {
        (row.scale_factor, row.arm, row.metric_name): row for row in rows
    }
    group_centers = {
        (1000, compute_util.METRICS[0][0]): 0.0,
        (1000, compute_util.METRICS[1][0]): 1.3,
        (3000, compute_util.METRICS[0][0]): 3.0,
        (3000, compute_util.METRICS[1][0]): 4.3,
    }
    violin_width = 0.37
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

    fig, ax = plt.subplots(figsize=(8.8, 6.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for scale_factor in (1000, 3000):
        for metric_name, _ in compute_util.METRICS:
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
        for metric_name, _ in compute_util.METRICS
    ]
    labels = [
        label
        for _ in (1000, 3000)
        for _, label in compute_util.METRICS
    ]
    ax.set_xticks(centers, labels)
    ax.text(
        0.65,
        -0.13,
        "SF=1000, 8 GPUs",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=14,
        color="#444444",
    )
    ax.text(
        3.65,
        -0.13,
        "SF=3000, 24 GPUs",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=14,
        color="#444444",
    )
    ax.axvline(2.15, color="#E2E2E2", linewidth=1.1, zorder=1)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Utilization")
    ax.set_title("TPC-H Compute-Stage GPU Utilization", pad=64)
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
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.19, top=0.77)

    stem = "distributed_tpch_compute_stage_utilization_arm_comparison"
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

    query_rows: list[compute_util.QueryUtilization] = []
    sample_rows: list[compute_util.UtilizationSample] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                compute_util.process_report,
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
            compute_util.METRIC_NAMES.index(row.metric_name),
        )
    )
    if len(query_rows) != expected_reports * len(compute_util.METRICS):
        raise ValueError("unexpected query-metric row count")
    sample_rows.sort(
        key=lambda row: (
            row.scale_factor,
            row.gpus,
            row.arm,
            row.query,
            compute_util.METRIC_NAMES.index(row.metric_name),
            row.sample_index,
        )
    )
    campaign_rows = summarize_campaigns(query_rows)
    compute_util.write_csv(
        output_dir
        / "distributed_tpch_compute_stage_utilization_arm_comparison_by_query.csv",
        [asdict(row) for row in query_rows],
    )
    compute_util.write_csv(
        output_dir
        / "distributed_tpch_compute_stage_utilization_arm_comparison_summary.csv",
        [asdict(row) for row in campaign_rows],
    )
    compute_util.write_csv(
        output_dir
        / "distributed_tpch_compute_stage_utilization_arm_comparison_samples.csv",
        [asdict(row) for row in sample_rows],
    )
    plot(campaign_rows, sample_rows, output_dir)


if __name__ == "__main__":
    main()
