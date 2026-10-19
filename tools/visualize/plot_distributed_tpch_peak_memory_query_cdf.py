#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot query-level peak-memory CDFs for four distributed TPC-H settings."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from matplotlib.ticker import PercentFormatter

import plot_distributed_tpch_peak_memory_arm_comparison as arm_memory


REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES = tuple(range(1, 23))
LINE_STYLES = {1000: "-", 3000: (0, (5, 2))}
MARKERS = {"legacy_cta64": "o", "synthesized_cta64": "s"}


@dataclass(frozen=True)
class CdfPoint:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    arm_label: str
    series_label: str
    cdf_order: int
    query: int
    peak_rank: int
    total_live_peak_bytes: int
    total_live_peak_gib: float
    cumulative_query_count: int
    cumulative_query_fraction: float


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


def series_label(setting: arm_memory.Setting) -> str:
    return (
        f"SF={setting.scale_factor}, {setting.gpus} GPUs — {setting.label}"
    )


def build_cdf_points(
    setting: arm_memory.Setting,
    query_rows: list[arm_memory.QueryPeak],
) -> list[CdfPoint]:
    if {row.query for row in query_rows} != set(QUERIES):
        raise ValueError(f"{setting.key}: incomplete Q1-Q22 coverage")
    ordered = sorted(
        query_rows,
        key=lambda row: (row.total_live_peak_bytes, row.query),
    )
    count = len(ordered)
    label = series_label(setting)
    return [
        CdfPoint(
            setting_key=setting.key,
            scale_factor=setting.scale_factor,
            gpus=setting.gpus,
            arm=setting.arm,
            arm_label=setting.label,
            series_label=label,
            cdf_order=index,
            query=row.query,
            peak_rank=row.peak_rank,
            total_live_peak_bytes=row.total_live_peak_bytes,
            total_live_peak_gib=row.total_live_peak_gib,
            cumulative_query_count=index,
            cumulative_query_fraction=index / count,
        )
        for index, row in enumerate(ordered, start=1)
    ]


def plot(
    points_by_setting: dict[arm_memory.Setting, list[CdfPoint]],
    output_dir: Path,
) -> None:
    plt = arm_memory.memory_base.plt
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for setting in arm_memory.SETTINGS:
        points = points_by_setting[setting]
        x_values = [points[0].total_live_peak_gib] + [
            point.total_live_peak_gib for point in points
        ]
        y_values = [0.0] + [
            point.cumulative_query_fraction for point in points
        ]
        ax.step(
            x_values,
            y_values,
            where="post",
            color=setting.color,
            linestyle=LINE_STYLES[setting.scale_factor],
            linewidth=2.2,
            marker=MARKERS[setting.arm],
            markersize=4.2,
            markeredgewidth=0.7,
            markeredgecolor="white",
            label=series_label(setting),
            zorder=3,
        )

    all_values = [
        point.total_live_peak_gib
        for points in points_by_setting.values()
        for point in points
    ]
    value_range = max(all_values) - min(all_values)
    padding = max(1.0, value_range * 0.05)
    ax.set_xlim(min(all_values) - padding, max(all_values) + padding)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([index / 5 for index in range(6)])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Per-query peak memory on any rank (GiB)")
    ax.set_ylabel("Fraction of queries")
    ax.grid(
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
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("TPC-H Query Peak-Memory Distribution", y=0.98, fontsize=17)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.8,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.15, top=0.76)

    stem = "distributed_tpch_peak_memory_query_cdf"
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
    selected = [
        arm_memory.select_memory(results_dir, setting, query)
        for setting in arm_memory.SETTINGS
        for query in QUERIES
    ]
    expected = len(arm_memory.SETTINGS) * len(QUERIES)
    if len(selected) != expected:
        raise ValueError(
            f"expected {expected} memory artifacts, found {len(selected)}"
        )
    query_rows = [
        arm_memory.load_query_peak(item, results_dir) for item in selected
    ]
    points_by_setting = {
        setting: build_cdf_points(
            setting,
            [row for row in query_rows if row.setting_key == setting.key],
        )
        for setting in arm_memory.SETTINGS
    }
    all_points = [
        point
        for setting in arm_memory.SETTINGS
        for point in points_by_setting[setting]
    ]
    arm_memory.memory_base.write_csv(
        output_dir / "distributed_tpch_peak_memory_query_cdf.csv",
        [asdict(point) for point in all_points],
    )
    plot(points_by_setting, output_dir)


if __name__ == "__main__":
    main()
