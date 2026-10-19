#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot rank-0 count/retrieve kernel time for all shuffle EqJoins.

For SF=1000 on 8 GPUs and SF=3000 on 24 GPUs, this tool reads the rank-0
Nsight Systems report for Q1-Q22 in four communication-overlap arms.  CUDA
launches are attributed to ``DistributedEqJoin(...;shuffle;...)`` NVTX step
ranges through the CUDA-runtime/kernel correlation ID.  Only kernels whose
exact Nsight ``shortName`` is ``count`` or ``retrieve`` are included.

Kernel time is the CUPTI GPU execution interval ``end - start``.  Durations
are summed across launches; they are not scaled to end-to-end time.  Temporary
SQLite exports are deleted after consumption.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.patches as mpatches

import plot_distributed_tpch_overlap_arm_comparison as overlap_plot
import plot_distributed_tpch_profile_kernels as profile_kernels
from plot_distributed_tpch_aggregate_breakdown import REPO_ROOT, plt


QUERIES = tuple(range(1, 23))
FIGURE_PREFIX = "distributed_tpch_eqjoin_count_retrieve_kernel_time"
COUNT_COLOR = "#4C78A8"
RETRIEVE_COLOR = "#F58518"


SETTINGS = tuple(
    profile_kernels.Setting(scale, gpus, arm, label, color)
    for scale, gpus in overlap_plot.CONFIGURATIONS
    for arm, label, color in overlap_plot.ARMS
)


@dataclass(frozen=True)
class QueryKernelTime:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    label: str
    query: int
    rank: int
    shuffle_eqjoin_count: int
    count_launches: int
    retrieve_launches: int
    count_time_ms: float
    retrieve_time_ms: float
    result_status: str
    failure_kind: str
    report: str


@dataclass(frozen=True)
class AggregateKernelTime:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    label: str
    query_count: int
    shuffle_eqjoin_count: int
    count_launches: int
    retrieve_launches: int
    count_time_s: float
    retrieve_time_s: float
    total_time_s: float


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


def extract_sqlite(
    selected: profile_kernels.SelectedReport,
    sqlite_path: Path,
    results_dir: Path,
) -> QueryKernelTime:
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
            (profile_kernels.PROFILE_RANGE,),
        ).fetchall()
        if len(profile_ranges) != 1:
            raise RuntimeError(
                f"{selected.report}: expected one completed "
                f"{profile_kernels.PROFILE_RANGE} range, "
                f"found {len(profile_ranges)}"
            )
        profile_start, profile_end = map(int, profile_ranges[0])
        if profile_start >= profile_end:
            raise RuntimeError(f"{selected.report}: invalid profile range")

        shuffle_ranges = connection.execute(
            """
            SELECT e.start, e.end, e.globalTid,
                   coalesce(e.text, s.value) AS label
            FROM NVTX_EVENTS e
            LEFT JOIN StringIds s ON s.id = e.textId
            WHERE e.end IS NOT NULL
              AND e.start >= ? AND e.end <= ?
              AND coalesce(e.text, s.value) LIKE 'step%:DistributedEqJoin(%'
              AND coalesce(e.text, s.value) LIKE '%;shuffle;%'
            ORDER BY e.start
            """,
            (profile_start, profile_end),
        ).fetchall()

        events = connection.execute(
            """
            SELECT n.start, n.end, n.globalTid,
                   coalesce(n.text, nt.value) AS step_label,
                   ks.value AS short_name,
                   k.start, k.end, k.gridId
            FROM NVTX_EVENTS n
            LEFT JOIN StringIds nt ON nt.id = n.textId
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
              ON r.globalTid = n.globalTid
             AND r.start >= n.start AND r.start <= n.end
            JOIN CUPTI_ACTIVITY_KIND_KERNEL k
              ON k.correlationId = r.correlationId
            JOIN StringIds ks ON ks.id = k.shortName
            WHERE n.end IS NOT NULL
              AND n.start >= ? AND n.end <= ?
              AND coalesce(n.text, nt.value) LIKE 'step%:DistributedEqJoin(%'
              AND coalesce(n.text, nt.value) LIKE '%;shuffle;%'
              AND ks.value IN ('count', 'retrieve')
              AND k.end > k.start
            ORDER BY n.start, k.start, k.gridId
            """,
            (profile_start, profile_end),
        ).fetchall()
    finally:
        connection.close()

    range_keys = {
        (int(start), int(end), int(global_tid), str(label))
        for start, end, global_tid, label in shuffle_ranges
    }
    if len(range_keys) != len(shuffle_ranges):
        raise RuntimeError(f"{selected.report}: duplicate shuffle EqJoin ranges")

    per_range: dict[tuple[int, int, int, str], dict[str, list[int]]] = {
        key: {"count": [], "retrieve": []} for key in range_keys
    }
    for (
        range_start,
        range_end,
        global_tid,
        step_label,
        short_name,
        kernel_start,
        kernel_end,
        _grid_id,
    ) in events:
        key = (
            int(range_start),
            int(range_end),
            int(global_tid),
            str(step_label),
        )
        if key not in per_range:
            raise RuntimeError(
                f"{selected.report}: kernel mapped to an unknown EqJoin range"
            )
        per_range[key][str(short_name)].append(
            int(kernel_end) - int(kernel_start)
        )

    for key, launches in per_range.items():
        count_count = len(launches["count"])
        retrieve_count = len(launches["retrieve"])
        if count_count != 1 or retrieve_count != 1:
            raise RuntimeError(
                f"{selected.report}: shuffle EqJoin {key[3]!r} has "
                f"{count_count} count and {retrieve_count} retrieve launches; "
                "expected exactly one of each"
            )

    count_durations = [
        duration
        for launches in per_range.values()
        for duration in launches["count"]
    ]
    retrieve_durations = [
        duration
        for launches in per_range.values()
        for duration in launches["retrieve"]
    ]
    return QueryKernelTime(
        setting_key=selected.setting.key,
        scale_factor=selected.setting.scale_factor,
        gpus=selected.setting.gpus,
        arm=selected.setting.arm,
        label=selected.setting.label.replace("\n", " "),
        query=selected.query,
        rank=0,
        shuffle_eqjoin_count=len(per_range),
        count_launches=len(count_durations),
        retrieve_launches=len(retrieve_durations),
        count_time_ms=sum(count_durations) / 1e6,
        retrieve_time_ms=sum(retrieve_durations) / 1e6,
        result_status=selected.result_status,
        failure_kind=selected.failure_kind,
        report=str(selected.report.relative_to(results_dir)),
    )


def process_report(
    nsys: str,
    selected: profile_kernels.SelectedReport,
    results_dir: Path,
    timeout: float,
) -> QueryKernelTime:
    with tempfile.TemporaryDirectory(
        prefix="tpch_nsys_eqjoin_kernels_"
    ) as directory:
        sqlite_path = Path(directory) / "report.sqlite"
        profile_kernels.export_sqlite(
            nsys, selected.report, sqlite_path, timeout
        )
        return extract_sqlite(selected, sqlite_path, results_dir)


def summarize(rows: list[QueryKernelTime]) -> list[AggregateKernelTime]:
    summaries = []
    for setting in SETTINGS:
        selected = [row for row in rows if row.setting_key == setting.key]
        if {row.query for row in selected} != set(QUERIES):
            raise ValueError(f"{setting.key}: incomplete Q1-Q22 coverage")
        shuffle_count = sum(row.shuffle_eqjoin_count for row in selected)
        count_launches = sum(row.count_launches for row in selected)
        retrieve_launches = sum(row.retrieve_launches for row in selected)
        if count_launches != shuffle_count or retrieve_launches != shuffle_count:
            raise ValueError(
                f"{setting.key}: launch totals do not match shuffle joins"
            )
        count_time_s = sum(row.count_time_ms for row in selected) / 1000.0
        retrieve_time_s = (
            sum(row.retrieve_time_ms for row in selected) / 1000.0
        )
        summaries.append(
            AggregateKernelTime(
                setting_key=setting.key,
                scale_factor=setting.scale_factor,
                gpus=setting.gpus,
                arm=setting.arm,
                label=setting.label.replace("\n", " "),
                query_count=len(selected),
                shuffle_eqjoin_count=shuffle_count,
                count_launches=count_launches,
                retrieve_launches=retrieve_launches,
                count_time_s=count_time_s,
                retrieve_time_s=retrieve_time_s,
                total_time_s=count_time_s + retrieve_time_s,
            )
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot_configuration(
    summaries: list[AggregateKernelTime],
    scale_factor: int,
    gpus: int,
    output_dir: Path,
) -> None:
    selected = [
        row
        for row in summaries
        if row.scale_factor == scale_factor and row.gpus == gpus
    ]
    arm_order = tuple(arm for arm, _, _ in overlap_plot.ARMS)
    lookup = {row.arm: row for row in selected}
    if set(lookup) != set(arm_order):
        raise ValueError(
            f"SF={scale_factor}, {gpus} GPUs: unexpected arm coverage"
        )

    ordered = [lookup[arm] for arm in arm_order]
    positions = list(range(len(ordered)))
    labels = [label for _, label, _ in overlap_plot.ARMS]
    count_values = [row.count_time_s for row in ordered]
    retrieve_values = [row.retrieve_time_s for row in ordered]
    totals = [row.total_time_s for row in ordered]
    maximum = max(totals)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    count_bars = ax.bar(
        positions,
        count_values,
        width=0.70,
        color=COUNT_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    retrieve_bars = ax.bar(
        positions,
        retrieve_values,
        width=0.70,
        bottom=count_values,
        color=RETRIEVE_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    for index, (count, retrieve, total) in enumerate(
        zip(count_values, retrieve_values, totals, strict=True)
    ):
        if count > maximum * 0.09:
            ax.text(
                index,
                count / 2.0,
                f"{count:.2f} s",
                ha="center",
                va="center",
                fontsize=10.0,
                fontweight="bold",
                color="white",
            )
        if retrieve > maximum * 0.09:
            ax.text(
                index,
                count + retrieve / 2.0,
                f"{retrieve:.2f} s",
                ha="center",
                va="center",
                fontsize=10.0,
                fontweight="bold",
                color="white",
            )
        ax.text(
            index,
            total + maximum * 0.022,
            f"{total:.2f} s",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", labelsize=10.5, pad=6)
    ax.set_ylabel("Aggregate rank-0 kernel time across Q1–Q22 (s)")
    ax.set_title(
        f"TPC-H Shuffle EqJoin Kernels — SF={scale_factor}, {gpus} GPUs",
        pad=16,
    )
    ax.set_ylim(0.0, maximum * 1.19)
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

    handles = [
        mpatches.Patch(color=COUNT_COLOR, label="Count"),
        mpatches.Patch(color=RETRIEVE_COLOR, label="Retrieve"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.20, top=0.89)

    stem = f"{FIGURE_PREFIX}_sf{scale_factor}_{gpus}gpu"
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
    expected = len(SETTINGS) * len(QUERIES)
    if len(selected_reports) != expected:
        raise ValueError("unexpected selected report count")
    print(
        f"processing {expected} rank-0 reports with "
        f"{args.jobs} concurrent exports"
    )

    query_rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_report,
                nsys,
                selected,
                results_dir,
                args.export_timeout,
            ): selected
            for selected in selected_reports
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            selected = futures[future]
            try:
                query_rows.append(future.result())
            except Exception as error:
                raise RuntimeError(
                    f"failed to process {selected.report}: {error}"
                ) from error
            if completed % 10 == 0 or completed == len(futures):
                print(f"processed {completed}/{len(futures)} reports")

    query_rows.sort(
        key=lambda row: (row.scale_factor, row.gpus, row.arm, row.query)
    )
    summaries = summarize(query_rows)
    write_csv(
        output_dir / f"{FIGURE_PREFIX}_by_query.csv",
        [asdict(row) for row in query_rows],
    )
    write_csv(
        output_dir / f"{FIGURE_PREFIX}_summary.csv",
        [asdict(row) for row in summaries],
    )
    for scale_factor, gpus in overlap_plot.CONFIGURATIONS:
        plot_configuration(summaries, scale_factor, gpus, output_dir)
    for row in summaries:
        print(
            f"SF={row.scale_factor}, {row.gpus} GPUs, {row.arm}: "
            f"count={row.count_time_s:.3f} s, "
            f"retrieve={row.retrieve_time_s:.3f} s, "
            f"joins={row.shuffle_eqjoin_count}"
        )


if __name__ == "__main__":
    main()
