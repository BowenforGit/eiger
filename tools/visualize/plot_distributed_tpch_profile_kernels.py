#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot the rank-0 kernels covering >80% of non-NCCL profile kernel time.

For four distributed TPC-H settings, this tool exports the rank-0 Nsight
Systems report for every Q1-Q22 profile attempt, restricts CUDA kernel events to
the completed ``tpch.profile`` NVTX range, groups exact ``shortName`` values,
and selects the smallest duration-sorted prefix whose cumulative time exceeds
80% of all non-NCCL kernel time.

Temporary SQLite exports are consumed immediately and deleted.  Reading CUDA
trace durations does not require GPU performance-counter access.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    sys.exit(f"error: matplotlib is required: {error}")


REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES = tuple(range(1, 23))
PROFILE_RANGE = "tpch.profile"
TARGET_COVERAGE = 0.80
ALLOWED_VERIFICATION_FAILURES = frozenset(((3000, 3), (3000, 18)))
RANK_ZERO_PATTERN = re.compile(r"_rank0\.nsys-rep$")


@dataclass(frozen=True)
class Setting:
    scale_factor: int
    gpus: int
    arm: str
    label: str
    color: str

    @property
    def key(self) -> str:
        return f"sf{self.scale_factor}_{self.gpus}gpu_{self.arm}"


SETTINGS = (
    Setting(1000, 8, "legacy_cta64", "Baseline", "#4C78A8"),
    Setting(1000, 8, "synthesized_cta64", "Opt. comp", "#54A24B"),
    Setting(3000, 24, "legacy_cta64", "Baseline", "#4C78A8"),
    Setting(3000, 24, "synthesized_cta64", "Opt. comp", "#54A24B"),
)


@dataclass(frozen=True)
class SelectedReport:
    setting: Setting
    query: int
    report: Path
    result_status: str
    failure_kind: str


@dataclass
class ExtractedReport:
    selected: SelectedReport
    profile_range_ns: int
    kernel_launches_in_range: int
    non_nccl_launches: int
    nccl_launches_excluded: int
    non_nccl_time_ns: int
    nccl_time_ns_excluded: int
    kernels: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class KernelSummary:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    setting_label: str
    short_name: str
    launch_count: int
    aggregate_time_ms: float
    aggregate_time_s: float
    fraction_of_non_nccl_time: float
    cumulative_fraction: float
    selected_above_80_percent: bool


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


def attempt_number(path: Path) -> int:
    return int(path.name.removeprefix("attempt-"))


def profile_query_root(results_dir: Path, setting: Setting, query: int) -> Path:
    return (
        results_dir
        / f"sf{setting.scale_factor}"
        / setting.arm
        / f"{setting.gpus}gpu_128p_profiles"
        / "arms"
        / setting.arm
        / "profiles"
        / f"q{query:02d}"
    )


def select_rank_zero_report(
    results_dir: Path, setting: Setting, query: int
) -> SelectedReport:
    root = profile_query_root(results_dir, setting, query)
    attempts = sorted(root.glob("attempt-*"), key=attempt_number, reverse=True)
    for attempt in attempts:
        result_path = attempt / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        accepted = result.get("status") == "pass"
        if (setting.scale_factor, query) in ALLOWED_VERIFICATION_FAILURES:
            accepted = accepted or (
                result.get("status") == "fail"
                and result.get("failure_kind") == "verification"
                and result.get("measurements_complete") is True
            )
        if not accepted:
            continue
        rank_zero_names = [
            name
            for name in result.get("expected_artifacts", ())
            if RANK_ZERO_PATTERN.search(name)
        ]
        if len(rank_zero_names) != 1:
            continue
        report = attempt / rank_zero_names[0]
        if not report.is_file() or report.stat().st_size == 0:
            continue
        return SelectedReport(
            setting=setting,
            query=query,
            report=report,
            result_status=str(result["status"]),
            failure_kind=str(result.get("failure_kind") or ""),
        )
    raise ValueError(f"no usable rank-0 profile attempt under {root}")


def export_sqlite(
    nsys: str,
    report: Path,
    sqlite_path: Path,
    timeout: float,
) -> None:
    result = subprocess.run(
        [
            nsys,
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(report),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nsys export failed for {report} (exit {result.returncode}):\n"
            f"{result.stdout}"
        )
    if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
        raise RuntimeError(f"nsys export did not create {sqlite_path}")


def extract_sqlite(
    selected: SelectedReport, sqlite_path: Path
) -> ExtractedReport:
    connection = sqlite3.connect(sqlite_path)
    try:
        ranges = connection.execute(
            """
            SELECT e.start, e.end
            FROM NVTX_EVENTS e
            LEFT JOIN StringIds s ON s.id = e.textId
            WHERE coalesce(e.text, s.value) = ? AND e.end IS NOT NULL
            ORDER BY e.start
            """,
            (PROFILE_RANGE,),
        ).fetchall()
        if len(ranges) != 1:
            raise RuntimeError(
                f"{selected.report}: expected one completed {PROFILE_RANGE} "
                f"range, found {len(ranges)}"
            )
        range_start, range_end = map(int, ranges[0])
        if range_start >= range_end:
            raise RuntimeError(f"{selected.report}: invalid profile range")
        events = connection.execute(
            """
            SELECT s.value, k.end - k.start
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.start >= ? AND k.end <= ? AND k.end > k.start
            """,
            (range_start, range_end),
        ).fetchall()
    finally:
        connection.close()

    kernels: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    non_nccl_time_ns = 0
    nccl_time_ns = 0
    non_nccl_launches = 0
    nccl_launches = 0
    for short_name, duration in events:
        name = str(short_name)
        duration_ns = int(duration)
        if "nccl" in name.casefold():
            nccl_launches += 1
            nccl_time_ns += duration_ns
            continue
        non_nccl_launches += 1
        non_nccl_time_ns += duration_ns
        kernels[name][0] += duration_ns
        kernels[name][1] += 1
    if not kernels or non_nccl_time_ns <= 0:
        raise RuntimeError(f"{selected.report}: no non-NCCL kernels in profile range")
    frozen_kernels = {
        name: (values[0], values[1]) for name, values in kernels.items()
    }
    return ExtractedReport(
        selected=selected,
        profile_range_ns=range_end - range_start,
        kernel_launches_in_range=len(events),
        non_nccl_launches=non_nccl_launches,
        nccl_launches_excluded=nccl_launches,
        non_nccl_time_ns=non_nccl_time_ns,
        nccl_time_ns_excluded=nccl_time_ns,
        kernels=frozen_kernels,
    )


def process_report(
    nsys: str, selected: SelectedReport, timeout: float
) -> ExtractedReport:
    with tempfile.TemporaryDirectory(prefix="tpch_nsys_rank0_") as directory:
        sqlite_path = Path(directory) / "report.sqlite"
        export_sqlite(nsys, selected.report, sqlite_path, timeout)
        return extract_sqlite(selected, sqlite_path)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def summarize_setting(
    setting: Setting, reports: list[ExtractedReport]
) -> list[KernelSummary]:
    if {report.selected.query for report in reports} != set(QUERIES):
        raise ValueError(f"{setting.key}: incomplete query coverage")
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for report in reports:
        for name, (duration_ns, launches) in report.kernels.items():
            totals[name][0] += duration_ns
            totals[name][1] += launches
    total_time_ns = sum(values[0] for values in totals.values())
    if total_time_ns != sum(report.non_nccl_time_ns for report in reports):
        raise ValueError(f"{setting.key}: report and kernel totals disagree")

    ordered = sorted(totals.items(), key=lambda item: (-item[1][0], item[0]))
    cumulative_ns = 0
    crossing_index = None
    cumulative_fractions = []
    for index, (_, (duration_ns, _)) in enumerate(ordered):
        cumulative_ns += duration_ns
        fraction = cumulative_ns / total_time_ns
        cumulative_fractions.append(fraction)
        if crossing_index is None and fraction > TARGET_COVERAGE:
            crossing_index = index
    if crossing_index is None:
        raise ValueError(f"{setting.key}: failed to cross target coverage")
    previous_coverage = (
        cumulative_fractions[crossing_index - 1] if crossing_index else 0.0
    )
    selected_coverage = cumulative_fractions[crossing_index]
    if previous_coverage > TARGET_COVERAGE or selected_coverage <= TARGET_COVERAGE:
        raise ValueError(f"{setting.key}: selected prefix is not minimal")

    rows = []
    for index, (name, (duration_ns, launches)) in enumerate(ordered):
        rows.append(
            KernelSummary(
                setting_key=setting.key,
                scale_factor=setting.scale_factor,
                gpus=setting.gpus,
                arm=setting.arm,
                setting_label=setting.label,
                short_name=name,
                launch_count=launches,
                aggregate_time_ms=duration_ns / 1e6,
                aggregate_time_s=duration_ns / 1e9,
                fraction_of_non_nccl_time=duration_ns / total_time_ns,
                cumulative_fraction=cumulative_fractions[index],
                selected_above_80_percent=index <= crossing_index,
            )
        )
    return rows


def display_name(short_name: str) -> str:
    return textwrap.fill(
        short_name,
        width=52,
        break_long_words=True,
        break_on_hyphens=False,
    )


def plot_setting(
    setting: Setting, rows: list[KernelSummary], output_dir: Path
) -> None:
    selected = [row for row in rows if row.selected_above_80_percent]
    if not selected:
        raise ValueError(f"{setting.key}: no selected kernels")
    coverage = sum(row.fraction_of_non_nccl_time for row in selected)
    previous = coverage - selected[-1].fraction_of_non_nccl_time
    if coverage <= TARGET_COVERAGE or previous > TARGET_COVERAGE:
        raise ValueError(f"{setting.key}: invalid plotted coverage")

    labels = [display_name(row.short_name) for row in selected]
    values = [row.aggregate_time_s for row in selected]
    line_count = sum(label.count("\n") + 1 for label in labels)
    height = max(4.3, 1.8 + 0.34 * line_count)
    fig, ax = plt.subplots(figsize=(10.8, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = list(range(len(selected)))
    ax.barh(
        y,
        values,
        color=setting.color,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    maximum = max(values)
    for index, row in enumerate(selected):
        ax.text(
            row.aggregate_time_s + maximum * 0.015,
            index,
            f"{row.aggregate_time_s:.3f} s  "
            f"({100.0 * row.fraction_of_non_nccl_time:.1f}%)",
            ha="left",
            va="center",
            fontsize=9,
            color="#2F2F2F",
        )
    ax.set_xlim(0.0, maximum * 1.30)
    ax.set_xlabel("Aggregate rank-0 kernel time across Q1–Q22 (s)")
    ax.set_title(
        f"Rank-0 TPC-H Non-NCCL Kernels — SF={setting.scale_factor}, "
        f"{setting.gpus} GPUs, {setting.label}",
        pad=13,
    )
    ax.text(
        0.995,
        0.015,
        f"{len(selected)} kernels cover {100.0 * coverage:.1f}% "
        "of non-NCCL kernel time",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.grid(
        axis="x",
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
    fig.tight_layout()

    name = f"rank0_profile_kernels_{setting.key}"
    for suffix in ("pdf", "png"):
        format_dir = output_dir / suffix
        format_dir.mkdir(parents=True, exist_ok=True)
        output = format_dir / f"{name}.{suffix}"
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        print(f"wrote {output}")
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
        select_rank_zero_report(results_dir, setting, query)
        for setting in SETTINGS
        for query in QUERIES
    ]
    if len(selected_reports) != len(SETTINGS) * len(QUERIES):
        raise ValueError("unexpected selected report count")
    print(
        f"processing {len(selected_reports)} rank-0 reports with "
        f"{args.jobs} concurrent exports"
    )

    extracted = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_report,
                nsys,
                selected,
                args.export_timeout,
            ): selected
            for selected in selected_reports
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            selected = futures[future]
            try:
                extracted.append(future.result())
            except Exception as error:
                raise RuntimeError(
                    f"failed to process {selected.report}: {error}"
                ) from error
            if completed % 10 == 0 or completed == len(futures):
                print(f"processed {completed}/{len(futures)} reports")

    report_rows = []
    kernel_rows_by_setting = {}
    all_kernel_rows = []
    for report in sorted(
        extracted,
        key=lambda item: (
            item.selected.setting.scale_factor,
            item.selected.setting.arm,
            item.selected.query,
        ),
    ):
        selected = report.selected
        report_rows.append(
            {
                "setting_key": selected.setting.key,
                "scale_factor": selected.setting.scale_factor,
                "gpus": selected.setting.gpus,
                "arm": selected.setting.arm,
                "query": selected.query,
                "rank": 0,
                "profile_range_ms": report.profile_range_ns / 1e6,
                "kernel_launches_in_range": report.kernel_launches_in_range,
                "non_nccl_launches": report.non_nccl_launches,
                "nccl_launches_excluded": report.nccl_launches_excluded,
                "non_nccl_time_ms": report.non_nccl_time_ns / 1e6,
                "nccl_time_ms_excluded": report.nccl_time_ns_excluded / 1e6,
                "result_status": selected.result_status,
                "failure_kind": selected.failure_kind,
                "report": str(selected.report.relative_to(results_dir)),
            }
        )
    for setting in SETTINGS:
        setting_reports = [
            report for report in extracted if report.selected.setting == setting
        ]
        rows = summarize_setting(setting, setting_reports)
        kernel_rows_by_setting[setting] = rows
        all_kernel_rows.extend(rows)

    write_csv(
        output_dir / "rank0_profile_report_summary.csv",
        report_rows,
        list(report_rows[0]),
    )
    kernel_dictionaries = [row.__dict__ for row in all_kernel_rows]
    write_csv(
        output_dir / "rank0_profile_kernel_summary.csv",
        kernel_dictionaries,
        list(kernel_dictionaries[0]),
    )
    for setting in SETTINGS:
        plot_setting(setting, kernel_rows_by_setting[setting], output_dir)


if __name__ == "__main__":
    main()
