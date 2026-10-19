#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot rank-0 CDFs of non-NCCL kernel waves for legacy TPC-H campaigns.

Each non-NCCL kernel launch inside the completed ``tpch.profile`` NVTX range
is one empirical-CDF sample.  A wave is the number of CTAs that can reside
concurrently on the GPU for that launch, so the number of waves is

    grid CTAs / (SM count * theoretical resident CTAs per SM).

The launch geometry, resource use, and GPU capacity are read from the Nsight
Systems report.  Temporary SQLite exports are deleted after consumption.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sqlite3
import tempfile
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter
except ModuleNotFoundError as error:
    raise SystemExit(f"error: matplotlib and numpy are required: {error}")

import plot_distributed_tpch_profile_kernels as profile_kernels


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_RANGE = profile_kernels.PROFILE_RANGE
QUERIES = profile_kernels.QUERIES
SETTINGS = (
    profile_kernels.Setting(1000, 8, "legacy_cta64", "Baseline", "#4C78A8"),
    profile_kernels.Setting(3000, 24, "legacy_cta64", "Baseline", "#4C78A8"),
    profile_kernels.Setting(1000, 32, "legacy_cta64", "Baseline", "#4C78A8"),
    profile_kernels.Setting(3000, 32, "legacy_cta64", "Baseline", "#4C78A8"),
)

# CUDA occupancy allocation granularities for the GB200 / sm_100 reports in
# these campaigns.  All capacity limits themselves come from TARGET_INFO_GPU.
REGISTER_ALLOCATION_UNIT_PER_WARP = 256
WARP_ALLOCATION_GRANULARITY = 4
SHARED_MEMORY_ALLOCATION_UNIT = 256


@dataclass(frozen=True)
class DeviceCapacity:
    device_id: int
    name: str
    compute_capability: str
    sm_count: int
    registers_per_sm: int
    shared_memory_per_sm: int
    threads_per_warp: int
    warps_per_sm: int
    blocks_per_sm: int


@dataclass(frozen=True)
class WaveSample:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    query: int
    rank: int
    launch_index: int
    short_name: str
    duration_us: float
    device_id: int
    gpu_name: str
    compute_capability: str
    sm_count: int
    grid_x: int
    grid_y: int
    grid_z: int
    grid_ctas: int
    block_x: int
    block_y: int
    block_z: int
    threads_per_cta: int
    warps_per_cta: int
    registers_per_thread: int
    static_shared_memory_bytes: int
    dynamic_shared_memory_bytes: int
    shared_memory_per_cta_bytes: int
    resident_ctas_per_sm: int
    theoretical_occupancy: float
    ctas_per_wave: int
    waves: float
    report: str


@dataclass
class ExtractedReport:
    selected: profile_kernels.SelectedReport
    profile_range_ns: int
    kernel_launches_in_range: int
    nccl_launches_excluded: int
    samples: list[WaveSample]


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


def ceil_to(value: int, unit: int) -> int:
    return ((value + unit - 1) // unit) * unit


def device_capacities(
    connection: sqlite3.Connection, report: Path
) -> dict[int, DeviceCapacity]:
    rows = connection.execute(
        """
        SELECT id, name, computeMajor, computeMinor, smCount,
               maxRegistersPerSm, maxShmemPerSm, threadsPerWarp,
               maxWarpsPerSm, maxBlocksPerSm
        FROM TARGET_INFO_GPU
        """
    ).fetchall()
    capacities: dict[int, DeviceCapacity] = {}
    for row in rows:
        (
            device_id,
            name,
            compute_major,
            compute_minor,
            sm_count,
            registers_per_sm,
            shared_memory_per_sm,
            threads_per_warp,
            warps_per_sm,
            blocks_per_sm,
        ) = row
        required = (
            sm_count,
            registers_per_sm,
            shared_memory_per_sm,
            threads_per_warp,
            warps_per_sm,
            blocks_per_sm,
        )
        if any(value is None or int(value) <= 0 for value in required):
            raise RuntimeError(f"{report}: incomplete GPU occupancy metadata")
        capability = f"{int(compute_major)}.{int(compute_minor)}"
        if capability != "10.0":
            raise RuntimeError(
                f"{report}: unsupported compute capability {capability}; "
                "allocation granularities are validated for sm_100"
            )
        capacity = DeviceCapacity(
            device_id=int(device_id),
            name=str(name),
            compute_capability=capability,
            sm_count=int(sm_count),
            registers_per_sm=int(registers_per_sm),
            shared_memory_per_sm=int(shared_memory_per_sm),
            threads_per_warp=int(threads_per_warp),
            warps_per_sm=int(warps_per_sm),
            blocks_per_sm=int(blocks_per_sm),
        )
        prior = capacities.get(capacity.device_id)
        if prior is not None and prior != capacity:
            raise RuntimeError(
                f"{report}: conflicting metadata for device {capacity.device_id}"
            )
        capacities[capacity.device_id] = capacity
    if not capacities:
        raise RuntimeError(f"{report}: TARGET_INFO_GPU is empty")
    return capacities


def resident_ctas_per_sm(
    *,
    capacity: DeviceCapacity,
    threads_per_cta: int,
    registers_per_thread: int,
    shared_memory_per_cta: int,
) -> tuple[int, int, float]:
    if threads_per_cta <= 0 or registers_per_thread < 0 or shared_memory_per_cta < 0:
        raise ValueError("invalid kernel resource values")
    warps_per_cta = math.ceil(threads_per_cta / capacity.threads_per_warp)
    by_warps = capacity.warps_per_sm // warps_per_cta

    if registers_per_thread:
        registers_per_warp = ceil_to(
            registers_per_thread * capacity.threads_per_warp,
            REGISTER_ALLOCATION_UNIT_PER_WARP,
        )
        register_warps = capacity.registers_per_sm // registers_per_warp
        register_warps -= register_warps % WARP_ALLOCATION_GRANULARITY
        by_registers = register_warps // warps_per_cta
    else:
        by_registers = capacity.blocks_per_sm

    if shared_memory_per_cta:
        allocated_shared_memory = ceil_to(
            shared_memory_per_cta, SHARED_MEMORY_ALLOCATION_UNIT
        )
        by_shared_memory = (
            capacity.shared_memory_per_sm // allocated_shared_memory
        )
    else:
        by_shared_memory = capacity.blocks_per_sm

    resident = min(
        capacity.blocks_per_sm,
        by_warps,
        by_registers,
        by_shared_memory,
    )
    if resident <= 0:
        raise ValueError(
            "occupancy calculation produced zero resident CTAs per SM: "
            f"threads={threads_per_cta}, registers/thread={registers_per_thread}, "
            f"shared-memory={shared_memory_per_cta}"
        )
    occupancy = resident * warps_per_cta / capacity.warps_per_sm
    if not 0.0 < occupancy <= 1.0:
        raise ValueError(f"invalid theoretical occupancy: {occupancy}")
    return resident, warps_per_cta, occupancy


def extract_sqlite(
    selected: profile_kernels.SelectedReport,
    sqlite_path: Path,
    results_dir: Path,
) -> ExtractedReport:
    connection = sqlite3.connect(sqlite_path)
    try:
        capacities = device_capacities(connection, selected.report)
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
        events = connection.execute(
            """
            SELECT s.value, k.start, k.end, k.deviceId,
                   k.registersPerThread,
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ,
                   k.staticSharedMemory, k.dynamicSharedMemory
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.start >= ? AND k.end <= ? AND k.end > k.start
            ORDER BY k.start, k.gridId
            """,
            (range_start, range_end),
        ).fetchall()
    finally:
        connection.close()

    samples: list[WaveSample] = []
    nccl_launches = 0
    relative_report = str(selected.report.relative_to(results_dir))
    for launch_index, event in enumerate(events, start=1):
        (
            short_name,
            start,
            end,
            device_id,
            registers_per_thread,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            static_shared_memory,
            dynamic_shared_memory,
        ) = event
        name = str(short_name)
        if "nccl" in name.casefold():
            nccl_launches += 1
            continue
        capacity = capacities.get(int(device_id))
        if capacity is None:
            raise RuntimeError(
                f"{selected.report}: no GPU metadata for device {device_id}"
            )
        grid_ctas = int(grid_x) * int(grid_y) * int(grid_z)
        threads_per_cta = int(block_x) * int(block_y) * int(block_z)
        shared_memory_per_cta = int(static_shared_memory) + int(
            dynamic_shared_memory
        )
        resident, warps_per_cta, occupancy = resident_ctas_per_sm(
            capacity=capacity,
            threads_per_cta=threads_per_cta,
            registers_per_thread=int(registers_per_thread),
            shared_memory_per_cta=shared_memory_per_cta,
        )
        ctas_per_wave = capacity.sm_count * resident
        waves = grid_ctas / ctas_per_wave
        if not math.isfinite(waves) or waves <= 0.0:
            raise RuntimeError(f"{selected.report}: invalid wave count {waves}")
        samples.append(
            WaveSample(
                setting_key=selected.setting.key,
                scale_factor=selected.setting.scale_factor,
                gpus=selected.setting.gpus,
                arm=selected.setting.arm,
                query=selected.query,
                rank=0,
                launch_index=launch_index,
                short_name=name,
                duration_us=(int(end) - int(start)) / 1e3,
                device_id=capacity.device_id,
                gpu_name=capacity.name,
                compute_capability=capacity.compute_capability,
                sm_count=capacity.sm_count,
                grid_x=int(grid_x),
                grid_y=int(grid_y),
                grid_z=int(grid_z),
                grid_ctas=grid_ctas,
                block_x=int(block_x),
                block_y=int(block_y),
                block_z=int(block_z),
                threads_per_cta=threads_per_cta,
                warps_per_cta=warps_per_cta,
                registers_per_thread=int(registers_per_thread),
                static_shared_memory_bytes=int(static_shared_memory),
                dynamic_shared_memory_bytes=int(dynamic_shared_memory),
                shared_memory_per_cta_bytes=shared_memory_per_cta,
                resident_ctas_per_sm=resident,
                theoretical_occupancy=occupancy,
                ctas_per_wave=ctas_per_wave,
                waves=waves,
                report=relative_report,
            )
        )
    if not samples:
        raise RuntimeError(f"{selected.report}: no non-NCCL kernels in profile range")
    return ExtractedReport(
        selected=selected,
        profile_range_ns=range_end - range_start,
        kernel_launches_in_range=len(events),
        nccl_launches_excluded=nccl_launches,
        samples=samples,
    )


def process_report(
    nsys: str,
    selected: profile_kernels.SelectedReport,
    results_dir: Path,
    timeout: float,
) -> ExtractedReport:
    with tempfile.TemporaryDirectory(prefix="tpch_nsys_rank0_waves_") as directory:
        sqlite_path = Path(directory) / "report.sqlite"
        profile_kernels.export_sqlite(nsys, selected.report, sqlite_path, timeout)
        return extract_sqlite(selected, sqlite_path, results_dir)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot_setting(
    setting: profile_kernels.Setting,
    samples: list[WaveSample],
    output_dir: Path,
) -> None:
    waves = np.sort(np.asarray([sample.waves for sample in samples], dtype=float))
    if waves.size == 0 or np.any(~np.isfinite(waves)) or np.any(waves <= 0):
        raise ValueError(f"{setting.key}: invalid CDF samples")
    cdf = np.arange(1, waves.size + 1, dtype=float) / waves.size
    plot_x = np.concatenate(([waves[0]], waves))
    plot_y = np.concatenate(([0.0], cdf))
    aggregate_time_us = sum(sample.duration_us for sample in samples)
    at_least_one_wave_time_us = sum(
        sample.duration_us for sample in samples if sample.waves >= 1.0
    )
    if aggregate_time_us <= 0.0:
        raise ValueError(f"{setting.key}: aggregate kernel time is not positive")
    at_least_one_wave_time_fraction = (
        at_least_one_wave_time_us / aggregate_time_us
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.step(
        plot_x,
        plot_y,
        where="post",
        color="#4C78A8",
        linewidth=2.2,
        zorder=3,
    )
    ax.axvline(
        1.0,
        color="#777777",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    ax.text(
        1.0,
        0.035,
        "  1 wave",
        ha="left",
        va="bottom",
        fontsize=12,
        color="#555555",
    )
    ax.text(
        0.025,
        0.965,
        "Kernels with ≥1 wave account for\n"
        f"{100.0 * at_least_one_wave_time_fraction:.1f}% of aggregate kernel time",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "linewidth": 0.8,
            "alpha": 0.94,
        },
        zorder=4,
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.01)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("Number of waves per kernel launch (log scale)")
    ax.set_ylabel("Fraction of non-NCCL kernel launches")
    ax.set_title(
        f"TPC-H Kernel Wave CDF — SF={setting.scale_factor}, "
        f"{setting.gpus} GPUs",
        pad=13,
    )
    ax.grid(
        axis="both",
        which="major",
        color="#D8D8D8",
        linewidth=0.8,
        linestyle=(0, (2, 3)),
        zorder=0,
    )
    ax.grid(
        axis="x",
        which="minor",
        color="#EEEEEE",
        linewidth=0.5,
        linestyle=(0, (2, 4)),
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="both", colors="#333333")
    fig.tight_layout()

    name = f"rank0_profile_kernel_wave_cdf_{setting.key}"
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
        profile_kernels.select_rank_zero_report(results_dir, setting, query)
        for setting in SETTINGS
        for query in QUERIES
    ]
    expected_reports = len(SETTINGS) * len(QUERIES)
    if len(selected_reports) != expected_reports:
        raise ValueError("unexpected selected report count")
    print(
        f"processing {len(selected_reports)} rank-0 legacy_cta64 reports "
        f"with {args.jobs} concurrent exports"
    )

    extracted: list[ExtractedReport] = []
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
                extracted.append(future.result())
            except Exception as error:
                raise RuntimeError(
                    f"failed to process {selected.report}: {error}"
                ) from error
            if completed % 10 == 0 or completed == len(futures):
                print(f"processed {completed}/{len(futures)} reports")

    sample_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for report in sorted(
        extracted,
        key=lambda item: (
            item.selected.setting.scale_factor,
            item.selected.query,
        ),
    ):
        selected = report.selected
        if len(report.samples) + report.nccl_launches_excluded != (
            report.kernel_launches_in_range
        ):
            raise ValueError(f"{selected.report}: launch accounting mismatch")
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
                "non_nccl_launches": len(report.samples),
                "nccl_launches_excluded": report.nccl_launches_excluded,
                "result_status": selected.result_status,
                "failure_kind": selected.failure_kind,
                "report": str(selected.report.relative_to(results_dir)),
            }
        )
        sample_rows.extend(asdict(sample) for sample in report.samples)

    for setting in SETTINGS:
        setting_reports = [
            report for report in extracted if report.selected.setting == setting
        ]
        if {report.selected.query for report in setting_reports} != set(QUERIES):
            raise ValueError(f"{setting.key}: incomplete Q1-Q22 coverage")
        samples = [sample for report in setting_reports for sample in report.samples]
        if any("nccl" in sample.short_name.casefold() for sample in samples):
            raise ValueError(f"{setting.key}: NCCL sample survived filtering")
        wave_values = np.asarray([sample.waves for sample in samples], dtype=float)
        duration_values_us = np.asarray(
            [sample.duration_us for sample in samples], dtype=float
        )
        cdf_terminal = len(wave_values) / len(wave_values)
        if cdf_terminal != 1.0:
            raise ValueError(f"{setting.key}: CDF does not terminate at one")
        aggregate_time_us = float(np.sum(duration_values_us))
        at_least_one_wave_time_us = float(
            np.sum(duration_values_us[wave_values >= 1.0])
        )
        if aggregate_time_us <= 0.0:
            raise ValueError(f"{setting.key}: aggregate kernel time is not positive")
        summary_rows.append(
            {
                "setting_key": setting.key,
                "scale_factor": setting.scale_factor,
                "gpus": setting.gpus,
                "arm": setting.arm,
                "queries": len(setting_reports),
                "non_nccl_launches": len(samples),
                "nccl_launches_excluded": sum(
                    report.nccl_launches_excluded for report in setting_reports
                ),
                "minimum_waves": float(np.min(wave_values)),
                "median_waves": float(np.median(wave_values)),
                "p90_waves": float(np.percentile(wave_values, 90)),
                "p95_waves": float(np.percentile(wave_values, 95)),
                "maximum_waves": float(np.max(wave_values)),
                "fraction_at_most_one_wave": float(np.mean(wave_values <= 1.0)),
                "aggregate_kernel_time_ms": aggregate_time_us / 1e3,
                "at_least_one_wave_kernel_time_ms": (
                    at_least_one_wave_time_us / 1e3
                ),
                "at_least_one_wave_fraction_of_aggregate_kernel_time": (
                    at_least_one_wave_time_us / aggregate_time_us
                ),
            }
        )
        plot_setting(setting, samples, output_dir)

    write_csv(output_dir / "rank0_profile_kernel_wave_samples.csv", sample_rows)
    write_csv(output_dir / "rank0_profile_kernel_wave_reports.csv", report_rows)
    write_csv(output_dir / "rank0_profile_kernel_wave_summary.csv", summary_rows)


if __name__ == "__main__":
    main()
