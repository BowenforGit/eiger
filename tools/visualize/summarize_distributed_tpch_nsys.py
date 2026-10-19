#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Summarize system-wide GPU metrics from distributed TPC-H NSYS reports."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


PROFILE_RANGE = "tpch.profile"
SM_ACTIVE = "SMs Active [Throughput %]"
GR_ACTIVE = "GR Active [Throughput %]"
SM_ISSUE = "SM Issue [Throughput %]"
TENSOR_ACTIVE = "Tensor Active [Throughput %]"
DRAM_READ = "DRAM Read Bandwidth [Throughput %]"
DRAM_WRITE = "DRAM Write Bandwidth [Throughput %]"
PCIE_RX = "PCIe RX Throughput [Throughput %]"
PCIE_TX = "PCIe TX Throughput [Throughput %]"
GPC_CLOCK = "GPC Clock Frequency [MHz]"
SYS_CLOCK = "SYS Clock Frequency [MHz]"

NVLINK_RX = (
    "NVLink RX Requests Protocol Data [Throughput %]",
    "NVLink RX Requests User Data [Throughput %]",
    "NVLink RX Responses Protocol Data [Throughput %]",
    "NVLink RX Responses User Data [Throughput %]",
)
NVLINK_TX = (
    "NVLink TX Requests Protocol Data [Throughput %]",
    "NVLink TX Requests User Data [Throughput %]",
    "NVLink TX Responses Protocol Data [Throughput %]",
    "NVLink TX Responses User Data [Throughput %]",
)

DERIVED_DRAM = "DRAM Total Bandwidth [Derived Throughput %]"
DERIVED_NVLINK_RX = "NVLink RX Total [Derived Throughput %]"
DERIVED_NVLINK_TX = "NVLink TX Total [Derived Throughput %]"
DERIVED_NVLINK_DUPLEX = "NVLink Full-duplex Total [Derived Throughput %]"
DERIVED_PCIE = "PCIe Total [Derived Throughput %]"

KEY_METRICS = {
    "gr_active": GR_ACTIVE,
    "sm_active": SM_ACTIVE,
    "sm_issue": SM_ISSUE,
    "tensor_active": TENSOR_ACTIVE,
    "dram_read": DRAM_READ,
    "dram_write": DRAM_WRITE,
    "dram_total": DERIVED_DRAM,
    "nvlink_rx": DERIVED_NVLINK_RX,
    "nvlink_tx": DERIVED_NVLINK_TX,
    "nvlink_duplex": DERIVED_NVLINK_DUPLEX,
    "pcie_rx": PCIE_RX,
    "pcie_tx": PCIE_TX,
    "pcie_total": DERIVED_PCIE,
}


def parse_int_spec(spec: str, minimum: int, maximum: int | None = None) -> list[int]:
    values: set[int] = set()
    for chunk in spec.split(","):
        if match := re.fullmatch(r"(\d+)-(\d+)", chunk):
            first, last = map(int, match.groups())
            if first > last:
                raise ValueError(f"descending range: {chunk}")
            values.update(range(first, last + 1))
        elif chunk.isdigit():
            values.add(int(chunk))
        else:
            raise ValueError(f"invalid integer specification: {chunk}")
    if not values or min(values) < minimum:
        raise ValueError(f"values must be >= {minimum}")
    if maximum is not None and max(values) > maximum:
        raise ValueError(f"values must be <= {maximum}")
    return sorted(values)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover_reports(root: Path) -> dict[tuple[int, int], Path]:
    reports: dict[tuple[int, int], Path] = {}
    for report in sorted(root.rglob("q*.nsys-rep")):
        query_match = re.fullmatch(r"q(\d+)", report.stem)
        scale_match = None
        for parent in report.parents:
            scale_match = re.fullmatch(r"sf(\d+)", parent.name)
            if scale_match is not None:
                break
        if query_match is None or scale_match is None:
            continue
        key = (int(scale_match.group(1)), int(query_match.group(1)))
        if key in reports:
            raise RuntimeError(f"duplicate report for SF{key[0]} Q{key[1]}: "
                               f"{reports[key]} and {report}")
        reports[key] = report
    return reports


def export_sqlite(report: Path, output: Path) -> None:
    result = subprocess.run(
        ["nsys", "export", "--type", "sqlite", "--force-overwrite=true",
         "--output", str(output), str(report)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nsys export failed for {report}:\n{result.stdout}")


def profile_window(
    connection: sqlite3.Connection,
) -> tuple[int, int, list[int], int, int, int]:
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
    if not ranges:
        raise RuntimeError(f"missing completed {PROFILE_RANGE} NVTX ranges")
    if max(row[0] for row in ranges) >= min(row[1] for row in ranges):
        raise RuntimeError(f"rank {PROFILE_RANGE} ranges do not overlap")
    start = min(row[0] for row in ranges)
    end = max(row[1] for row in ranges)
    if start >= end:
        raise RuntimeError(f"invalid rank {PROFILE_RANGE} ranges")
    durations = [row[1] - row[0] for row in ranges]
    start_skew = max(row[0] for row in ranges) - min(row[0] for row in ranges)
    end_skew = max(row[1] for row in ranges) - min(row[1] for row in ranges)
    return start, end, durations, start_skew, end_skew, len(ranges)


def add_derived_metrics(metrics: dict[str, list[float]]) -> None:
    sample_count = len(next(iter(metrics.values())))
    for name, values in metrics.items():
        if len(values) != sample_count:
            raise RuntimeError(
                f"metric sample count mismatch for {name}: "
                f"{len(values)} != {sample_count}")

    def sum_metrics(names: Iterable[str]) -> list[float]:
        missing = [name for name in names if name not in metrics]
        if missing:
            raise RuntimeError(f"missing required GPU metrics: {', '.join(missing)}")
        return [sum(values) for values in zip(*(metrics[name] for name in names))]

    metrics[DERIVED_DRAM] = sum_metrics((DRAM_READ, DRAM_WRITE))
    metrics[DERIVED_NVLINK_RX] = sum_metrics(NVLINK_RX)
    metrics[DERIVED_NVLINK_TX] = sum_metrics(NVLINK_TX)
    metrics[DERIVED_NVLINK_DUPLEX] = [
        (rx + tx) / 2.0
        for rx, tx in zip(metrics[DERIVED_NVLINK_RX], metrics[DERIVED_NVLINK_TX])
    ]
    metrics[DERIVED_PCIE] = sum_metrics((PCIE_RX, PCIE_TX))


def summarize_report(
    report: Path,
    scale: int,
    query: int,
    sqlite_path: Path,
    nvlink_unidirectional_gbps: float | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    export_sqlite(report, sqlite_path)
    connection = sqlite3.connect(sqlite_path)
    try:
        (start, end, rank_durations, rank_start_skew, rank_end_skew,
         profile_rank_ranges) = profile_window(connection)
        gpu_info = {
            row[0]: {
                "gpu_name": row[1],
                "gpu_uuid": row[2],
                "dram_peak_bytes_per_second": row[3],
            }
            for row in connection.execute(
                "SELECT id, name, uuid, memoryBandwidth FROM TARGET_INFO_GPU")
        }
        if profile_rank_ranges != len(gpu_info):
            raise RuntimeError(
                f"found {profile_rank_ranges} completed {PROFILE_RANGE} ranges "
                f"for {len(gpu_info)} GPUs")
        samples: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list))
        rows = connection.execute(
            """
            SELECT m.typeId, m.metricName, g.value
            FROM GPU_METRICS g
            JOIN TARGET_INFO_GPU_METRICS m
              ON m.typeId = g.typeId AND m.metricId = g.metricId
            WHERE g.rawTimestamp BETWEEN ? AND ?
            ORDER BY m.typeId, m.metricId, g.rawTimestamp
            """,
            (start, end),
        )
        for type_id, metric_name, value in rows:
            numeric_value = float(value)
            # NSYS stores these two counters in Hz although their display names
            # declare MHz.
            if metric_name in (GPC_CLOCK, SYS_CLOCK):
                numeric_value /= 1e6
            samples[type_id & 0xFFFFFFFF][metric_name].append(numeric_value)
    finally:
        connection.close()

    expected_gpus = set(gpu_info)
    if set(samples) != expected_gpus:
        raise RuntimeError(
            f"GPU metric streams {sorted(samples)} do not match GPU inventory "
            f"{sorted(expected_gpus)}")

    long_rows: list[dict[str, object]] = []
    gpu_rows: list[dict[str, object]] = []
    aggregate: dict[str, list[float]] = defaultdict(list)
    for gpu in sorted(samples):
        metrics = samples[gpu]
        add_derived_metrics(metrics)
        for metric_name in sorted(metrics):
            summary = stats(metrics[metric_name])
            long_rows.append({
                "scale": scale,
                "query": query,
                "gpu": gpu,
                "metric": metric_name,
                **summary,
                "report": str(report),
            })

        row: dict[str, object] = {
            "scale": scale,
            "query": query,
            "gpu": gpu,
            "gpu_name": gpu_info[gpu]["gpu_name"],
            "gpu_uuid": gpu_info[gpu]["gpu_uuid"],
            "profile_rank_ranges": profile_rank_ranges,
            "profile_envelope_ms": (end - start) / 1e6,
            "rank_profile_min_ms": min(rank_durations) / 1e6,
            "rank_profile_max_ms": max(rank_durations) / 1e6,
            "rank_start_skew_ms": rank_start_skew / 1e6,
            "rank_end_skew_ms": rank_end_skew / 1e6,
            "sample_count": len(metrics[SM_ACTIVE]),
        }
        for prefix, metric_name in KEY_METRICS.items():
            summary = stats(metrics[metric_name])
            for statistic in ("mean", "p50", "p95", "max"):
                row[f"{prefix}_{statistic}_pct"] = summary[statistic]
            aggregate[metric_name].extend(metrics[metric_name])

        dram_peak_gbps = gpu_info[gpu]["dram_peak_bytes_per_second"] / 1e9
        row["dram_peak_gbps"] = dram_peak_gbps
        row["dram_mean_gbps"] = (
            float(row["dram_total_mean_pct"]) * dram_peak_gbps / 100.0)
        if nvlink_unidirectional_gbps is not None:
            row["nvlink_unidirectional_peak_gbps"] = nvlink_unidirectional_gbps
            row["nvlink_rx_mean_gbps"] = (
                float(row["nvlink_rx_mean_pct"]) * nvlink_unidirectional_gbps / 100.0)
            row["nvlink_tx_mean_gbps"] = (
                float(row["nvlink_tx_mean_pct"]) * nvlink_unidirectional_gbps / 100.0)
        else:
            row["nvlink_unidirectional_peak_gbps"] = ""
            row["nvlink_rx_mean_gbps"] = ""
            row["nvlink_tx_mean_gbps"] = ""
        gpu_rows.append(row)

    query_row: dict[str, object] = {
        "scale": scale,
        "query": query,
        "gpus": len(samples),
        "profile_rank_ranges": profile_rank_ranges,
        "profile_envelope_ms": (end - start) / 1e6,
        "rank_profile_min_ms": min(rank_durations) / 1e6,
        "rank_profile_max_ms": max(rank_durations) / 1e6,
        "rank_start_skew_ms": rank_start_skew / 1e6,
        "rank_end_skew_ms": rank_end_skew / 1e6,
        "samples_per_gpu_min": min(row["sample_count"] for row in gpu_rows),
        "samples_per_gpu_max": max(row["sample_count"] for row in gpu_rows),
    }
    for prefix, metric_name in KEY_METRICS.items():
        summary = stats(aggregate[metric_name])
        for statistic in ("mean", "p50", "p95", "max"):
            query_row[f"{prefix}_{statistic}_pct"] = summary[statistic]
    query_row["dram_aggregate_mean_gbps"] = sum(
        float(row["dram_mean_gbps"]) for row in gpu_rows)
    if nvlink_unidirectional_gbps is not None:
        query_row["nvlink_endpoint_aggregate_rx_mean_gbps"] = sum(
            float(row["nvlink_rx_mean_gbps"]) for row in gpu_rows)
        query_row["nvlink_endpoint_aggregate_tx_mean_gbps"] = sum(
            float(row["nvlink_tx_mean_gbps"]) for row in gpu_rows)
    else:
        query_row["nvlink_endpoint_aggregate_rx_mean_gbps"] = ""
        query_row["nvlink_endpoint_aggregate_tx_mean_gbps"] = ""
    query_row["report"] = str(report)
    return long_rows, gpu_rows, query_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scales", default="100,300")
    parser.add_argument("--queries", default="1-22")
    parser.add_argument(
        "--nvlink-unidirectional-gbps",
        type=float,
        help="aggregate one-direction peak over all active links on one GPU",
    )
    args = parser.parse_args()

    try:
        scales = parse_int_spec(args.scales, 1)
        queries = parse_int_spec(args.queries, 1, 22)
    except ValueError as error:
        parser.error(str(error))
    output_dir = args.output_dir or args.input_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = discover_reports(args.input_dir)
    expected = {(scale, query) for scale in scales for query in queries}
    missing = sorted(expected - set(reports))
    unexpected = sorted(set(reports) - expected)
    if missing:
        formatted = ", ".join(f"SF{s}/Q{q}" for s, q in missing)
        raise SystemExit(f"error: missing reports: {formatted}")
    if unexpected:
        formatted = ", ".join(f"SF{s}/Q{q}" for s, q in unexpected)
        raise SystemExit(f"error: unexpected reports: {formatted}")

    all_long_rows: list[dict[str, object]] = []
    all_gpu_rows: list[dict[str, object]] = []
    all_query_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="tpch-nsys-") as temporary:
        temporary_dir = Path(temporary)
        for scale, query in sorted(expected):
            report = reports[(scale, query)]
            print(f"Summarizing SF{scale} Q{query}: {report}", flush=True)
            long_rows, gpu_rows, query_row = summarize_report(
                report,
                scale,
                query,
                temporary_dir / "report.sqlite",
                args.nvlink_unidirectional_gbps,
            )
            all_long_rows.extend(long_rows)
            all_gpu_rows.extend(gpu_rows)
            all_query_rows.append(query_row)

    write_csv(
        output_dir / "hardware_metrics_long.csv",
        all_long_rows,
        ["scale", "query", "gpu", "metric", "samples", "mean", "p50",
         "p95", "max", "report"],
    )
    gpu_fields = list(all_gpu_rows[0])
    write_csv(output_dir / "hardware_utilization_per_gpu.csv", all_gpu_rows, gpu_fields)
    query_fields = list(all_query_rows[0])
    write_csv(
        output_dir / "hardware_utilization_per_query.csv",
        all_query_rows,
        query_fields,
    )
    print(f"Wrote {len(all_query_rows)} query and {len(all_gpu_rows)} GPU summaries "
          f"to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
