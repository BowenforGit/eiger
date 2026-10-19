#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Native-section preflight and report validation for NCU captures."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


SECTION_IDENTIFIER = "AlpsTpchMetrics"
NCCL_EXCLUDING_KERNEL_EXPRESSION = "regex:^(?!.*[Nn][Cc][Cc][Ll]).*$"
NVTX_INCLUDE = "gamax.tpch@tpch.profile/"
_LAUNCH_METRIC_RE = re.compile(r"^launch__[A-Za-z0-9_.]+$")
_METRIC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]*$")


class NcuProfileError(RuntimeError):
    pass


def section_arguments(section_file: Path) -> list[str]:
    return [
        "--section-folder",
        str(section_file.parent),
        "--section",
        SECTION_IDENTIFIER,
    ]


def parse_kernel_substrings(path: Path) -> tuple[str, ...]:
    try:
        values: list[str] = []
        for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
            value = raw.strip()
            if not value or value.startswith("#"):
                continue
            if any(character in value for character in ("\x00", "\r", "\n")):
                raise NcuProfileError(
                    f"invalid kernel substring at {path}:{line_number}"
                )
            if value in values:
                raise NcuProfileError(f"duplicate kernel substring in {path}: {value}")
            values.append(value)
    except OSError as error:
        raise NcuProfileError(f"cannot read NCU kernels file {path}: {error}") from error
    if not values:
        raise NcuProfileError(f"NCU kernels file is empty: {path}")
    return tuple(values)


def kernel_name_expression(kernel_substrings: Iterable[str]) -> str:
    escaped = "|".join(re.escape(value) for value in kernel_substrings)
    if not escaped:
        return NCCL_EXCLUDING_KERNEL_EXPRESSION
    return f"regex:^(?!.*[Nn][Cc][Cc][Ll]).*({escaped}).*$"


def run_bounded(
    command: list[str], timeout_seconds: int, *, preserve_output: bool = False
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
    return {
        "argv": command,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "output": stdout if preserve_output else stdout[-16000:],
    }


def query_section_metrics(
    section_file: Path, ncu: str, timeout_seconds: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not section_file.is_file():
        raise NcuProfileError(f"NCU section file does not exist: {section_file}")
    result = run_bounded(
        [
            ncu,
            *section_arguments(section_file),
            "--disable-extra-suffixes",
            "--list-metrics",
        ],
        timeout_seconds,
        preserve_output=True,
    )
    if result["timed_out"] or result["exit_code"] != 0:
        raise NcuProfileError(
            f"could not load NCU section {section_file}: "
            + str(result["output"]).strip()
        )
    selectors = tuple(
        dict.fromkeys(
            line.strip()
            for line in str(result["output"]).splitlines()
            if line.strip()
            and not line.strip().startswith("-")
            and (
                line.strip().startswith("regex:")
                or _METRIC_RE.fullmatch(line.strip())
            )
        )
    )
    requested = tuple(
        selector for selector in selectors if not selector.startswith("launch__")
    )
    launch = tuple(
        selector for selector in selectors if selector.startswith("launch__")
    )
    if not requested:
        raise NcuProfileError(f"NCU section has no counter selectors: {section_file}")
    if not launch:
        raise NcuProfileError(f"NCU section has no launch__ metrics: {section_file}")
    return requested, launch


def selector_matches(selector: str, metric: str) -> bool:
    if selector.startswith("regex:"):
        try:
            return re.search(selector.removeprefix("regex:"), metric) is not None
        except re.error as error:
            raise NcuProfileError(
                f"invalid metric selector {selector!r}: {error}"
            ) from error
    return metric == selector


def inspect_report(
    report: Path,
    requested_metrics: Iterable[str],
    launch_metrics: Iterable[str],
    timeout_seconds: int,
    ncu: str = "ncu",
    kernel_substrings: Iterable[str] = (),
) -> tuple[dict[str, Any], str]:
    imported = run_bounded(
        [
            ncu,
            "--import",
            str(report),
            "--page",
            "details",
            "--csv",
            "--print-metric-name",
            "name",
            "--print-details",
            "all",
            "--print-units",
            "base",
        ],
        timeout_seconds,
        preserve_output=True,
    )
    import_metadata = {key: value for key, value in imported.items() if key != "output"}
    import_metadata["output_bytes"] = len(str(imported["output"]).encode())
    document: dict[str, Any] = {
        "report": str(report),
        "import": import_metadata,
        "status": "fail",
    }
    output = str(imported["output"])
    if imported["timed_out"] or imported["exit_code"] != 0:
        document["reason"] = "report import failed or timed out"
        return document, output

    output_lines = output.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(output_lines)
            if "Kernel Name" in line and "Metric Name" in line
        ),
        None,
    )
    if header_index is None:
        document["reason"] = "raw report CSV lacks a kernel/metric header"
        return document, output
    reader = csv.DictReader(io.StringIO("\n".join(output_lines[header_index:])))
    rows = list(reader)
    if not rows or reader.fieldnames is None:
        document["reason"] = "raw report CSV is empty"
        return document, output
    kernel_column = next(
        (name for name in reader.fieldnames if name.strip().lower() == "kernel name"),
        None,
    )
    metric_column = next(
        (name for name in reader.fieldnames if name.strip().lower() == "metric name"),
        None,
    )
    if kernel_column is None or metric_column is None:
        document["reason"] = "raw report CSV lacks kernel or metric columns"
        document["columns"] = reader.fieldnames
        return document, output

    kernels = sorted(
        {row.get(kernel_column, "") for row in rows if row.get(kernel_column)}
    )
    observed_metrics = {
        row.get(metric_column, "") for row in rows if row.get(metric_column)
    }
    nccl_kernels = [name for name in kernels if "nccl" in name.lower()]
    selected = tuple(kernel_substrings)
    unexpected_kernels = [
        name
        for name in kernels
        if selected and not any(value in name for value in selected)
    ]
    missing_requested = [
        selector
        for selector in requested_metrics
        if not any(selector_matches(selector, observed) for observed in observed_metrics)
    ]
    missing_launch = [
        metric for metric in launch_metrics if metric not in observed_metrics
    ]
    document.update(
        kernel_count=len(kernels),
        kernels=kernels,
        observed_metric_count=len(observed_metrics),
        nccl_kernels=nccl_kernels,
        unexpected_kernels=unexpected_kernels,
        missing_requested_metrics=missing_requested,
        missing_launch_metrics=missing_launch,
    )
    if not kernels:
        document["reason"] = "report contains no kernels"
    elif nccl_kernels:
        document["reason"] = "report contains excluded NCCL kernels"
    elif unexpected_kernels:
        document["reason"] = "report contains kernels outside the selected subset"
    elif missing_requested:
        document["reason"] = "report is missing requested metrics"
    elif missing_launch:
        document["reason"] = "report is missing launch__ metrics"
    else:
        document.update(status="pass", reason="report and metric inventory validated")
    return document, output


def run_preflight(
    probe: Path,
    section_file: Path,
    timeout_seconds: int,
    ncu: str = "ncu",
    kernel_substrings: Iterable[str] = (),
) -> dict[str, Any]:
    selected_kernels = tuple(kernel_substrings)
    metrics, launch_metrics = query_section_metrics(
        section_file, ncu, timeout_seconds
    )
    with tempfile.TemporaryDirectory(prefix="alps-tpch-ncu-") as temporary:
        report_prefix = Path(temporary) / "counter_probe"
        capture = run_bounded(
            [
                ncu,
                "--nvtx",
                "--nvtx-push-pop-scope",
                "process",
                "--nvtx-include",
                NVTX_INCLUDE,
                "--kernel-name-base",
                "demangled",
                "--kernel-name",
                NCCL_EXCLUDING_KERNEL_EXPRESSION,
                *section_arguments(section_file),
                "--disable-extra-suffixes",
                "--launch-count",
                "1",
                "--clock-control",
                "none",
                "--target-processes",
                "application-only",
                "--force-overwrite",
                "--export",
                str(report_prefix),
                str(probe),
            ],
            timeout_seconds,
        )
        result: dict[str, Any] = {
            "status": "fail",
            "ncu": ncu,
            "section_file": str(section_file),
            "kernels": list(selected_kernels),
            "kernel_expression": kernel_name_expression(selected_kernels),
            "metrics": list(metrics),
            "launch_metrics": list(launch_metrics),
            "capture": capture,
        }
        report = report_prefix.with_suffix(".ncu-rep")
        if capture["timed_out"] or capture["exit_code"] != 0 or not report.is_file():
            output = str(capture["output"])
            result["reason"] = (
                "performance counters are not accessible"
                if re.search(
                    r"ERR_NVGPUCTRPERM|insufficient permission|"
                    r"does not have permission",
                    output,
                    re.IGNORECASE,
                )
                else "standalone NCU capture failed or timed out"
            )
            return result
        validation, _ = inspect_report(
            report, metrics, launch_metrics, timeout_seconds, ncu
        )
        result["validation"] = validation
        if validation["status"] != "pass":
            result["reason"] = validation["reason"]
            return result
        version = run_bounded([ncu, "--version"], timeout_seconds)
        result.update(
            status="pass",
            reason="standalone NCU capture and report validation passed",
            version=str(version["output"]).strip(),
        )
        return result


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--section-file", type=Path, required=True)
    parser.add_argument("--kernels-file", type=Path)
    parser.add_argument("--timeout-seconds", type=positive_int, default=60)
    parser.add_argument("--ncu", default="ncu")
    args = parser.parse_args()
    try:
        kernel_substrings = (
            parse_kernel_substrings(args.kernels_file) if args.kernels_file else ()
        )
        result = run_preflight(
            args.probe,
            args.section_file,
            args.timeout_seconds,
            args.ncu,
            kernel_substrings,
        )
    except (OSError, NcuProfileError) as error:
        result = {"status": "fail", "reason": str(error)}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
