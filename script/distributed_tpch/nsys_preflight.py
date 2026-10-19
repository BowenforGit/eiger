#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Select an Nsight Systems GPU-metrics device with a bounded real capture."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


class PreflightError(RuntimeError):
    pass


def physical_gpu_candidates(cuda_visible_devices: str, expected: int) -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        index_text, separator, uuid = line.partition(",")
        if not separator or not index_text.strip().isdigit():
            raise PreflightError(f"unexpected nvidia-smi GPU row: {line!r}")
        inventory.append((int(index_text.strip()), uuid.strip()))

    if cuda_visible_devices == "<unset>":
        candidates = [index for index, _ in inventory]
    else:
        candidates = []
        for token in cuda_visible_devices.split(","):
            token = token.strip()
            if token.isdigit():
                candidates.append(int(token))
                continue
            matches = [
                index
                for index, uuid in inventory
                if uuid == token or uuid.startswith(token) or token.startswith(uuid)
            ]
            if len(matches) != 1:
                raise PreflightError(
                    f"cannot map CUDA_VISIBLE_DEVICES token {token!r} to one GPU"
                )
            candidates.append(matches[0])
    if len(candidates) != expected:
        raise PreflightError(
            f"resolved {len(candidates)} physical GPUs, expected {expected}: {candidates}"
        )
    return candidates


def run_bounded(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    start = time.monotonic()
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
        "elapsed_seconds": time.monotonic() - start,
        "output": stdout[-8000:],
    }


def metric_row_counts(database: Path) -> tuple[int, int]:
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "GPU_METRICS" not in tables or "TARGET_INFO_GPU_METRICS" not in tables:
            return 0, 0
        samples = connection.execute("SELECT COUNT(*) FROM GPU_METRICS").fetchone()[0]
        definitions = connection.execute(
            "SELECT COUNT(*) FROM TARGET_INFO_GPU_METRICS"
        ).fetchone()[0]
    return int(samples), int(definitions)


def probe_gpu(
    gpu_index: int,
    timeout_seconds: int,
    sample_seconds: int,
    frequency: int,
    nsys: str = "nsys",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"eiger-nsys-gpu{gpu_index}-") as temp:
        root = Path(temp)
        report_prefix = root / "metrics"
        capture = run_bounded(
            [
                nsys,
                "profile",
                "--trace=none",
                "--sample=none",
                "--cpuctxsw=none",
                "--force-overwrite=true",
                f"--gpu-metrics-devices={gpu_index}",
                f"--gpu-metrics-frequency={frequency}",
                "--output",
                str(report_prefix),
                "/bin/sleep",
                str(sample_seconds),
            ],
            timeout_seconds,
        )
        result: dict[str, Any] = {"gpu_index": gpu_index, "capture": capture}
        report = report_prefix.with_suffix(".nsys-rep")
        if capture["timed_out"] or capture["exit_code"] != 0 or not report.is_file():
            result.update(status="fail", reason="capture failed or timed out")
            return result

        database = root / "metrics.sqlite"
        exported = run_bounded(
            [
                nsys,
                "export",
                "--type=sqlite",
                "--force-overwrite=true",
                "--output",
                str(database),
                str(report),
            ],
            timeout_seconds,
        )
        result["export"] = exported
        if (
            exported["timed_out"]
            or exported["exit_code"] != 0
            or not database.is_file()
        ):
            result.update(status="fail", reason="report export failed or timed out")
            return result
        samples, definitions = metric_row_counts(database)
        result.update(metric_samples=samples, metric_definitions=definitions)
        if samples < 1 or definitions < 1:
            result.update(status="fail", reason="GPU metric tables are empty")
            return result
        result.update(status="pass", reason="nonempty GPU metrics captured")
        return result


def select_gpu(
    candidates: list[int],
    timeout_seconds: int,
    sample_seconds: int,
    frequency: int,
    nsys: str = "nsys",
) -> dict[str, Any]:
    attempts = []
    for gpu_index in candidates:
        attempt = probe_gpu(gpu_index, timeout_seconds, sample_seconds, frequency, nsys)
        attempts.append(attempt)
        if attempt["status"] == "pass":
            return {"status": "pass", "gpu_index": gpu_index, "attempts": attempts}
    return {"status": "fail", "gpu_index": None, "attempts": attempts}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--candidates")
    selection.add_argument(
        "--expected-gpus",
        type=positive_int,
        help="resolve candidates from local inventory and CUDA visibility",
    )
    parser.add_argument("--cuda-visible-devices", default="<unset>")
    parser.add_argument("--timeout-seconds", type=positive_int, default=20)
    parser.add_argument("--sample-seconds", type=positive_int, default=2)
    parser.add_argument("--frequency", type=positive_int, default=10000)
    args = parser.parse_args()
    if args.candidates is not None:
        try:
            candidates = [int(token) for token in args.candidates.split(",")]
            if not candidates or any(index < 0 for index in candidates):
                raise ValueError
        except ValueError:
            parser.error("--candidates must be comma-separated nonnegative GPU indices")
    else:
        try:
            candidates = physical_gpu_candidates(
                args.cuda_visible_devices, args.expected_gpus
            )
        except (OSError, subprocess.CalledProcessError, PreflightError) as error:
            parser.error(f"cannot resolve physical GPU candidates: {error}")
    result = select_gpu(
        candidates,
        args.timeout_seconds,
        args.sample_seconds,
        args.frequency,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
