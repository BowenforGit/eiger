#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Regression checks for native and regenerated concurrency CSV summaries."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as source:
        return list(csv.reader(source))


def assert_rectangular(path: Path) -> None:
    rows = csv_rows(path)
    if not rows:
        raise AssertionError(f"empty CSV: {path}")
    width = len(rows[0])
    malformed = [index for index, row in enumerate(rows[1:], 2) if len(row) != width]
    if malformed:
        raise AssertionError(
            f"{path} header has {width} fields; malformed rows: {malformed}"
        )
    with path.open(newline="") as source:
        for index, row in enumerate(csv.DictReader(source), 2):
            if None in row:
                raise AssertionError(
                    f"{path} row {index} has overflow fields: {row[None]}"
                )


def dict_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    assert_rectangular(path)
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def assert_close(name: str, native: str, regenerated: str) -> None:
    native_value = float(native)
    regenerated_value = float(regenerated)
    # Native derived metrics use ostream's default six significant digits.
    if not math.isclose(native_value, regenerated_value, rel_tol=2e-5, abs_tol=1e-9):
        raise AssertionError(
            f"{name} differs: native={native_value}, regenerated={regenerated_value}"
        )


def check_regeneration(
    summarizer: Path, raw: Path, metadata: Path, native_summary: Path
) -> None:
    with tempfile.TemporaryDirectory(prefix="concurrency-summary-") as directory:
        regenerated = Path(directory) / "summary.csv"
        subprocess.run(
            [
                sys.executable,
                str(summarizer),
                str(raw),
                str(regenerated),
                "--metadata",
                str(metadata),
            ],
            check=True,
        )
        native_header, native_rows = dict_rows(native_summary)
        regenerated_header, regenerated_rows = dict_rows(regenerated)

    if native_header != regenerated_header:
        raise AssertionError("native and regenerated summary headers differ")
    if len(native_rows) != len(regenerated_rows):
        raise AssertionError("native and regenerated summary row counts differ")

    exact_fields = (
        "status",
        "stable",
        "a_chunk_size",
        "b_chunk_size",
        "a_chunk_count",
        "b_chunk_count",
        "chunked_a_alone_n",
        "chunked_b_alone_n",
        "chunked_concurrent_makespan_n",
        "chunked_concurrent_a_n",
        "chunked_concurrent_b_n",
        "chunked_start_skew_n",
        "chunked_material_start_skew_count",
    )
    numeric_fields = (
        "chunked_a_alone_median_ms",
        "chunked_b_alone_median_ms",
        "chunked_concurrent_makespan_median_ms",
        "chunked_isolated_slowdown_a",
        "chunked_isolated_slowdown_b",
        "chunked_concurrent_speedup",
        "chunked_interference_slowdown_a",
        "chunked_interference_slowdown_b",
        "chunked_concurrent_a_vs_whole",
        "chunked_concurrent_b_vs_whole",
        "chunked_max_start_skew_fraction",
    )
    required_chunked_fields = [
        field for field in native_header if field.startswith("chunked_")
    ]
    for native, regenerated in zip(native_rows, regenerated_rows):
        for field in required_chunked_fields:
            if native[field] and not regenerated[field]:
                raise AssertionError(f"regenerated summary omits {field}")
        for field in exact_fields:
            if native[field] != regenerated[field]:
                raise AssertionError(
                    f"{field} differs: native={native[field]!r}, "
                    f"regenerated={regenerated[field]!r}"
                )
        for field in numeric_fields:
            assert_close(field, native[field], regenerated[field])


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    width = subparsers.add_parser("width")
    width.add_argument("csv", type=Path)

    regenerate = subparsers.add_parser("regenerate")
    regenerate.add_argument("summarizer", type=Path)
    regenerate.add_argument("raw", type=Path)
    regenerate.add_argument("metadata", type=Path)
    regenerate.add_argument("native_summary", type=Path)

    args = parser.parse_args()
    if args.command == "width":
        assert_rectangular(args.csv)
    else:
        check_regeneration(
            args.summarizer, args.raw, args.metadata, args.native_summary
        )


if __name__ == "__main__":
    main()
