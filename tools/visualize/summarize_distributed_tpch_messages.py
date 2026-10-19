#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Summarize exact NCCL API payload sizes from distributed TPC-H campaigns."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def successful_messages_csv(query_root: Path) -> Path | None:
    for attempt in sorted(query_root.glob("attempt-*"), reverse=True):
        messages = attempt / "messages.csv"
        result = attempt / "result.json"
        if not messages.is_file() or messages.stat().st_size == 0 or not result.is_file():
            continue
        try:
            verdict = json.loads(result.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if verdict.get("status") == "pass":
            return messages
    return None


def discover_inputs(paths: Iterable[Path]) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    seen_files: set[Path] = set()

    def add(label: str, messages: Path) -> None:
        messages = messages.resolve()
        if messages in seen_files:
            raise ValueError(f"duplicate messages.csv input: {messages}")
        seen_files.add(messages)
        discovered.append((label, messages))

    for supplied in paths:
        path = supplied.resolve()
        if path.is_file():
            # A basename such as "attempt-0001" is not a campaign identity:
            # every standard campaign uses it. Keep explicitly supplied CSVs
            # distinct by using their resolved paths as stable labels.
            add(str(path), path)
            continue
        direct = path / "messages.csv"
        if direct.is_file():
            add(str(path), direct)
            continue
        query_roots = sorted((path / "arms" / "legacy" / "messages").glob("q*"))
        if not query_roots:
            raise ValueError(f"no Legacy message artifacts found under {path}")
        for query_root in query_roots:
            messages = successful_messages_csv(query_root)
            if messages is not None:
                add(path.name, messages)
    if not discovered:
        raise ValueError("no messages.csv inputs found")
    return discovered


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(
    inputs: list[tuple[str, Path]], *, include_receives: bool, by_rank: bool
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for campaign, path in inputs:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "query", "ranks", "rank", "phase_name", "api", "direction",
                "component", "payload_bytes",
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
            for row in reader:
                if not include_receives and row["api"] == "ncclRecv":
                    continue
                key = (
                    campaign,
                    row["query"],
                    row["ranks"],
                    row["rank"] if by_rank else "all",
                    row["phase_name"],
                    row["api"],
                    row["direction"],
                    row["component"],
                )
                groups[key].append(int(row["payload_bytes"]))

    rows: list[dict[str, object]] = []
    for key, values in sorted(groups.items()):
        campaign, query, ranks, rank, phase, api, direction, component = key
        rows.append(
            {
                "campaign": campaign,
                "query": query,
                "ranks": int(ranks),
                "rank": rank,
                "phase_name": phase,
                "api": api,
                "direction": direction,
                "component": component,
                "message_count": len(values),
                "total_payload_bytes": sum(values),
                "mean_payload_bytes": statistics.fmean(values),
                "min_payload_bytes": min(values),
                "p50_payload_bytes": percentile(values, 0.50),
                "p95_payload_bytes": percentile(values, 0.95),
                "max_payload_bytes": max(values),
            }
        )
    return rows


def write_rows(rows: list[dict[str, object]], output: Path | None) -> None:
    fields = [
        "campaign", "query", "ranks", "rank", "phase_name", "api",
        "direction", "component", "message_count", "total_payload_bytes",
        "mean_payload_bytes", "min_payload_bytes", "p50_payload_bytes",
        "p95_payload_bytes", "max_payload_bytes",
    ]
    stream = output.open("w", newline="") if output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output:
            stream.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="campaigns or messages.csv files")
    parser.add_argument("--output", type=Path, help="write summary CSV instead of stdout")
    parser.add_argument(
        "--include-receives",
        action="store_true",
        help="include ncclRecv rows (excluded by default to avoid P2P double-counting)",
    )
    parser.add_argument(
        "--by-rank", action="store_true", help="produce separate rank-level groups"
    )
    args = parser.parse_args(argv)
    try:
        inputs = discover_inputs(args.inputs)
        rows = summarize(
            inputs, include_receives=args.include_receives, by_rank=args.by_rank
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        write_rows(rows, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
