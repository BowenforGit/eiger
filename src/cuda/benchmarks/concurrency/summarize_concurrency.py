#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Regenerate concurrency summary CSV from preserved raw samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable


WHOLE_PREFIXES = (
    "a_alone",
    "b_alone",
    "sequential_ab",
    "sequential_ba",
    "concurrent_makespan",
    "concurrent_a",
    "concurrent_b",
    "start_skew",
)

CHUNKED_PREFIXES = (
    "chunked_a_alone",
    "chunked_b_alone",
    "chunked_concurrent_makespan",
    "chunked_concurrent_a",
    "chunked_concurrent_b",
    "chunked_start_skew",
)

PREFIXES = WHOLE_PREFIXES + CHUNKED_PREFIXES

BASE_FIELDS = [
    "run_id",
    "timestamp_utc",
    "git_commit",
    "git_dirty",
    "gpu_name",
    "device_id",
    "op_a",
    "op_b",
    "config_a",
    "config_b",
    "a_exponent",
    "b_exponent",
    "a_size",
    "b_size",
    "a_chunk_size",
    "b_chunk_size",
    "a_chunk_count",
    "b_chunk_count",
    "status",
    "reason",
    "stable",
]

DIST_FIELDS = [
    f"{prefix}_{suffix}"
    for prefix in PREFIXES
    for suffix in ("n", "median_ms", "mean_ms", "stddev_ms", "p5_ms", "p95_ms")
]

WHOLE_RATIO_FIELDS = [
    "h_ideal",
    "h_ideal_ci_low",
    "h_ideal_ci_high",
    "h_sequential",
    "h_sequential_ci_low",
    "h_sequential_ci_high",
    "slowdown_a",
    "slowdown_a_ci_low",
    "slowdown_a_ci_high",
    "slowdown_b",
    "slowdown_b_ci_low",
    "slowdown_b_ci_high",
    "sequential_ab_vs_isolated_sum",
    "sequential_ba_vs_isolated_sum",
    "material_start_skew_count",
    "max_start_skew_fraction",
]

CHUNKED_RATIO_FIELDS = [
    "chunked_isolated_slowdown_a",
    "chunked_isolated_slowdown_a_ci_low",
    "chunked_isolated_slowdown_a_ci_high",
    "chunked_isolated_slowdown_b",
    "chunked_isolated_slowdown_b_ci_low",
    "chunked_isolated_slowdown_b_ci_high",
    "chunked_concurrent_speedup",
    "chunked_concurrent_speedup_ci_low",
    "chunked_concurrent_speedup_ci_high",
    "chunked_interference_slowdown_a",
    "chunked_interference_slowdown_b",
    "chunked_concurrent_a_vs_whole",
    "chunked_concurrent_b_vs_whole",
    "chunked_material_start_skew_count",
    "chunked_max_start_skew_fraction",
]

RATIO_FIELDS = WHOLE_RATIO_FIELDS + CHUNKED_RATIO_FIELDS

FIELDS = BASE_FIELDS + DIST_FIELDS + RATIO_FIELDS


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "stddev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p5_ms": quantile(values, 0.05),
        "p95_ms": quantile(values, 0.95),
    }


def bootstrap(
    a: list[float],
    b: list[float],
    c: list[float],
    iterations: int,
    seed: int,
    ratio: Callable[[float, float, float], float],
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        aa = [rng.choice(a) for _ in a]
        bb = [rng.choice(b) for _ in b]
        cc = [rng.choice(c) for _ in c]
        estimates.append(
            ratio(statistics.median(aa), statistics.median(bb), statistics.median(cc))
        )
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def measured(
    rows: list[dict[str, str]], mode: str, field: str, variant: str = "whole"
) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if row["success"] == "1"
        and row["warmup"] == "0"
        and row["mode"] == mode
        and row.get("execution_variant", "whole") == variant
    ]


def base_row(first: dict[str, str]) -> dict[str, str]:
    result = {field: first.get(field, "") for field in BASE_FIELDS[:18]}
    # Raw files from before chunking did not contain these four fields. Regenerate
    # them as the equivalent one-chunk configuration.
    result["a_chunk_size"] = result["a_chunk_size"] or result["a_size"]
    result["b_chunk_size"] = result["b_chunk_size"] or result["b_size"]
    result["a_chunk_count"] = result["a_chunk_count"] or "1"
    result["b_chunk_count"] = result["b_chunk_count"] or "1"
    return result


def empty_distribution() -> dict[str, float | int]:
    return {
        "n": 0,
        "median_ms": math.nan,
        "mean_ms": math.nan,
        "stddev_ms": math.nan,
        "p5_ms": math.nan,
        "p95_ms": math.nan,
    }


def start_skew_fractions(rows: list[dict[str, str]], variant: str) -> list[float]:
    concurrent_rows = [
        row
        for row in rows
        if row["success"] == "1"
        and row["warmup"] == "0"
        and row["mode"] == "concurrent"
        and row.get("execution_variant", "whole") == variant
    ]
    return [
        float(row["start_skew_ms"]) / min(float(row["a_ms"]), float(row["b_ms"]))
        for row in concurrent_rows
    ]


def summarize_cell(
    rows: list[dict[str, str]], iterations: int, threshold: float
) -> dict[str, str | int | float]:
    first = rows[0]
    result: dict[str, str | int | float] = base_row(first)
    failure = next((row for row in rows if row["success"] != "1"), None)
    if failure:
        result.update(
            status=failure["enqueue_order"], reason=failure["reason"], stable=0
        )
        return result

    chunking_enabled = any(
        row.get("execution_variant", "whole") == "chunked" for row in rows
    )
    values = {
        "a_alone": measured(rows, "a_isolated", "pair_makespan_ms"),
        "b_alone": measured(rows, "b_isolated", "pair_makespan_ms"),
        "sequential_ab": measured(rows, "sequential_a_then_b", "pair_makespan_ms"),
        "sequential_ba": measured(rows, "sequential_b_then_a", "pair_makespan_ms"),
        "concurrent_makespan": measured(rows, "concurrent", "pair_makespan_ms"),
        "concurrent_a": measured(rows, "concurrent", "a_ms"),
        "concurrent_b": measured(rows, "concurrent", "b_ms"),
        "start_skew": measured(rows, "concurrent", "start_skew_ms"),
    }
    if any(not population for population in values.values()):
        result.update(
            status="failed_incomplete_raw",
            reason="missing measured mode samples",
            stable=0,
        )
        return result

    chunked_values: dict[str, list[float]] = {}
    if chunking_enabled:
        chunked_values = {
            "chunked_a_alone": measured(
                rows, "a_isolated", "pair_makespan_ms", "chunked"
            ),
            "chunked_b_alone": measured(
                rows, "b_isolated", "pair_makespan_ms", "chunked"
            ),
            "chunked_concurrent_makespan": measured(
                rows, "concurrent", "pair_makespan_ms", "chunked"
            ),
            "chunked_concurrent_a": measured(rows, "concurrent", "a_ms", "chunked"),
            "chunked_concurrent_b": measured(rows, "concurrent", "b_ms", "chunked"),
            "chunked_start_skew": measured(
                rows, "concurrent", "start_skew_ms", "chunked"
            ),
        }
        if any(not population for population in chunked_values.values()):
            result.update(
                status="failed_incomplete_raw",
                reason="missing measured chunked mode samples",
                stable=0,
            )
            return result

    result.update(status="success", reason="")
    all_values = values | chunked_values
    for prefix in PREFIXES:
        population = all_values.get(prefix)
        summary = distribution(population) if population else empty_distribution()
        for suffix, value in summary.items():
            result[f"{prefix}_{suffix}"] = value

    med = {key: statistics.median(value) for key, value in all_values.items()}
    result["h_ideal"] = med["concurrent_makespan"] / max(med["a_alone"], med["b_alone"])
    result["h_sequential"] = med["concurrent_makespan"] / (
        med["a_alone"] + med["b_alone"]
    )
    result["slowdown_a"] = med["concurrent_a"] / med["a_alone"]
    result["slowdown_b"] = med["concurrent_b"] / med["b_alone"]
    isolated_sum = med["a_alone"] + med["b_alone"]
    result["sequential_ab_vs_isolated_sum"] = med["sequential_ab"] / isolated_sum
    result["sequential_ba_vs_isolated_sum"] = med["sequential_ba"] / isolated_sum

    if chunking_enabled:
        result["chunked_isolated_slowdown_a"] = med["chunked_a_alone"] / med["a_alone"]
        result["chunked_isolated_slowdown_b"] = med["chunked_b_alone"] / med["b_alone"]
        result["chunked_concurrent_speedup"] = (
            med["concurrent_makespan"] / med["chunked_concurrent_makespan"]
        )
        result["chunked_interference_slowdown_a"] = (
            med["chunked_concurrent_a"] / med["chunked_a_alone"]
        )
        result["chunked_interference_slowdown_b"] = (
            med["chunked_concurrent_b"] / med["chunked_b_alone"]
        )
        result["chunked_concurrent_a_vs_whole"] = (
            med["chunked_concurrent_a"] / med["concurrent_a"]
        )
        result["chunked_concurrent_b_vs_whole"] = (
            med["chunked_concurrent_b"] / med["concurrent_b"]
        )
    else:
        for field in CHUNKED_RATIO_FIELDS[:-2]:
            result[field] = math.nan

    cell_seed = int(first["random_seed"])
    ci_specs = [
        (
            "h_ideal",
            values["a_alone"],
            values["b_alone"],
            values["concurrent_makespan"],
            11,
            lambda x, y, z: z / max(x, y),
        ),
        (
            "h_sequential",
            values["a_alone"],
            values["b_alone"],
            values["concurrent_makespan"],
            23,
            lambda x, y, z: z / (x + y),
        ),
        (
            "slowdown_a",
            values["a_alone"],
            [1.0],
            values["concurrent_a"],
            37,
            lambda x, _y, z: z / x,
        ),
        (
            "slowdown_b",
            values["b_alone"],
            [1.0],
            values["concurrent_b"],
            53,
            lambda x, _y, z: z / x,
        ),
    ]
    if chunking_enabled:
        ci_specs.extend(
            [
                (
                    "chunked_isolated_slowdown_a",
                    values["a_alone"],
                    [1.0],
                    chunked_values["chunked_a_alone"],
                    67,
                    lambda x, _y, z: z / x,
                ),
                (
                    "chunked_isolated_slowdown_b",
                    values["b_alone"],
                    [1.0],
                    chunked_values["chunked_b_alone"],
                    79,
                    lambda x, _y, z: z / x,
                ),
                (
                    "chunked_concurrent_speedup",
                    values["concurrent_makespan"],
                    [1.0],
                    chunked_values["chunked_concurrent_makespan"],
                    97,
                    lambda x, _y, z: x / z,
                ),
            ]
        )
    for name, a, b, c, offset, function in ci_specs:
        low, high = bootstrap(a, b, c, iterations, cell_seed + offset, function)
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high

    fractions = start_skew_fractions(rows, "whole")
    result["material_start_skew_count"] = sum(value > 0.05 for value in fractions)
    result["max_start_skew_fraction"] = max(fractions)

    if chunking_enabled:
        chunked_fractions = start_skew_fractions(rows, "chunked")
        result["chunked_material_start_skew_count"] = sum(
            value > 0.05 for value in chunked_fractions
        )
        result["chunked_max_start_skew_fraction"] = max(chunked_fractions)
    else:
        result["chunked_material_start_skew_count"] = 0
        result["chunked_max_start_skew_fraction"] = 0.0

    def half_width(name: str) -> float:
        return (float(result[f"{name}_ci_high"]) - float(result[f"{name}_ci_low"])) / (
            2.0 * abs(float(result[name]))
        )

    chunked_converged = not chunking_enabled or all(
        half_width(name) <= threshold
        for name in (
            "chunked_isolated_slowdown_a",
            "chunked_isolated_slowdown_b",
            "chunked_concurrent_speedup",
        )
    )
    result["stable"] = int(
        half_width("h_ideal") <= threshold
        and half_width("h_sequential") <= threshold
        and chunked_converged
        and result["material_start_skew_count"] == 0
        and result["chunked_material_start_skew_count"] == 0
    )
    return result


def default_metadata(raw: Path) -> Path:
    suffix = "_raw.csv"
    if raw.name.endswith(suffix):
        return raw.with_name(raw.name[: -len(suffix)] + "_metadata.json")
    return raw.with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    metadata_path = args.metadata or default_metadata(args.raw_csv)
    with metadata_path.open() as source:
        metadata = json.load(source)
    iterations = int(metadata["bootstrap_iterations"])
    threshold = float(metadata["ci_relative_half_width_threshold"])

    with args.raw_csv.open(newline="") as source:
        rows = list(csv.DictReader(source))
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["a_exponent"]), int(row["b_exponent"]))].append(row)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(grouped):
            writer.writerow(summarize_cell(grouped[key], iterations, threshold))


if __name__ == "__main__":
    main()
