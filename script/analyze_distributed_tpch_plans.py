#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Analyze stock-vs-synthesized distributed TPC-H artifacts.

The script has no repository imports. It reads benchmark CSV/JSON and optional
Nsight Systems reports, exports reports to SQLite when needed, and writes flat
derived CSV files suitable for the experiment archive's standalone plotter.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sqlite3
import statistics
import subprocess
from collections import defaultdict


METRICS = {
    "sm_active_pct": ["SMs Active [Throughput %]"],
    "compute_warps_pct": ["Compute Warps in Flight [Throughput %]"],
    "dram_read_pct": ["DRAM Read Bandwidth [Throughput %]"],
    "dram_write_pct": ["DRAM Write Bandwidth [Throughput %]"],
    "nvlink_rx_wire_pct": [
        "NVLink RX Requests Protocol Data [Throughput %]",
        "NVLink RX Requests User Data [Throughput %]",
        "NVLink RX Responses Protocol Data [Throughput %]",
        "NVLink RX Responses User Data [Throughput %]",
    ],
    "nvlink_tx_wire_pct": [
        "NVLink TX Requests Protocol Data [Throughput %]",
        "NVLink TX Requests User Data [Throughput %]",
        "NVLink TX Responses Protocol Data [Throughput %]",
        "NVLink TX Responses User Data [Throughput %]",
    ],
}

PHASE_ORDER = (
    "Compute", "Shuffle", "Broadcast", "Gather", "Reduce", "Sampling",
    "Other communication", "Overlap", "Overhead",
)

HASH_PARTITION_PHASE_NAMES = frozenset({
    "shuffle.hash_partition",
    "shuffle.left.hash_partition",
    "shuffle.right.hash_partition",
})


def normalized_phase_category(row):
    """Return the current taxonomy for both old and new phase CSV schemas."""
    if (
        row.get("collective_primitive") == "hash_partition"
        or row.get("phase_name") in HASH_PARTITION_PHASE_NAMES
    ):
        return "compute"
    return row["category"]


def phase_column_identity(row):
    """Read optional per-column metadata without rejecting old CSV files."""
    side = row.get("column_side") or ""
    ordinal = row.get("column_ordinal")
    return side, int(ordinal) if ordinal not in (None, "") else -1


def normalized_nvtx_phase_scope(text):
    """Normalize phase NVTX labels emitted before and after the taxonomy fix."""
    if not text.startswith("phase:"):
        return None
    phase_name = text.rsplit(":", 1)[-1]
    if phase_name in HASH_PARTITION_PHASE_NAMES:
        return "phase_compute"
    if text.startswith("phase:compute:"):
        return "phase_compute"
    if text.startswith("phase:communication:"):
        return "phase_communication"
    return None


def mechanism_nvtx_phase_scope(text):
    """Return the alignment-study communication mechanism, when applicable."""
    if text.startswith("phase:communication:shuffle:"):
        return "phase_shuffle"
    if (
        text.startswith("phase:communication:broadcast:")
        or text.endswith(":sampling.replicate")
    ):
        return "phase_replication"
    return None


def query_number(value):
    return int(str(value).lstrip("Qq"))


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def timing_summary(evaluation):
    rows = []
    by_preset = {}
    for preset in ("stock", "synthesized"):
        path = os.path.join(evaluation, preset, "timing", "e2e.csv")
        grouped = {}
        for row in read_csv(path):
            grouped.setdefault(query_number(row["query"]), []).append(
                float(row["rank_max_ms"])
            )
        by_preset[preset] = {
            query: statistics.median(values)
            for query, values in grouped.items()
        }
    common = sorted(set(by_preset["stock"]) & set(by_preset["synthesized"]))
    for query in common:
        stock = by_preset["stock"][query]
        synthesized = by_preset["synthesized"][query]
        rows.append({
            "query": query,
            "stock_median_ms": stock,
            "synthesized_median_ms": synthesized,
            "speedup": stock / synthesized if synthesized else math.nan,
            "delta_pct": 100.0 * (synthesized - stock) / stock if stock else math.nan,
        })
    if rows:
        speedups = [row["speedup"] for row in rows if row["speedup"] > 0]
        rows.append({
            "query": "aggregate",
            "stock_median_ms": sum(row["stock_median_ms"] for row in rows),
            "synthesized_median_ms": sum(
                row["synthesized_median_ms"] for row in rows
            ),
            "speedup": math.exp(
                sum(math.log(value) for value in speedups) / len(speedups)
            ),
            "delta_pct": "",
        })
    return rows


def experiment_setup(evaluation):
    records = []
    for preset in ("stock", "synthesized"):
        candidates = (
            os.path.join(evaluation, preset, "breakdown", "steps.csv"),
            os.path.join(evaluation, preset, "timing", "e2e.csv"),
        )
        source_rows = []
        for path in candidates:
            source_rows = read_csv(path)
            if source_rows:
                break
        if not source_rows:
            continue
        first = source_rows[0]
        records.append({
            "preset": preset,
            "scale": int(first["scale"]),
            "ranks": int(first["ranks"]),
            "nodes": int(first["nodes"]),
            "hostname": first["hostname"],
            "gpu_name": first["gpu_name"],
        })
    if not records:
        return []
    fields = ("scale", "ranks", "nodes", "hostname", "gpu_name")
    for field in fields:
        values = {record[field] for record in records}
        if len(values) != 1:
            raise ValueError(
                f"inconsistent experiment setup field {field}: {values}"
            )
    return [{field: records[0][field] for field in fields}]


def breakdown_summaries(evaluation):
    step_rows = []
    step_kind_rows = []
    phase_rows = []
    for preset in ("stock", "synthesized"):
        directory = os.path.join(evaluation, preset, "breakdown")
        steps_path = os.path.join(directory, "steps.csv")
        phases_path = os.path.join(directory, "phases.csv")
        steps = read_csv(steps_path)
        phases = read_csv(phases_path)
        if not steps or not phases:
            raise ValueError(
                f"{directory}: comparison analysis requires non-empty "
                "steps.csv and phases.csv from the five-sample breakdown pass"
            )

        queries = sorted({
            query_number(row["query"]) for row in steps
            if row["record_type"] == "aggregate"
        })
        for query in queries:
            query_steps = [
                row for row in steps
                if query_number(row["query"]) == query
            ]
            aggregate = [
                row for row in query_steps
                if row["record_type"] == "aggregate"
            ]
            grouped_steps = defaultdict(list)
            for row in aggregate:
                grouped_steps[(
                    int(row["step_index"]),
                    row["step_kind"],
                    row["step_label"],
                )].append(float(row["rank_max_ms"]))

            kind_totals = defaultdict(float)
            for (step_index, step_kind, step_label), values in sorted(
                grouped_steps.items()
            ):
                if len(values) != 5:
                    raise ValueError(
                        f"{steps_path}: Q{query} step {step_index} expected "
                        f"5 samples, got {len(values)}"
                    )
                median_ms = statistics.median(values)
                step_rows.append({
                    "preset": preset,
                    "query": query,
                    "step_index": step_index,
                    "step_kind": step_kind,
                    "step_label": step_label,
                    "samples": len(values),
                    "median_rank_max_ms": median_ms,
                })
                kind_totals[step_kind] += median_ms
            for step_kind, value in sorted(kind_totals.items()):
                step_kind_rows.append({
                    "preset": preset,
                    "query": query,
                    "step_kind": step_kind,
                    "time_ms": value,
                })

            rank_steps = [
                row for row in query_steps if row["record_type"] == "rank"
            ]
            iteration_totals = defaultdict(float)
            for row in rank_steps:
                key = (
                    row["timestamp_utc"],
                    int(row["iteration"]),
                    int(row["rank"]),
                )
                iteration_totals[key] += float(row["gpu_ms"])

            critical = []
            iterations = sorted({
                (timestamp, iteration)
                for timestamp, iteration, _ in iteration_totals
            })
            for timestamp, iteration in iterations:
                candidates = [
                    (total, rank)
                    for (value_timestamp, value_iteration, rank), total
                    in iteration_totals.items()
                    if value_timestamp == timestamp
                    and value_iteration == iteration
                ]
                total, rank = sorted(
                    candidates, key=lambda item: (-item[0], item[1])
                )[0]
                critical.append((total, timestamp, iteration, rank))
            critical.sort(key=lambda item: (item[0], item[1], item[2]))
            if len(critical) != 5:
                raise ValueError(
                    f"{steps_path}: Q{query} expected 5 critical-rank "
                    f"samples, got {len(critical)}"
                )
            total, timestamp, iteration, rank = critical[
                (len(critical) - 1) // 2
            ]

            selected_steps = [
                row for row in rank_steps
                if row["timestamp_utc"] == timestamp
                and int(row["iteration"]) == iteration
                and int(row["rank"]) == rank
            ]
            selected_phases = [
                row for row in phases
                if row["record_type"] == "rank"
                and query_number(row["query"]) == query
                and row["timestamp_utc"] == timestamp
                and int(row["iteration"]) == iteration
                and int(row["rank"]) == rank
            ]
            values = defaultdict(float)
            for row in selected_phases:
                if row["category"] == "overhead":
                    values["Overhead"] += float(row["gpu_ms"])

            durations = {
                int(row["step_index"]): float(row["gpu_ms"])
                for row in selected_steps
            }
            phases_by_step = defaultdict(list)
            for row in selected_phases:
                if (row["quality"] == "direct"
                        and row["category"] != "overhead"):
                    phases_by_step[int(row["step_index"])].append(row)
            for step_index, rows in phases_by_step.items():
                for component, duration in attribute_intervals(
                    rows, durations[step_index]
                ).items():
                    values[component] += duration

            phase_total = sum(values.values())
            tolerance = max(0.05, total * 0.01)
            if abs(phase_total - total) > tolerance:
                raise ValueError(
                    f"{directory}: Q{query} phase total {phase_total} "
                    f"!= step total {total}"
                )
            output = {
                "preset": preset,
                "query": query,
                "timestamp_utc": timestamp,
                "iteration": iteration,
                "rank": rank,
                "step_total_ms": total,
            }
            output.update({
                component: values.get(component, 0.0)
                for component in PHASE_ORDER
            })
            phase_rows.append(output)
    return step_rows, step_kind_rows, phase_rows


def phase_component(row):
    category = normalized_phase_category(row)
    if category == "compute":
        return "Compute"
    if category == "overhead":
        return "Overhead"
    if category != "communication":
        raise ValueError(f"unknown phase category {category!r}")
    return {
        "shuffle": "Shuffle",
        "broadcast": "Broadcast",
        "gather": "Gather",
        "reduce": "Reduce",
        "sampling": "Sampling",
        "other": "Other communication",
    }.get(row["communication_pattern"], "Other communication")


def attribute_intervals(rows, step_ms):
    intervals = []
    for row in rows:
        start = max(0.0, float(row["start_ms"]))
        end = min(step_ms, float(row["end_ms"]))
        if start < end:
            intervals.append((start, end, phase_component(row)))
    points = sorted({
        point for start, end, _ in intervals for point in (start, end)
    })
    values = defaultdict(float)
    for start, end in zip(points, points[1:]):
        active = {
            component
            for interval_start, interval_end, component in intervals
            if interval_start < end and interval_end > start
        }
        if active:
            component = next(iter(active)) if len(active) == 1 else "Overlap"
            values[component] += end - start
    return values


def compute_summary(phase_rows):
    by_key = {
        (row["preset"], row["query"]): row for row in phase_rows
    }
    queries = sorted({
        query for _, query in by_key
        if ("stock", query) in by_key
        and ("synthesized", query) in by_key
    })
    result = []
    for query in queries:
        stock = by_key["stock", query]["Compute"]
        synthesized = by_key["synthesized", query]["Compute"]
        result.append({
            "query": query,
            "stock_compute_ms": stock,
            "synthesized_compute_ms": synthesized,
            "speedup": stock / synthesized if synthesized else math.nan,
        })
    if result:
        speedups = [row["speedup"] for row in result if row["speedup"] > 0]
        result.append({
            "query": "aggregate",
            "stock_compute_ms": sum(
                row["stock_compute_ms"] for row in result
            ),
            "synthesized_compute_ms": sum(
                row["synthesized_compute_ms"] for row in result
            ),
            "speedup": math.exp(
                sum(math.log(value) for value in speedups) / len(speedups)
            ),
        })
    return result


def suite_summary(timing):
    queries = [row for row in timing if row["query"] != "aggregate"]
    if not queries:
        return []
    aggregate = next(
        (row for row in timing if row["query"] == "aggregate"), None
    )
    improved = [row for row in queries if row["speedup"] > 1.01]
    regressed = [row for row in queries if row["speedup"] < 0.99]
    unchanged = [
        row for row in queries if 0.99 <= row["speedup"] <= 1.01
    ]
    largest = max(queries, key=lambda row: row["speedup"])
    worst = min(queries, key=lambda row: row["speedup"])
    values = {
        "queries": len(queries),
        "geometric_mean_speedup": aggregate["speedup"] if aggregate else math.nan,
        "stock_sum_of_medians_ms": (
            aggregate["stock_median_ms"] if aggregate else math.nan
        ),
        "synthesized_sum_of_medians_ms": (
            aggregate["synthesized_median_ms"] if aggregate else math.nan
        ),
        "improved_gt_1pct": len(improved),
        "unchanged_within_1pct": len(unchanged),
        "regressed_gt_1pct": len(regressed),
        "largest_gain_query": largest["query"],
        "largest_gain_speedup": largest["speedup"],
        "worst_regression_query": worst["query"],
        "worst_regression_speedup": worst["speedup"],
    }
    return [{"metric": key, "value": value} for key, value in values.items()]


def memory_summary(evaluation):
    result = []
    for preset in ("stock", "synthesized"):
        path = os.path.join(evaluation, preset, "memory", "memory.csv")
        for row in read_csv(path):
            if row["record_type"] != "aggregate":
                continue
            result.append({
                "preset": preset,
                "query": query_number(row["query"]),
                "resident_bytes": int(row["resident_current_bytes"]),
                "query_peak_bytes": int(row["query_peak_bytes"]),
                "total_live_peak_bytes": int(row["total_live_peak_bytes"]),
                "resident_max_rank": int(row["resident_max_rank"]),
                "query_max_rank": int(row["query_max_rank"]),
                "total_max_rank": int(row["total_max_rank"]),
            })
    return result


def memory_aggregate(rows):
    result = []
    for preset in ("stock", "synthesized"):
        values = [row for row in rows if row["preset"] == preset]
        if not values:
            continue
        maximum = max(values, key=lambda row: row["total_live_peak_bytes"])
        result.append({
            "preset": preset,
            "queries": len(values),
            "equal_query_mean_total_live_peak_bytes": sum(
                row["total_live_peak_bytes"] for row in values
            ) / len(values),
            "maximum_total_live_peak_bytes": maximum["total_live_peak_bytes"],
            "maximum_query": maximum["query"],
            "maximum_rank": maximum["total_max_rank"],
        })
    return result


def plan_choices(plan_dir):
    result = []
    changed_steps = {}
    for path in sorted(glob.glob(os.path.join(plan_dir, "q??.json"))):
        with open(path, encoding="utf-8") as source:
            config = json.load(source)
        query = int(config["focus"]["query_id"])
        for override in config.get("join_overrides", []):
            changed = override["implementation"] != "hash_gfur"
            result.append({
                "query": query,
                "local_step_index": int(override["step_index"]),
                "semantic_label": override["semantic_label"],
                "implementation": override["implementation"],
                "changed_from_stock": int(changed),
            })
            if changed:
                changed_steps.setdefault(query, set()).add(
                    int(override["step_index"])
                )
    return result, changed_steps


def sqlite_for(report):
    base, _ = os.path.splitext(report)
    if base.endswith(".nsys"):
        base = base[:-5]
    sqlite_path = base + ".sqlite"
    if not os.path.exists(sqlite_path):
        subprocess.run([
            "nsys", "export", "--type=sqlite", "--force-overwrite=true",
            "--quiet=true", "--output", sqlite_path, report,
        ], check=True)
    return sqlite_path


def table_exists(database, table):
    return database.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()[0] != 0


def string_expression(database):
    columns = {
        row[1] for row in database.execute("PRAGMA table_info(NVTX_EVENTS)")
    }
    if "textId" in columns and table_exists(database, "StringIds"):
        return "COALESCE(NVTX_EVENTS.text, StringIds.value)", (
            " LEFT JOIN StringIds ON NVTX_EVENTS.textId = StringIds.id"
        )
    return "NVTX_EVENTS.text", ""


def nvtx_intervals(database):
    expression, join = string_expression(database)
    rows = database.execute(
        f"SELECT NVTX_EVENTS.start,NVTX_EVENTS.end,{expression} "
        f"FROM NVTX_EVENTS{join} WHERE NVTX_EVENTS.end IS NOT NULL"
    )
    query_ranges = []
    steps = {}
    pattern = re.compile(
        r"(?:exec\.)?step\[(\d+)\]:(DistributedEqJoin.*)"
    )
    for start, end, text in rows:
        if not text:
            continue
        if text == "tpch.profile":
            query_ranges.append((int(start), int(end)))
        match = pattern.match(text)
        if match:
            steps.setdefault(int(match.group(1)), []).append(
                (int(start), int(end), match.group(2))
            )
    if not query_ranges:
        raise RuntimeError("no tpch.profile NVTX ranges")
    query_window = (
        min(value[0] for value in query_ranges),
        max(value[1] for value in query_ranges),
    )
    step_windows = {
        step: (
            min(value[0] for value in values),
            max(value[1] for value in values),
            values[0][2],
        )
        for step, values in steps.items()
    }
    return query_window, step_windows


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(intervals, excluded):
    """Return the half-open portions of intervals not covered by excluded."""
    result = []
    excluded = merge_intervals(excluded)
    for start, end in merge_intervals(intervals):
        cursor = start
        for excluded_start, excluded_end in excluded:
            if excluded_end <= cursor:
                continue
            if excluded_start >= end:
                break
            if cursor < excluded_start:
                result.append((cursor, min(excluded_start, end)))
            cursor = max(cursor, excluded_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def process_gpu_map(database):
    if not (
        table_exists(database, "TARGET_INFO_CUDA_CONTEXT_INFO")
        and table_exists(database, "TARGET_INFO_CUDA_DEVICE")
    ):
        return {}
    devices = defaultdict(set)
    for process_id, gpu_id in database.execute(
        "SELECT DISTINCT context.processId,device.gpuId "
        "FROM TARGET_INFO_CUDA_CONTEXT_INFO AS context "
        "JOIN TARGET_INFO_CUDA_DEVICE AS device "
        "ON device.pid=context.processId "
        "AND device.cudaId=context.deviceId"
    ):
        devices[int(process_id)].add(int(gpu_id))
    ambiguous = {
        process_id: values
        for process_id, values in devices.items()
        if len(values) != 1
    }
    if ambiguous:
        raise ValueError(f"processes mapped to multiple GPUs: {ambiguous}")
    return {
        process_id: next(iter(values))
        for process_id, values in devices.items()
    }


def is_nccl_kernel_name(name):
    """Return whether a CUPTI kernel name belongs to NCCL."""
    return bool(name) and "nccl" in name.lower()


def nccl_kernel_intervals(database):
    """Collect the union of actual NCCL GPU-kernel intervals per GPU."""
    if not (
        table_exists(database, "CUPTI_ACTIVITY_KIND_KERNEL")
        and table_exists(database, "StringIds")
    ):
        return {}
    gpu_by_process = process_gpu_map(database)
    raw = defaultdict(list)
    rows = database.execute(
        "SELECT kernel.start,kernel.end,kernel.globalPid,"
        "COALESCE(demangled.value,short.value) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel "
        "LEFT JOIN StringIds AS demangled "
        "ON demangled.id=kernel.demangledName "
        "LEFT JOIN StringIds AS short "
        "ON short.id=kernel.shortName "
        "WHERE kernel.end IS NOT NULL"
    )
    for start, end, global_pid, name in rows:
        if global_pid is None or not is_nccl_kernel_name(name):
            continue
        # CUPTI globalPid uses the same packed process identifier as NVTX's
        # globalTid: process ID in bits 24..47.
        process_id = (int(global_pid) >> 24) & 0xFFFFFF
        gpu = gpu_by_process.get(process_id)
        if gpu is not None:
            raw[gpu].append((int(start), int(end)))
    return {
        gpu: merge_intervals(intervals)
        for gpu, intervals in raw.items()
    }


def nvtx_scope_intervals(database):
    """Collect per-GPU, disjoint phase and operator intervals.

    Compute/communication overlap is removed from both phase scopes to match
    the breakdown CSV's separate Overlap category.
    """
    expression, join = string_expression(database)
    rows = database.execute(
        f"SELECT NVTX_EVENTS.start,NVTX_EVENTS.end,{expression},"
        f"NVTX_EVENTS.globalTid FROM NVTX_EVENTS{join} "
        f"WHERE NVTX_EVENTS.end IS NOT NULL"
    )
    gpu_by_process = process_gpu_map(database)
    raw = defaultdict(lambda: defaultdict(list))
    step_pattern = re.compile(r"(?:exec\.)?step\[\d+\]:(.+)")
    for start, end, text, global_tid in rows:
        if not text or global_tid is None:
            continue
        # Nsight globalTid packs globalPid in bits 24..47 and thread ID in
        # bits 0..23. NVTX ranges may be emitted from non-main threads.
        process_id = (int(global_tid) >> 24) & 0xFFFFFF
        gpu = gpu_by_process.get(process_id)
        if gpu is None:
            continue
        interval = (int(start), int(end))
        phase_scope = normalized_nvtx_phase_scope(text)
        if phase_scope is not None:
            raw[phase_scope][gpu].append(interval)
            mechanism_scope = mechanism_nvtx_phase_scope(text)
            if mechanism_scope is not None:
                raw[mechanism_scope][gpu].append(interval)
        else:
            match = step_pattern.match(text)
            if not match:
                continue
            kind = match.group(1).split("(", 1)[0]
            if kind == "DistributedEqJoin":
                raw["distributed_eq_join"][gpu].append(interval)
            elif kind == "Filter":
                raw["filter"][gpu].append(interval)

    result = defaultdict(dict)
    gpus = set(raw["phase_compute"]) | set(raw["phase_communication"])
    for gpu in gpus:
        compute = merge_intervals(raw["phase_compute"].get(gpu, []))
        communication = merge_intervals(
            raw["phase_communication"].get(gpu, [])
        )
        result["phase_compute"][gpu] = subtract_intervals(
            compute, communication
        )
        result["phase_communication"][gpu] = subtract_intervals(
            communication, compute
        )
    for scope in (
        "phase_shuffle", "phase_replication",
        "distributed_eq_join", "filter",
    ):
        for gpu, intervals in raw[scope].items():
            result[scope][gpu] = merge_intervals(intervals)
    return result


def metric_samples(database):
    active_devices = set()
    if table_exists(database, "TARGET_INFO_CUDA_CONTEXT_INFO"):
        if table_exists(database, "TARGET_INFO_CUDA_DEVICE"):
            # CUDA context deviceId is the process-local CUDA ordinal, while
            # GPU_METRICS typeId encodes Nsight's physical gpuId.  They differ
            # whenever CUDA_VISIBLE_DEVICES reorders or filters the devices.
            active_devices = {
                int(row[0]) for row in database.execute(
                    "SELECT DISTINCT device.gpuId "
                    "FROM TARGET_INFO_CUDA_CONTEXT_INFO AS context "
                    "JOIN TARGET_INFO_CUDA_DEVICE AS device "
                    "ON device.pid=context.processId "
                    "AND device.cudaId=context.deviceId"
                )
            }
        else:
            active_devices = {
                int(row[0]) for row in database.execute(
                    "SELECT DISTINCT deviceId "
                    "FROM TARGET_INFO_CUDA_CONTEXT_INFO"
                )
            }
    names = {
        (int(type_id), int(metric_id)): name
        for type_id, metric_id, name in database.execute(
            "SELECT typeId,metricId,metricName FROM TARGET_INFO_GPU_METRICS"
        )
    }
    result = {}
    for timestamp, type_id, metric_id, value in database.execute(
        "SELECT timestamp,typeId,metricId,value FROM GPU_METRICS "
        "ORDER BY timestamp"
    ):
        name = names.get((int(type_id), int(metric_id)))
        if name is None:
            continue
        # GPU metric type IDs encode the CUDA device in the low 32 bits.
        # Exclude visible devices that have no CUDA context in this capture.
        device_id = int(type_id) & 0xFFFFFFFF
        if active_devices and device_id not in active_devices:
            continue
        result.setdefault((device_id, name), []).append(
            # Nsight stores Throughput % metrics directly as integer percent
            # samples (0..100). Clock metrics use a different fixed-point
            # representation, but this analysis intentionally excludes them.
            (int(timestamp), float(value))
        )
    return result


def interval_metrics(samples, start, end):
    by_gpu = sorted({key[0] for key in samples})
    rows = []
    for gpu in by_gpu:
        selected = {}
        all_timestamps = []
        for output_name, source_names in METRICS.items():
            source_values = []
            for source_name in source_names:
                values = [
                    (timestamp, value)
                    for timestamp, value in samples.get((gpu, source_name), [])
                    if start <= timestamp <= end
                ]
                source_values.append(values)
                all_timestamps.extend(timestamp for timestamp, _ in values)
            timestamp_values = {}
            for values in source_values:
                for timestamp, value in values:
                    timestamp_values[timestamp] = (
                        timestamp_values.get(timestamp, 0.0) + value
                    )
            selected[output_name] = (
                statistics.mean(timestamp_values.values())
                if timestamp_values else math.nan
            )
        rows.append({
            "gpu_device": gpu,
            **selected,
            "sample_count": len(set(all_timestamps)),
            "covered_ms": (
                (max(all_timestamps) - min(all_timestamps)) / 1e6
                if len(all_timestamps) >= 2 else 0.0
            ),
            "window_ms": (end - start) / 1e6,
        })
    return rows


def samples_in_intervals(values, intervals):
    selected = []
    interval_index = 0
    for timestamp, value in values:
        while (
            interval_index < len(intervals)
            and timestamp >= intervals[interval_index][1]
        ):
            interval_index += 1
        if interval_index >= len(intervals):
            break
        start, end = intervals[interval_index]
        if start <= timestamp < end:
            selected.append((timestamp, value))
    return selected


def interval_set_metrics(samples, intervals_by_gpu):
    rows = []
    for gpu, intervals in sorted(intervals_by_gpu.items()):
        intervals = merge_intervals(intervals)
        if not intervals:
            continue
        selected = {}
        all_timestamps = []
        for output_name, source_names in METRICS.items():
            timestamp_values = {}
            for source_name in source_names:
                values = samples_in_intervals(
                    samples.get((gpu, source_name), []), intervals
                )
                all_timestamps.extend(timestamp for timestamp, _ in values)
                for timestamp, value in values:
                    timestamp_values[timestamp] = (
                        timestamp_values.get(timestamp, 0.0) + value
                    )
            selected[output_name] = (
                statistics.mean(timestamp_values.values())
                if timestamp_values else math.nan
            )
        rows.append({
            "gpu_device": gpu,
            **selected,
            "sample_count": len(set(all_timestamps)),
            "interval_count": len(intervals),
            "window_ms": sum(end - start for start, end in intervals) / 1e6,
        })
    return rows


def profile_rows(evaluation, changed_steps):
    rows = []
    scoped_rows = []
    for preset in ("stock", "synthesized"):
        pattern = os.path.join(
            evaluation, preset, "profiles", "sf*", "q??.nsys-rep"
        )
        for report in sorted(glob.glob(pattern)):
            query = int(re.search(r"q(\d+)\.nsys-rep$", report).group(1))
            database = sqlite3.connect(sqlite_for(report))
            try:
                if not table_exists(database, "GPU_METRICS"):
                    continue
                query_window, step_windows = nvtx_intervals(database)
                samples = metric_samples(database)
                scope_intervals = nvtx_scope_intervals(database)
                scope_intervals["nccl_kernel"] = nccl_kernel_intervals(
                    database
                )
                for row in interval_metrics(samples, *query_window):
                    rows.append({
                        "preset": preset,
                        "query": query,
                        "scope": "query",
                        "step_index": -1,
                        "step_label": "tpch.profile",
                        **row,
                    })
                for step in sorted(changed_steps.get(query, set())):
                    if step not in step_windows:
                        continue
                    start, end, label = step_windows[step]
                    for row in interval_metrics(samples, start, end):
                        rows.append({
                            "preset": preset,
                            "query": query,
                            "scope": "changed_join",
                            "step_index": step,
                            "step_label": label,
                            **row,
                        })
                for scope, intervals_by_gpu in scope_intervals.items():
                    for row in interval_set_metrics(
                        samples, intervals_by_gpu
                    ):
                        scoped_rows.append({
                            "preset": preset,
                            "query": query,
                            "scope": scope,
                            **row,
                        })
            finally:
                database.close()
    return rows, scoped_rows


def utilization_aggregates(rows):
    fields = list(METRICS)
    result = []
    query_rows = [row for row in rows if row["scope"] == "query"]
    per_query = {}
    for row in query_rows:
        per_query.setdefault((row["preset"], row["query"]), []).append(row)
    query_means = []
    for (preset, query), values in per_query.items():
        output = {"preset": preset, "query": query}
        for field in fields:
            valid = [value[field] for value in values if not math.isnan(value[field])]
            output[field] = statistics.mean(valid) if valid else math.nan
        output["window_ms"] = statistics.mean(
            value["window_ms"] for value in values
        )
        query_means.append(output)
    for preset in ("stock", "synthesized"):
        values = [row for row in query_means if row["preset"] == preset]
        for reduction in ("equal_query", "time_weighted"):
            output = {"preset": preset, "reduction": reduction}
            for field in fields:
                valid = [row for row in values if not math.isnan(row[field])]
                if not valid:
                    output[field] = math.nan
                elif reduction == "equal_query":
                    output[field] = statistics.mean(row[field] for row in valid)
                else:
                    weight = sum(row["window_ms"] for row in valid)
                    output[field] = (
                        sum(row[field] * row["window_ms"] for row in valid) /
                        weight if weight else math.nan
                    )
            output["queries"] = len(values)
            result.append(output)
    return query_means, result


def scoped_utilization_aggregates(rows):
    result = []
    scopes = (
        "phase_compute", "phase_communication",
        "phase_shuffle", "phase_replication",
        "distributed_eq_join", "filter", "nccl_kernel",
    )
    for preset in ("stock", "synthesized"):
        for scope in scopes:
            values = [
                row for row in rows
                if row["preset"] == preset and row["scope"] == scope
            ]
            if not values:
                continue
            output = {
                "preset": preset,
                "scope": scope,
                "reduction": "gpu_time_weighted",
            }
            for field in METRICS:
                valid = [
                    row for row in values
                    if not math.isnan(row[field]) and row["window_ms"] > 0
                ]
                weight = sum(row["window_ms"] for row in valid)
                output[field] = (
                    sum(
                        row[field] * row["window_ms"] for row in valid
                    ) / weight if weight else math.nan
                )
            output["queries"] = len({row["query"] for row in values})
            output["gpu_windows"] = len(values)
            output["total_gpu_window_ms"] = sum(
                row["window_ms"] for row in values
            )
            result.append(output)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--physical-plan-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    timing = timing_summary(args.evaluation_dir)
    setup = experiment_setup(args.evaluation_dir)
    steps, step_kinds, phases = breakdown_summaries(args.evaluation_dir)
    compute = compute_summary(phases)
    suite = suite_summary(timing)
    memory = memory_summary(args.evaluation_dir)
    memory_suite = memory_aggregate(memory)
    choices, changed = plan_choices(args.physical_plan_dir)
    utilization, scoped_utilization = profile_rows(
        args.evaluation_dir, changed
    )
    query_utilization, aggregate_utilization = utilization_aggregates(
        utilization
    )
    scoped_utilization_aggregate = scoped_utilization_aggregates(
        scoped_utilization
    )

    write_csv(
        os.path.join(args.output_dir, "timing_summary.csv"),
        timing,
        ["query", "stock_median_ms", "synthesized_median_ms",
         "speedup", "delta_pct"],
    )
    write_csv(
        os.path.join(args.output_dir, "experiment_setup.csv"),
        setup,
        ["scale", "ranks", "nodes", "hostname", "gpu_name"],
    )
    write_csv(
        os.path.join(args.output_dir, "step_summary.csv"),
        steps,
        ["preset", "query", "step_index", "step_kind", "step_label",
         "samples", "median_rank_max_ms"],
    )
    write_csv(
        os.path.join(args.output_dir, "step_kind_summary.csv"),
        step_kinds,
        ["preset", "query", "step_kind", "time_ms"],
    )
    write_csv(
        os.path.join(args.output_dir, "phase_summary.csv"),
        phases,
        ["preset", "query", "timestamp_utc", "iteration", "rank",
         "step_total_ms", *PHASE_ORDER],
    )
    write_csv(
        os.path.join(args.output_dir, "compute_summary.csv"),
        compute,
        ["query", "stock_compute_ms", "synthesized_compute_ms", "speedup"],
    )
    write_csv(
        os.path.join(args.output_dir, "suite_summary.csv"),
        suite,
        ["metric", "value"],
    )
    write_csv(
        os.path.join(args.output_dir, "memory_summary.csv"),
        memory,
        ["preset", "query", "resident_bytes", "query_peak_bytes",
         "total_live_peak_bytes", "resident_max_rank", "query_max_rank",
         "total_max_rank"],
    )
    write_csv(
        os.path.join(args.output_dir, "memory_aggregate.csv"),
        memory_suite,
        ["preset", "queries", "equal_query_mean_total_live_peak_bytes",
         "maximum_total_live_peak_bytes", "maximum_query", "maximum_rank"],
    )
    write_csv(
        os.path.join(args.output_dir, "join_choices.csv"),
        choices,
        ["query", "local_step_index", "semantic_label", "implementation",
         "changed_from_stock"],
    )
    utilization_fields = [
        "preset", "query", "scope", "step_index", "step_label", "gpu_device",
        *METRICS, "sample_count", "covered_ms", "window_ms",
    ]
    write_csv(
        os.path.join(args.output_dir, "utilization_raw.csv"),
        utilization,
        utilization_fields,
    )
    write_csv(
        os.path.join(args.output_dir, "utilization_query.csv"),
        query_utilization,
        ["preset", "query", *METRICS, "window_ms"],
    )
    write_csv(
        os.path.join(args.output_dir, "utilization_aggregate.csv"),
        aggregate_utilization,
        ["preset", "reduction", *METRICS, "queries"],
    )
    write_csv(
        os.path.join(args.output_dir, "utilization_scopes_raw.csv"),
        scoped_utilization,
        [
            "preset", "query", "scope", "gpu_device", *METRICS,
            "sample_count", "interval_count", "window_ms",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "utilization_scopes_aggregate.csv"),
        scoped_utilization_aggregate,
        [
            "preset", "scope", "reduction", *METRICS, "queries",
            "gpu_windows", "total_gpu_window_ms",
        ],
    )


if __name__ == "__main__":
    main()
