#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Measure kernel-level overlap on the named column-overlap CUDA streams."""

import argparse
import collections
import sqlite3


STREAMS = {
    "comm": "alps.column_join.comm",
    "prepare": "alps.column_join.prepare",
    "match_materialize": "alps.column_join.match_materialize",
}


def interval_union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def interval_intersection(left, right):
    left = interval_union(left)
    right = interval_union(right)
    result = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if start < end:
            result.append((start, end))
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return result


def duration_ms(intervals):
    return sum(end - start for start, end in interval_union(intervals)) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", help="SQLite file exported from an NSYS report")
    parser.add_argument(
        "--iteration", type=int, default=-1,
        help="zero-based iteration to report; negative selects the last (default: -1)")
    args = parser.parse_args()

    database = sqlite3.connect(args.sqlite)
    stream_rows = database.execute(
        "SELECT processId, streamId, name FROM TARGET_INFO_NVTX_CUDA_STREAM"
    ).fetchall()
    stream_ids = {
        (process_id, name): stream_id
        for process_id, stream_id, name in stream_rows
        if name in STREAMS.values()
    }
    processes = sorted({process_id for process_id, _ in stream_ids})
    if not processes:
        raise SystemExit("no named column-overlap streams found")

    kernels = collections.defaultdict(list)
    for start, end, stream_id, global_pid in database.execute(
        "SELECT start, end, streamId, globalPid "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ):
        # NSYS encodes GlobalId as (host_id << 48) | (process_id << 24).
        process_id = (global_pid >> 24) & 0xFFFFFF
        kernels[process_id, stream_id].append((start, end))

    nvtx_count = database.execute(
        "SELECT COUNT(*) FROM NVTX_EVENTS "
        "WHERE text = 'column_overlap.match_keys'"
    ).fetchone()[0]
    iterations = nvtx_count // len(processes)
    if iterations <= 0:
        raise SystemExit("no column_overlap.match_keys iteration ranges found")
    iteration = args.iteration if args.iteration >= 0 else iterations + args.iteration
    if iteration < 0 or iteration >= iterations:
        raise SystemExit(f"iteration must be in [0, {iterations - 1}]")

    print(
        "process_id,iteration,comm_busy_ms,prepare_busy_ms,"
        "match_materialize_busy_ms,comm_match_overlap_ms,"
        "comm_busy_overlapped_pct,active_union_ms,kernel_span_ms"
    )
    overlap_values = []
    overlap_percentages = []
    for process_id in processes:
        selected = {}
        for logical_name, cuda_name in STREAMS.items():
            values = kernels[process_id, stream_ids[process_id, cuda_name]]
            if len(values) % iterations != 0:
                raise SystemExit(
                    f"process {process_id} stream {cuda_name} has {len(values)} "
                    f"kernels across {iterations} iterations; cannot split evenly")
            per_iteration = len(values) // iterations
            begin = iteration * per_iteration
            selected[logical_name] = values[begin:begin + per_iteration]

        comm_ms = duration_ms(selected["comm"])
        prepare_ms = duration_ms(selected["prepare"])
        match_ms = duration_ms(selected["match_materialize"])
        overlap_ms = duration_ms(interval_intersection(
            selected["comm"], selected["match_materialize"]))
        overlap_pct = 100.0 * overlap_ms / comm_ms if comm_ms else 0.0
        all_intervals = sum(selected.values(), [])
        active_union_ms = duration_ms(all_intervals)
        kernel_span_ms = (
            max(end for _, end in all_intervals)
            - min(start for start, _ in all_intervals)
        ) / 1e6
        overlap_values.append(overlap_ms)
        overlap_percentages.append(overlap_pct)
        print(
            f"{process_id},{iteration},{comm_ms:.6f},{prepare_ms:.6f},"
            f"{match_ms:.6f},{overlap_ms:.6f},{overlap_pct:.3f},"
            f"{active_union_ms:.6f},{kernel_span_ms:.6f}"
        )

    print(
        f"mean,{iteration},,,,,{sum(overlap_percentages) / len(processes):.3f},,"
        f" # mean overlap_ms={sum(overlap_values) / len(processes):.6f}"
    )


if __name__ == "__main__":
    main()
