#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Model SF=1000 distributed TPC-H strong scaling, query by query.

The model has two independently derived terms:

* non-communication time for each query is fitted as alpha * N**beta from
  legacy_cta64 timing/breakdown campaigns at N = 8, 16, 24, and 32 GPUs;
* shuffle and replication/all-gather time is reconstructed per physical
  column from message traces and a CE-aligned per-GPU cross-bandwidth curve.

The CE curve is B(m) = m / (a + b*m), where m is the average per-peer message
size.  The shuffle microbenchmark reports B = (N-1)*m/t, so a real column is
predicted as t = average_cross_bytes_per_gpu / B(m).  Shuffle receive records
are excluded, self traffic is excluded from cross bytes, and ncclBroadcast
records are de-duplicated across observer ranks.

The output CSVs retain source paths and intermediate quantities so every bar
and curve can be audited without reverse engineering the plotting code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    import numpy as np
except ModuleNotFoundError as error:
    sys.exit(f"error: numpy is required: {error}")

try:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ModuleNotFoundError as error:
    sys.exit(f"error: matplotlib is required: {error}")

from plot_distributed_tpch_aggregate_breakdown import (
    communication_fraction_summary,
    timing_summary,
)
from summarize_distributed_tpch_messages import successful_messages_csv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_GPU_COUNTS = (8, 16, 24, 32)
DEFAULT_PREDICTION_GPU_COUNTS = tuple(range(8, 73, 8))
DEFAULT_MESSAGE_ANCHOR_GPUS = 8
QUERIES = tuple(range(1, 23))
MESSAGE_PATTERNS_FOR_ALLGATHER = frozenset(("broadcast", "sampling"))
CE_ARM = "p2p_ce_push_aligned"
BUS_BW_FIELD = "per_gpu_cross_gbps_median_median_across_reps"
BYTES_PER_GB = 1_000_000_000.0
COLORS = {
    "compute": "#4C78A8",
    "shuffle": "#F58518",
    "allgather": "#54A24B",
    "measured": "#303030",
    "g8": "#7B2CBF",
    "g24": "#00A6A6",
    "pooled": "#202020",
}


@dataclass(frozen=True)
class ComputeSample:
    scale_factor: int
    gpus: int
    query: int
    measured_e2e_ms: float
    communication_fraction: float
    measured_compute_ms: float
    measured_legacy_communication_ms: float
    timing_attempt_dir: str
    breakdown_attempt_dir: str


@dataclass(frozen=True)
class ComputeFit:
    query: int
    alpha_ms: float
    beta: float
    r2_log: float
    rmse_log: float
    training_mape_percent: float
    loo_mape_percent: float
    training_points: int


@dataclass(frozen=True)
class BandwidthFit:
    scope: str
    gpu_counts: str
    a_seconds: float
    b_seconds_per_byte: float
    knee_message_bytes: float
    saturation_gbps: float
    r2_log: float
    rmse_log: float
    mape_percent: float
    max_ape_percent: float
    sample_count: int
    objective: str


def parse_gpu_counts(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("GPU counts must be comma-separated integers >= 2")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("GPU counts must be unique")
    return values


def query_number(value: str | int) -> int:
    text = str(value).strip()
    if text[:1].lower() == "q":
        text = text[1:]
    return int(text)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "distributed_tpch",
        help="Distributed TPC-H results root",
    )
    parser.add_argument(
        "--bandwidth-campaign",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "alps_shuffle_busbw"
        / "main_ordered_ce_20260816",
        help="ALPS shuffle-bandwidth campaign containing summary.csv",
    )
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument(
        "--training-gpu-counts",
        type=parse_gpu_counts,
        default=DEFAULT_TRAINING_GPU_COUNTS,
        help="Comma-separated compute/message validation counts (default: 8,16,24,32)",
    )
    parser.add_argument(
        "--prediction-gpu-counts",
        type=parse_gpu_counts,
        default=DEFAULT_PREDICTION_GPU_COUNTS,
        help="Comma-separated prediction counts (default: 8,16,...,72)",
    )
    parser.add_argument(
        "--message-anchor-gpus",
        type=int,
        default=DEFAULT_MESSAGE_ANCHOR_GPUS,
        help="Measured message campaign used as the scaling anchor (default: 8)",
    )
    parser.add_argument("--timing-arm", default="legacy_cta64")
    parser.add_argument("--message-arm", default="legacy")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "distributed_tpch"
        / "figures"
        / "presentation",
        help="Output root; figures go to png/ and pdf/, CSVs stay at the root",
    )
    return parser.parse_args(argv)


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def write_csv(path: Path, rows: Sequence[object | Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    dictionaries: list[dict[str, object]] = []
    for row in rows:
        dictionaries.append(asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)
    print(f"wrote {path}")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        directory = output_dir / extension
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def load_compute_samples(
    results_dir: Path,
    scale: int,
    gpu_counts: Sequence[int],
    timing_arm: str,
) -> list[ComputeSample]:
    samples: list[ComputeSample] = []
    for gpus in gpu_counts:
        for query in QUERIES:
            e2e_ms, timing_attempt, _ = timing_summary(
                results_dir, scale, gpus, query, timing_arm
            )
            comm_fraction, breakdown_attempt, _ = communication_fraction_summary(
                results_dir, scale, gpus, query, timing_arm
            )
            communication_ms = e2e_ms * comm_fraction
            compute_ms = e2e_ms - communication_ms
            if compute_ms <= 0.0:
                raise ValueError(
                    f"Q{query} on {gpus} GPUs has non-positive compute time {compute_ms}"
                )
            samples.append(
                ComputeSample(
                    scale_factor=scale,
                    gpus=gpus,
                    query=query,
                    measured_e2e_ms=e2e_ms,
                    communication_fraction=comm_fraction,
                    measured_compute_ms=compute_ms,
                    measured_legacy_communication_ms=communication_ms,
                    timing_attempt_dir=relative_to_repo(timing_attempt),
                    breakdown_attempt_dir=relative_to_repo(breakdown_attempt),
                )
            )
    expected = len(gpu_counts) * len(QUERIES)
    if len(samples) != expected:
        raise ValueError(f"loaded {len(samples)} compute samples, expected {expected}")
    return samples


def fit_power_xy(gpus: Sequence[float], values: Sequence[float]) -> tuple[float, float]:
    x = np.log(np.asarray(gpus, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    if np.any(~np.isfinite(y)) or np.any(np.asarray(values) <= 0.0):
        raise ValueError("power-model samples must be finite and positive")
    beta, intercept = np.polyfit(x, y, 1)
    return float(math.exp(intercept)), float(beta)


def power_prediction(alpha: float, beta: float, gpus: float) -> float:
    return alpha * gpus**beta


def fit_compute_models(samples: Sequence[ComputeSample]) -> list[ComputeFit]:
    grouped: dict[int, list[ComputeSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.query].append(sample)
    fits: list[ComputeFit] = []
    for query in QUERIES:
        rows = sorted(grouped[query], key=lambda row: row.gpus)
        gpus = [row.gpus for row in rows]
        values = [row.measured_compute_ms for row in rows]
        alpha, beta = fit_power_xy(gpus, values)
        observed_log = np.log(np.asarray(values))
        predicted = np.asarray([power_prediction(alpha, beta, n) for n in gpus])
        predicted_log = np.log(predicted)
        residual = observed_log - predicted_log
        denominator = float(np.sum((observed_log - np.mean(observed_log)) ** 2))
        r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 1.0
        training_mape = float(
            np.mean(np.abs(predicted - np.asarray(values)) / np.asarray(values)) * 100.0
        )
        loo_errors = []
        for held_out in range(len(rows)):
            train_gpus = [n for index, n in enumerate(gpus) if index != held_out]
            train_values = [v for index, v in enumerate(values) if index != held_out]
            loo_alpha, loo_beta = fit_power_xy(train_gpus, train_values)
            loo_prediction = power_prediction(loo_alpha, loo_beta, gpus[held_out])
            loo_errors.append(abs(loo_prediction - values[held_out]) / values[held_out])
        fits.append(
            ComputeFit(
                query=query,
                alpha_ms=alpha,
                beta=beta,
                r2_log=r2,
                rmse_log=float(math.sqrt(float(np.mean(residual**2)))),
                training_mape_percent=training_mape,
                loo_mape_percent=100.0 * statistics.fmean(loo_errors),
                training_points=len(rows),
            )
        )
    return fits


def golden_section_minimize(function, lower: float, upper: float) -> float:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    f_left = function(left)
    f_right = function(right)
    for _ in range(160):
        if f_left <= f_right:
            upper = right
            right = left
            f_right = f_left
            left = upper - ratio * (upper - lower)
            f_left = function(left)
        else:
            lower = left
            left = right
            f_left = f_right
            right = lower + ratio * (upper - lower)
            f_right = function(right)
    return (lower + upper) / 2.0


def fit_bandwidth_xy(
    message_bytes: Sequence[float], bandwidth_gbps: Sequence[float], scope: str
) -> BandwidthFit:
    messages = np.asarray(message_bytes, dtype=float)
    bandwidth_bps = np.asarray(bandwidth_gbps, dtype=float) * BYTES_PER_GB
    if np.any(messages <= 0.0) or np.any(bandwidth_bps <= 0.0):
        raise ValueError("bandwidth samples must be positive")

    log_bandwidth = np.log(bandwidth_bps)

    def objective(log_knee: float) -> float:
        knee = math.exp(log_knee)
        shape = messages / (messages + knee)
        log_saturation = float(np.mean(log_bandwidth - np.log(shape)))
        residual = log_bandwidth - (log_saturation + np.log(shape))
        return float(np.mean(residual**2))

    lower = math.log(float(np.min(messages)) / 1_000.0)
    upper = math.log(float(np.max(messages)) * 1_000.0)
    log_knee = golden_section_minimize(objective, lower, upper)
    knee = math.exp(log_knee)
    shape = messages / (messages + knee)
    log_saturation = float(np.mean(log_bandwidth - np.log(shape)))
    saturation_bps = math.exp(log_saturation)
    predictions = saturation_bps * shape
    residual = log_bandwidth - np.log(predictions)
    denominator = float(np.sum((log_bandwidth - np.mean(log_bandwidth)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 1.0
    relative = np.abs(predictions - bandwidth_bps) / bandwidth_bps
    b = 1.0 / saturation_bps
    a = knee * b
    return BandwidthFit(
        scope=scope,
        gpu_counts="",
        a_seconds=a,
        b_seconds_per_byte=b,
        knee_message_bytes=knee,
        saturation_gbps=saturation_bps / BYTES_PER_GB,
        r2_log=r2,
        rmse_log=float(math.sqrt(float(np.mean(residual**2)))),
        mape_percent=100.0 * float(np.mean(relative)),
        max_ape_percent=100.0 * float(np.max(relative)),
        sample_count=len(messages),
        objective="equal-weight least squares in log bandwidth",
    )


def bandwidth_gbps(fit: BandwidthFit, message_bytes: float) -> float:
    if message_bytes <= 0.0:
        return 0.0
    bandwidth_bps = message_bytes / (
        fit.a_seconds + fit.b_seconds_per_byte * message_bytes
    )
    return bandwidth_bps / BYTES_PER_GB


def load_bandwidth_model(
    campaign: Path,
) -> tuple[list[dict[str, object]], list[BandwidthFit], float, float]:
    summary = campaign / "summary.csv"
    with summary.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["arm"] == CE_ARM]
    if not rows:
        raise ValueError(f"{summary}: no rows for {CE_ARM}")
    for row in rows:
        if BUS_BW_FIELD not in row:
            raise ValueError(f"{summary}: missing {BUS_BW_FIELD}")

    messages = [float(row["actual_per_peer_bytes"]) for row in rows]
    bandwidths = [float(row[BUS_BW_FIELD]) for row in rows]
    pooled = fit_bandwidth_xy(messages, bandwidths, "pooled")
    pooled = BandwidthFit(**{**asdict(pooled), "gpu_counts": ",".join(
        str(value) for value in sorted({int(row["g"]) for row in rows})
    )})
    fits = [pooled]
    for gpu_count in sorted({int(row["g"]) for row in rows}):
        selected = [row for row in rows if int(row["g"]) == gpu_count]
        fit = fit_bandwidth_xy(
            [float(row["actual_per_peer_bytes"]) for row in selected],
            [float(row[BUS_BW_FIELD]) for row in selected],
            f"g{gpu_count}",
        )
        fits.append(BandwidthFit(**{**asdict(fit), "gpu_counts": str(gpu_count)}))

    sample_rows: list[dict[str, object]] = []
    for row in rows:
        message = float(row["actual_per_peer_bytes"])
        observed = float(row[BUS_BW_FIELD])
        predicted = bandwidth_gbps(pooled, message)
        sample_rows.append(
            {
                "gpus": int(row["g"]),
                "arm": row["arm"],
                "nominal_per_peer_bytes": int(row["nominal_per_peer_bytes"]),
                "actual_per_peer_bytes": int(message),
                "observed_per_gpu_cross_gbps": observed,
                "pooled_predicted_per_gpu_cross_gbps": predicted,
                "absolute_percentage_error": 100.0 * abs(predicted - observed) / observed,
                "repetitions": int(row["repetitions"]),
                "source_summary_csv": relative_to_repo(summary),
            }
        )
    return sample_rows, fits, min(messages), max(messages)


def message_campaign_root(
    results_dir: Path, scale: int, gpus: int, message_arm: str
) -> Path:
    return (
        results_dir
        / f"sf{scale}"
        / f"{gpus}gpu_128p_message_size"
        / "arms"
        / message_arm
        / "messages"
    )


def physical_message_key(row: Mapping[str, str]) -> tuple[object, ...]:
    return (
        query_number(row["query"]),
        int(row["step_index"]),
        row["step_kind"],
        row["step_label"],
        int(row["phase_index"]),
        row["phase_name"],
        row["communication_pattern"],
        row["collective_primitive"],
        int(row["column_ordinal"]),
        row["column_type"],
        row["component"],
        row["segment"],
    )


def message_row_from_key(
    key: tuple[object, ...],
    *,
    scale: int,
    gpus: int,
    kind: str,
    message_bytes: float,
    observed_messages: int,
    expected_messages: int,
    minimum_bytes: float,
    maximum_bytes: float,
    total_cross_bytes_all_ranks: float,
    bandwidth_fit: BandwidthFit,
    calibrated_min: float,
    calibrated_max: float,
    source: Path,
) -> dict[str, object]:
    (
        query,
        step_index,
        step_kind,
        step_label,
        phase_index,
        phase_name,
        pattern,
        primitive,
        column_ordinal,
        column_type,
        component,
        segment,
    ) = key
    average_cross_bytes = total_cross_bytes_all_ranks / gpus
    achievable_gbps = bandwidth_gbps(bandwidth_fit, message_bytes)
    predicted_ms = (
        average_cross_bytes / (achievable_gbps * BYTES_PER_GB) * 1_000.0
        if average_cross_bytes > 0.0
        else 0.0
    )
    return {
        "scale_factor": scale,
        "gpus": gpus,
        "query": query,
        "communication_kind": kind,
        "step_index": step_index,
        "step_kind": step_kind,
        "step_label": step_label,
        "phase_index": phase_index,
        "phase_name": phase_name,
        "communication_pattern": pattern,
        "collective_primitive": primitive,
        "column_ordinal": column_ordinal,
        "column_type": column_type,
        "component": component,
        "segment": segment,
        "average_per_peer_message_bytes": message_bytes,
        "minimum_message_bytes": minimum_bytes,
        "maximum_message_bytes": maximum_bytes,
        "observed_messages": observed_messages,
        "expected_messages": expected_messages,
        "total_cross_bytes_all_ranks": total_cross_bytes_all_ranks,
        "average_cross_bytes_per_gpu": average_cross_bytes,
        "achievable_per_gpu_cross_gbps": achievable_gbps,
        "predicted_time_ms": predicted_ms,
        "outside_calibrated_message_range": int(
            message_bytes > 0.0
            and (message_bytes < calibrated_min or message_bytes > calibrated_max)
        ),
        "source_messages_csv": relative_to_repo(source),
    }


def extract_shuffle_columns(
    rows: Iterable[Mapping[str, str]],
    *,
    scale: int,
    gpus: int,
    bandwidth_fit: BandwidthFit,
    calibrated_min: float,
    calibrated_max: float,
    source: Path,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], dict[tuple[int, int], int]] = defaultdict(dict)
    for row in rows:
        if row["communication_pattern"] != "shuffle" or row["api"] != "ncclSend":
            continue
        if row["direction"] != "send":
            raise ValueError(f"{source}: ncclSend row is not marked send")
        if row["is_self"] == "1":
            continue
        key = physical_message_key(row)
        pair = (int(row["rank"]), int(row["peer"]))
        payload = int(row["payload_bytes"])
        if pair in groups[key]:
            raise ValueError(f"{source}: duplicate shuffle send for {key}, pair {pair}")
        groups[key][pair] = payload

    output = []
    expected = gpus * (gpus - 1)
    for key, pair_payloads in sorted(groups.items()):
        if any(rank == peer for rank, peer in pair_payloads):
            raise ValueError(f"{source}: self pair survived shuffle filtering")
        if any(not (0 <= rank < gpus and 0 <= peer < gpus) for rank, peer in pair_payloads):
            raise ValueError(f"{source}: invalid shuffle rank/peer")
        values = list(pair_payloads.values())
        if len(values) > expected:
            raise ValueError(f"{source}: too many shuffle messages for {key}")
        total = float(sum(values))
        average = total / expected
        output.append(
            message_row_from_key(
                key,
                scale=scale,
                gpus=gpus,
                kind="shuffle",
                message_bytes=average,
                observed_messages=len(values),
                expected_messages=expected,
                minimum_bytes=0.0 if len(values) < expected else float(min(values)),
                maximum_bytes=float(max(values)),
                total_cross_bytes_all_ranks=total,
                bandwidth_fit=bandwidth_fit,
                calibrated_min=calibrated_min,
                calibrated_max=calibrated_max,
                source=source,
            )
        )
    return output


def extract_allgather_columns(
    rows: Iterable[Mapping[str, str]],
    *,
    scale: int,
    gpus: int,
    bandwidth_fit: BandwidthFit,
    calibrated_min: float,
    calibrated_max: float,
    source: Path,
) -> list[dict[str, object]]:
    groups: dict[
        tuple[object, ...], dict[int, dict[int, int]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if (
            row["communication_pattern"] not in MESSAGE_PATTERNS_FOR_ALLGATHER
            or row["api"] != "ncclBroadcast"
        ):
            continue
        key = physical_message_key(row)
        root = int(row["root"])
        observer = int(row["rank"])
        payload = int(row["payload_bytes"])
        if observer in groups[key][root]:
            raise ValueError(
                f"{source}: duplicate all-gather observation for {key}, root {root}, rank {observer}"
            )
        groups[key][root][observer] = payload

    output = []
    for key, roots in sorted(groups.items()):
        root_payloads: dict[int, int] = {}
        observed_records = 0
        for root, observers in roots.items():
            if not 0 <= root < gpus:
                raise ValueError(f"{source}: invalid all-gather root {root}")
            if set(observers) != set(range(gpus)):
                raise ValueError(
                    f"{source}: incomplete all-gather observations for {key}, root {root}"
                )
            unique_payloads = set(observers.values())
            if len(unique_payloads) != 1:
                raise ValueError(
                    f"{source}: ranks disagree on all-gather payload for {key}, root {root}"
                )
            root_payloads[root] = next(iter(unique_payloads))
            observed_records += len(observers)
        if len(root_payloads) > gpus:
            raise ValueError(f"{source}: too many all-gather roots for {key}")
        values = list(root_payloads.values())
        average = float(sum(values)) / gpus
        total_cross = float(gpus - 1) * sum(values)
        output.append(
            message_row_from_key(
                key,
                scale=scale,
                gpus=gpus,
                kind="allgather",
                message_bytes=average,
                observed_messages=len(values),
                expected_messages=gpus,
                minimum_bytes=0.0 if len(values) < gpus else float(min(values)),
                maximum_bytes=float(max(values)),
                total_cross_bytes_all_ranks=total_cross,
                bandwidth_fit=bandwidth_fit,
                calibrated_min=calibrated_min,
                calibrated_max=calibrated_max,
                source=source,
            )
        )
        if observed_records < len(values):
            raise ValueError(f"{source}: invalid all-gather observer count")
    return output


def load_message_columns(
    results_dir: Path,
    scale: int,
    gpu_counts: Sequence[int],
    message_arm: str,
    bandwidth_fit: BandwidthFit,
    calibrated_min: float,
    calibrated_max: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    columns: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for gpus in gpu_counts:
        root = message_campaign_root(results_dir, scale, gpus, message_arm)
        for query in QUERIES:
            query_root = root / f"q{query:02d}"
            messages = successful_messages_csv(query_root)
            if messages is None:
                raise ValueError(f"no successful messages.csv under {query_root}")
            with messages.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                raise ValueError(f"{messages}: empty message trace")
            ranks = {int(row["ranks"]) for row in rows}
            scales = {int(row["scale"]) for row in rows}
            queries = {query_number(row["query"]) for row in rows}
            if ranks != {gpus} or scales != {scale} or queries != {query}:
                raise ValueError(f"{messages}: campaign identity mismatch")
            shuffle = extract_shuffle_columns(
                rows,
                scale=scale,
                gpus=gpus,
                bandwidth_fit=bandwidth_fit,
                calibrated_min=calibrated_min,
                calibrated_max=calibrated_max,
                source=messages,
            )
            allgather = extract_allgather_columns(
                rows,
                scale=scale,
                gpus=gpus,
                bandwidth_fit=bandwidth_fit,
                calibrated_min=calibrated_min,
                calibrated_max=calibrated_max,
                source=messages,
            )
            columns.extend(shuffle)
            columns.extend(allgather)
            patterns = defaultdict(int)
            for row in rows:
                patterns[row["communication_pattern"]] += 1
            coverage.append(
                {
                    "scale_factor": scale,
                    "gpus": gpus,
                    "query": query,
                    "shuffle_column_messages": len(shuffle),
                    "allgather_column_messages": len(allgather),
                    "raw_message_records": len(rows),
                    "raw_shuffle_records": patterns["shuffle"],
                    "raw_broadcast_records": patterns["broadcast"],
                    "raw_sampling_records": patterns["sampling"],
                    "excluded_gather_records": patterns["gather"],
                    "excluded_reduce_records": patterns["reduce"],
                    "source_messages_csv": relative_to_repo(messages),
                }
            )
    return columns, coverage


def message_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["query"]),
        str(row["communication_kind"]),
        int(row["step_index"]),
        str(row["step_kind"]),
        str(row["step_label"]),
        int(row["phase_index"]),
        str(row["phase_name"]),
        str(row["communication_pattern"]),
        str(row["collective_primitive"]),
        int(row["column_ordinal"]),
        str(row["column_type"]),
        str(row["component"]),
        str(row["segment"]),
    )


def project_message_columns(
    measured_columns: Sequence[Mapping[str, object]],
    *,
    anchor_gpus: int,
    prediction_gpu_counts: Sequence[int],
    bandwidth_fit: BandwidthFit,
    calibrated_min: float,
    calibrated_max: float,
) -> list[dict[str, object]]:
    anchor_rows = [row for row in measured_columns if int(row["gpus"]) == anchor_gpus]
    if not anchor_rows:
        raise ValueError(f"no measured message columns for {anchor_gpus}-GPU anchor")
    identities = [message_identity(row) for row in anchor_rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate physical message identities at {anchor_gpus} GPUs")

    projected_rows: list[dict[str, object]] = []
    for target_gpus in prediction_gpu_counts:
        ratio = anchor_gpus / target_gpus
        for anchor in anchor_rows:
            kind = str(anchor["communication_kind"])
            exponent = 2 if kind == "shuffle" else 1
            scale = ratio**exponent
            average_message = float(anchor["average_per_peer_message_bytes"]) * scale
            minimum_message = float(anchor["minimum_message_bytes"]) * scale
            maximum_message = float(anchor["maximum_message_bytes"]) * scale
            average_cross = (target_gpus - 1) * average_message
            total_cross = target_gpus * average_cross
            achievable_gbps = bandwidth_gbps(bandwidth_fit, average_message)
            predicted_ms = (
                average_cross / (achievable_gbps * BYTES_PER_GB) * 1_000.0
                if average_cross > 0.0
                else 0.0
            )
            projected = dict(anchor)
            observed_at_anchor = projected.pop("observed_messages")
            expected_at_anchor = projected.pop("expected_messages")
            projected.update(
                {
                    "gpus": target_gpus,
                    "anchor_gpus": anchor_gpus,
                    "message_size_source": (
                        "measured_anchor" if target_gpus == anchor_gpus else "projected_from_anchor"
                    ),
                    "message_size_scaling_exponent": exponent,
                    "anchor_average_per_peer_message_bytes": float(
                        anchor["average_per_peer_message_bytes"]
                    ),
                    "average_per_peer_message_bytes": average_message,
                    "minimum_message_bytes": minimum_message,
                    "maximum_message_bytes": maximum_message,
                    "observed_messages_at_anchor": observed_at_anchor,
                    "expected_messages_at_anchor": expected_at_anchor,
                    "total_cross_bytes_all_ranks": total_cross,
                    "average_cross_bytes_per_gpu": average_cross,
                    "achievable_per_gpu_cross_gbps": achievable_gbps,
                    "predicted_time_ms": predicted_ms,
                    "outside_calibrated_message_range": int(
                        average_message > 0.0
                        and (
                            average_message < calibrated_min
                            or average_message > calibrated_max
                        )
                    ),
                }
            )
            projected_rows.append(projected)
    return projected_rows


def validate_message_projection(
    measured_columns: Sequence[Mapping[str, object]],
    projected_columns: Sequence[Mapping[str, object]],
    validation_gpu_counts: Sequence[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    projected = {
        (int(row["gpus"]), message_identity(row)): row for row in projected_columns
    }
    detailed: list[dict[str, object]] = []
    for actual in measured_columns:
        gpus = int(actual["gpus"])
        if gpus not in validation_gpu_counts:
            continue
        identity = message_identity(actual)
        prediction = projected.get((gpus, identity))
        if prediction is None:
            raise ValueError(f"missing projected message for {gpus} GPUs: {identity}")
        actual_message = float(actual["average_per_peer_message_bytes"])
        predicted_message = float(prediction["average_per_peer_message_bytes"])
        if actual_message <= 0.0:
            raise ValueError(f"non-positive measured message for {gpus} GPUs: {identity}")
        detailed.append(
            {
                "scale_factor": int(actual["scale_factor"]),
                "gpus": gpus,
                "query": int(actual["query"]),
                "communication_kind": actual["communication_kind"],
                "step_index": actual["step_index"],
                "step_label": actual["step_label"],
                "phase_index": actual["phase_index"],
                "phase_name": actual["phase_name"],
                "column_ordinal": actual["column_ordinal"],
                "component": actual["component"],
                "segment": actual["segment"],
                "measured_average_per_peer_message_bytes": actual_message,
                "projected_average_per_peer_message_bytes": predicted_message,
                "projected_to_measured_ratio": predicted_message / actual_message,
                "absolute_percentage_error": 100.0
                * abs(predicted_message - actual_message)
                / actual_message,
                "measured_average_cross_bytes_per_gpu": float(
                    actual["average_cross_bytes_per_gpu"]
                ),
                "projected_average_cross_bytes_per_gpu": float(
                    prediction["average_cross_bytes_per_gpu"]
                ),
                "measured_size_modeled_time_ms": float(actual["predicted_time_ms"]),
                "projected_size_modeled_time_ms": float(prediction["predicted_time_ms"]),
                "source_messages_csv": actual["source_messages_csv"],
            }
        )

    expected = len(validation_gpu_counts) * len(
        [row for row in projected_columns if int(row["gpus"]) == validation_gpu_counts[0]]
    )
    if len(detailed) != expected:
        raise ValueError(
            f"message projection validation has {len(detailed)} rows, expected {expected}"
        )
    summary: list[dict[str, object]] = []
    for gpus in validation_gpu_counts:
        for kind in ("shuffle", "allgather"):
            rows = [
                row
                for row in detailed
                if int(row["gpus"]) == gpus and row["communication_kind"] == kind
            ]
            errors = [float(row["absolute_percentage_error"]) for row in rows]
            ratios = [float(row["projected_to_measured_ratio"]) for row in rows]
            measured_bytes = sum(
                float(row["measured_average_per_peer_message_bytes"]) for row in rows
            )
            projected_bytes = sum(
                float(row["projected_average_per_peer_message_bytes"]) for row in rows
            )
            absolute_byte_error = sum(
                abs(
                    float(row["projected_average_per_peer_message_bytes"])
                    - float(row["measured_average_per_peer_message_bytes"])
                )
                for row in rows
            )
            measured_modeled_time = sum(
                float(row["measured_size_modeled_time_ms"]) for row in rows
            )
            projected_modeled_time = sum(
                float(row["projected_size_modeled_time_ms"]) for row in rows
            )
            summary.append(
                {
                    "gpus": gpus,
                    "communication_kind": kind,
                    "physical_column_messages": len(rows),
                    "median_projected_to_measured_ratio": statistics.median(ratios),
                    "mean_absolute_percentage_error": statistics.fmean(errors),
                    "median_absolute_percentage_error": statistics.median(errors),
                    "p95_absolute_percentage_error": float(np.percentile(errors, 95)),
                    "aggregate_projected_to_measured_bytes_ratio": projected_bytes
                    / measured_bytes,
                    "byte_weighted_absolute_percentage_error": 100.0
                    * absolute_byte_error
                    / measured_bytes,
                    "aggregate_projected_to_measured_modeled_time_ratio": projected_modeled_time
                    / measured_modeled_time,
                }
            )
    return detailed, summary


def assemble_predictions(
    compute_samples: Sequence[ComputeSample],
    compute_fits: Sequence[ComputeFit],
    message_columns: Sequence[Mapping[str, object]],
    prediction_gpu_counts: Sequence[int],
    scale_factor: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sample_by_key = {(row.query, row.gpus): row for row in compute_samples}
    fit_by_query = {row.query: row for row in compute_fits}
    comm: dict[tuple[int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in message_columns:
        comm[(int(row["query"]), int(row["gpus"]), str(row["communication_kind"]))].append(row)

    query_rows: list[dict[str, object]] = []
    maximum_training_gpus = max(row.gpus for row in compute_samples)
    for gpus in prediction_gpu_counts:
        for query in QUERIES:
            sample = sample_by_key.get((query, gpus))
            fit = fit_by_query[query]
            predicted_compute = power_prediction(fit.alpha_ms, fit.beta, gpus)
            shuffle_rows = comm[(query, gpus, "shuffle")]
            allgather_rows = comm[(query, gpus, "allgather")]
            predicted_shuffle = sum(
                float(row["predicted_time_ms"]) for row in shuffle_rows
            )
            predicted_allgather = sum(
                float(row["predicted_time_ms"]) for row in allgather_rows
            )
            predicted_communication = predicted_shuffle + predicted_allgather
            modeled_total = predicted_compute + predicted_communication
            query_rows.append(
                {
                    "scale_factor": scale_factor,
                    "gpus": gpus,
                    "query": query,
                    "is_compute_extrapolation": int(gpus > maximum_training_gpus),
                    "measured_legacy_e2e_ms": (
                        sample.measured_e2e_ms if sample is not None else ""
                    ),
                    "measured_compute_ms": (
                        sample.measured_compute_ms if sample is not None else ""
                    ),
                    "predicted_compute_ms": predicted_compute,
                    "compute_error_percent": (
                        100.0
                        * (predicted_compute - sample.measured_compute_ms)
                        / sample.measured_compute_ms
                        if sample is not None
                        else ""
                    ),
                    "measured_legacy_communication_ms": (
                        sample.measured_legacy_communication_ms if sample is not None else ""
                    ),
                    "predicted_ce_shuffle_ms": predicted_shuffle,
                    "predicted_ce_allgather_ms": predicted_allgather,
                    "predicted_ce_shuffle_allgather_ms": predicted_communication,
                    "legacy_comm_minus_ce_model_ms": (
                        sample.measured_legacy_communication_ms - predicted_communication
                        if sample is not None
                        else ""
                    ),
                    "modeled_compute_plus_ce_communication_ms": modeled_total,
                    "shuffle_physical_column_messages": len(shuffle_rows),
                    "allgather_physical_column_messages": len(allgather_rows),
                    "out_of_calibration_column_messages": sum(
                        int(row["outside_calibrated_message_range"])
                        for row in shuffle_rows + allgather_rows
                    ),
                    "compute_alpha_ms": fit.alpha_ms,
                    "compute_beta": fit.beta,
                }
            )

    aggregate_rows: list[dict[str, object]] = []
    for gpus in sorted({int(row["gpus"]) for row in query_rows}):
        rows = [row for row in query_rows if int(row["gpus"]) == gpus]
        if len(rows) != len(QUERIES):
            raise ValueError(f"{gpus} GPUs has {len(rows)} query predictions")
        measured = all(row["measured_legacy_e2e_ms"] != "" for row in rows)
        measured_e2e = (
            sum(float(row["measured_legacy_e2e_ms"]) for row in rows)
            if measured
            else ""
        )
        predicted_total = sum(
            float(row["modeled_compute_plus_ce_communication_ms"]) for row in rows
        )
        aggregate_rows.append(
            {
                "scale_factor": int(rows[0]["scale_factor"]),
                "gpus": gpus,
                "query_count": len(rows),
                "is_compute_extrapolation": int(
                    any(int(row["is_compute_extrapolation"]) for row in rows)
                ),
                "measured_legacy_e2e_ms": measured_e2e,
                "measured_compute_ms": (
                    sum(float(row["measured_compute_ms"]) for row in rows)
                    if measured
                    else ""
                ),
                "predicted_compute_ms": sum(float(row["predicted_compute_ms"]) for row in rows),
                "measured_legacy_communication_ms": (
                    sum(float(row["measured_legacy_communication_ms"]) for row in rows)
                    if measured
                    else ""
                ),
                "predicted_ce_shuffle_ms": sum(
                    float(row["predicted_ce_shuffle_ms"]) for row in rows
                ),
                "predicted_ce_allgather_ms": sum(
                    float(row["predicted_ce_allgather_ms"]) for row in rows
                ),
                "predicted_ce_shuffle_allgather_ms": sum(
                    float(row["predicted_ce_shuffle_allgather_ms"]) for row in rows
                ),
                "modeled_compute_plus_ce_communication_ms": predicted_total,
                "modeled_total_vs_measured_legacy_percent": (
                    100.0 * (predicted_total - float(measured_e2e)) / float(measured_e2e)
                    if measured
                    else ""
                ),
                "shuffle_physical_column_messages": sum(
                    int(row["shuffle_physical_column_messages"]) for row in rows
                ),
                "allgather_physical_column_messages": sum(
                    int(row["allgather_physical_column_messages"]) for row in rows
                ),
                "out_of_calibration_column_messages": sum(
                    int(row["out_of_calibration_column_messages"]) for row in rows
                ),
            }
        )
    return query_rows, aggregate_rows


def plot_bandwidth_fit(
    samples: Sequence[Mapping[str, object]],
    fits: Sequence[BandwidthFit],
    output_dir: Path,
) -> None:
    pooled = next(fit for fit in fits if fit.scope == "pooled")
    by_scope = {fit.scope: fit for fit in fits}
    figure, axis = plt.subplots(figsize=(10.5, 6.6))
    for gpu_count, color in ((8, COLORS["g8"]), (24, COLORS["g24"])):
        rows = sorted(
            (row for row in samples if int(row["gpus"]) == gpu_count),
            key=lambda row: int(row["actual_per_peer_bytes"]),
        )
        axis.scatter(
            [float(row["actual_per_peer_bytes"]) / (1024.0**2) for row in rows],
            [float(row["observed_per_gpu_cross_gbps"]) for row in rows],
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=f"{gpu_count} GPUs measured",
            zorder=3,
        )
    minimum = min(float(row["actual_per_peer_bytes"]) for row in samples)
    maximum = max(float(row["actual_per_peer_bytes"]) for row in samples)
    grid = np.geomspace(minimum, maximum, 400)
    axis.plot(
        grid / (1024.0**2),
        [bandwidth_gbps(pooled, float(value)) for value in grid],
        color=COLORS["pooled"],
        linewidth=2.8,
        label="Pooled fit",
    )
    for scope, color in (("g8", COLORS["g8"]), ("g24", COLORS["g24"])):
        if scope not in by_scope:
            continue
        axis.plot(
            grid / (1024.0**2),
            [bandwidth_gbps(by_scope[scope], float(value)) for value in grid],
            color=color,
            linewidth=1.4,
            linestyle="--",
            alpha=0.8,
            label=f"{scope[1:]}-GPU diagnostic fit",
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Per-peer message size (MiB)")
    axis.set_ylabel("Per-GPU cross bandwidth (GB/s)")
    axis.set_title("CE-Aligned NVLink Bandwidth Model", pad=16)
    axis.grid(axis="both", linestyle=(0, (2, 3)), linewidth=0.9, alpha=0.45)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.text(
        0.025,
        0.965,
        "$B(m)=m/(a+bm)$\n"
        f"Saturation = {pooled.saturation_gbps:.1f} GB/s\n"
        f"MAPE = {pooled.mape_percent:.1f}%",
        transform=axis.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "distributed_tpch_strong_scaling_busbw_fit")


def plot_compute_fits(
    samples: Sequence[ComputeSample],
    fits: Sequence[ComputeFit],
    output_dir: Path,
) -> None:
    sample_groups: dict[int, list[ComputeSample]] = defaultdict(list)
    for row in samples:
        sample_groups[row.query].append(row)
    fit_by_query = {row.query: row for row in fits}
    figure, axes = plt.subplots(4, 6, figsize=(18, 12), sharex=True)
    for query, axis in zip(QUERIES, axes.flat):
        rows = sorted(sample_groups[query], key=lambda row: row.gpus)
        fit = fit_by_query[query]
        x_values = np.geomspace(min(row.gpus for row in rows), max(row.gpus for row in rows), 120)
        axis.scatter(
            [row.gpus for row in rows],
            [row.measured_compute_ms for row in rows],
            color=COLORS["compute"],
            s=30,
            zorder=3,
        )
        axis.plot(
            x_values,
            [power_prediction(fit.alpha_ms, fit.beta, float(n)) for n in x_values],
            color="#222222",
            linewidth=1.5,
        )
        axis.set_title(f"Q{query}  ($b$={fit.beta:.2f})", fontsize=12, pad=5)
        axis.grid(axis="y", linestyle=(0, (2, 3)), alpha=0.35)
        axis.set_xticks(DEFAULT_TRAINING_GPU_COUNTS)
    for axis in axes.flat[len(QUERIES):]:
        axis.axis("off")
    figure.suptitle("Per-Query Compute-Time Power Models", fontsize=22, y=0.995)
    figure.supxlabel("Number of GPUs", fontsize=16)
    figure.supylabel("Non-communication time (ms)", fontsize=16)
    figure.tight_layout(rect=(0.03, 0.03, 1.0, 0.97))
    save_figure(figure, output_dir, "distributed_tpch_strong_scaling_compute_fits")


def plot_aggregate_model(
    aggregate_rows: Sequence[Mapping[str, object]], output_dir: Path
) -> None:
    rows = sorted(aggregate_rows, key=lambda row: int(row["gpus"]))
    x = np.arange(len(rows), dtype=float)
    compute = np.asarray([float(row["predicted_compute_ms"]) / 1000.0 for row in rows])
    shuffle = np.asarray([float(row["predicted_ce_shuffle_ms"]) / 1000.0 for row in rows])
    allgather = np.asarray([float(row["predicted_ce_allgather_ms"]) / 1000.0 for row in rows])
    measured = np.asarray(
        [
            float(row["measured_legacy_e2e_ms"]) / 1000.0
            if row["measured_legacy_e2e_ms"] != ""
            else np.nan
            for row in rows
        ]
    )

    figure, axis = plt.subplots(figsize=(13.0, 7.0))
    width = 0.68
    axis.bar(x, compute, width, color=COLORS["compute"], label="Compute model")
    axis.bar(x, shuffle, width, bottom=compute, color=COLORS["shuffle"], label="Shuffle model")
    axis.bar(
        x,
        allgather,
        width,
        bottom=compute + shuffle,
        color=COLORS["allgather"],
        label="All-gather model",
    )
    measured_mask = np.isfinite(measured)
    axis.plot(
        x[measured_mask],
        measured[measured_mask],
        color=COLORS["measured"],
        marker="D",
        markersize=7,
        linewidth=2.0,
        label="Measured",
        zorder=4,
    )
    totals = compute + shuffle + allgather
    for position, total in zip(x, totals):
        axis.text(
            position,
            total + max(totals) * 0.018,
            f"{total:.2f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )
    training_positions = [
        index for index, row in enumerate(rows) if not int(row["is_compute_extrapolation"])
    ]
    if training_positions and training_positions[-1] < len(rows) - 1:
        boundary = training_positions[-1] + 0.5
        axis.axvspan(boundary, len(rows) - 0.5, color="#888888", alpha=0.065, zorder=0)
        axis.axvline(boundary, color="#888888", linestyle="--", linewidth=1.1)
        axis.text(
            boundary + 0.18,
            0.82,
            "Predicted by model",
            transform=axis.get_xaxis_transform(),
            ha="left",
            va="top",
            color="#555555",
            fontsize=12,
        )
    axis.set_xticks(x, [str(int(row["gpus"])) for row in rows])
    axis.set_xlabel("Number of GPUs")
    axis.set_ylabel("Aggregate time across 22 queries (s)")
    scale_factor = int(rows[0]["scale_factor"])
    axis.set_title(
        f"TPC-H Strong-Scaling Performance Model — SF={scale_factor}", pad=18
    )
    axis.grid(axis="y", linestyle=(0, (2, 3)), linewidth=0.9, alpha=0.45)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.set_ylim(0.0, max(max(totals), float(np.nanmax(measured))) * 1.20)
    for text_artist in figure.findobj(match=matplotlib.text.Text):
        text_artist.set_fontsize(text_artist.get_fontsize() * 1.5)
    figure.tight_layout()
    save_figure(figure, output_dir, "distributed_tpch_strong_scaling_model")


def plot_model_validation(
    query_rows: Sequence[Mapping[str, object]], output_dir: Path
) -> None:
    measured_rows = [row for row in query_rows if row["measured_compute_ms"] != ""]
    observed_compute = np.asarray(
        [float(row["measured_compute_ms"]) for row in measured_rows]
    )
    predicted_compute = np.asarray(
        [float(row["predicted_compute_ms"]) for row in measured_rows]
    )
    observed_comm = np.asarray(
        [float(row["measured_legacy_communication_ms"]) for row in measured_rows]
    )
    predicted_comm = np.asarray(
        [float(row["predicted_ce_shuffle_allgather_ms"]) for row in measured_rows]
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.7))
    panels = (
        (
            axes[0],
            observed_compute,
            predicted_compute,
            COLORS["compute"],
            "Compute fit",
            "Measured non-communication time (ms)",
            "Power-model prediction (ms)",
        ),
        (
            axes[1],
            observed_comm,
            predicted_comm,
            COLORS["shuffle"],
            "Communication context",
            "Measured legacy communication (ms)",
            "CE shuffle + all-gather model (ms)",
        ),
    )
    for axis, observed, predicted, color, title, xlabel, ylabel in panels:
        positive = np.concatenate((observed[observed > 0], predicted[predicted > 0]))
        lower = max(float(np.min(positive)) * 0.75, 1e-3)
        upper = float(np.max(positive)) * 1.35
        axis.scatter(observed, predicted, color=color, alpha=0.78, s=38, edgecolor="white", linewidth=0.6)
        axis.plot([lower, upper], [lower, upper], color="#666666", linestyle="--", linewidth=1.4)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(which="both", linestyle=(0, (2, 3)), alpha=0.35)
    axes[1].text(
        0.04,
        0.96,
        "Context only: different backends\nand gather/reduce are excluded",
        transform=axes[1].transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
        fontsize=10,
    )
    figure.suptitle("TPC-H Strong-Scaling Model Diagnostics", fontsize=20, y=1.01)
    figure.tight_layout()
    save_figure(figure, output_dir, "distributed_tpch_strong_scaling_validation")


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    pooled_fit: BandwidthFit,
    calibrated_min: float,
    calibrated_max: float,
) -> None:
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": relative_to_repo(Path(__file__)),
        "scale_factor": args.scale,
        "training_gpu_counts": list(args.training_gpu_counts),
        "prediction_gpu_counts": list(args.prediction_gpu_counts),
        "message_anchor_gpus": args.message_anchor_gpus,
        "queries": list(QUERIES),
        "timing_arm": args.timing_arm,
        "message_arm": args.message_arm,
        "compute_definition": (
            "clean median rank-max E2E multiplied by one minus the median direct "
            "communication interval-union fraction on the critical rank; includes "
            "hash partitioning and non-communication overhead"
        ),
        "compute_model": "C_q(N) = alpha_q * N ** beta_q",
        "bandwidth_source": relative_to_repo(args.bandwidth_campaign / "summary.csv"),
        "bandwidth_arm": CE_ARM,
        "bandwidth_metric": BUS_BW_FIELD,
        "bandwidth_model": "B(m) = m / (a + b*m), B is per-GPU cross bandwidth",
        "bandwidth_fit": asdict(pooled_fit),
        "calibrated_message_bytes": [calibrated_min, calibrated_max],
        "shuffle_average": (
            "mean off-rank ncclSend payload over all N*(N-1) directed rank pairs; "
            "missing calls count as zero"
        ),
        "allgather_average": (
            "mean ncclBroadcast root contribution over N roots after de-duplicating "
            "the same root payload observed on every rank; missing roots count as zero"
        ),
        "message_size_projection": {
            "shuffle": "m(N) = m(anchor) * (anchor/N)^2",
            "allgather": "m(N) = m(anchor) * (anchor/N)",
        },
        "column_time_model": "t = average_cross_bytes_per_gpu / B(average_per_peer_bytes)",
        "included_communication_patterns": ["shuffle", "broadcast", "sampling"],
        "excluded_communication_patterns": ["gather", "reduce", "other"],
        "composition": "T_q = predicted_noncommunication + predicted_shuffle + predicted_allgather",
        "overlap_assumption": "none; physical column collective calls are serialized",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {path}")


def write_report(
    path: Path,
    args: argparse.Namespace,
    bandwidth_fit: BandwidthFit,
    compute_fits: Sequence[ComputeFit],
    message_columns: Sequence[Mapping[str, object]],
    message_validation_summary: Sequence[Mapping[str, object]],
    aggregate_rows: Sequence[Mapping[str, object]],
    calibrated_min: float,
    calibrated_max: float,
) -> None:
    median_training_mape = statistics.median(
        fit.training_mape_percent for fit in compute_fits
    )
    median_loo_mape = statistics.median(fit.loo_mape_percent for fit in compute_fits)
    lines = [
        "# Distributed TPC-H strong-scaling model",
        "",
        "This report is generated by `tools/visualize/model_distributed_tpch_strong_scaling.py`.",
        "The model predicts each query independently and sums the 22 predictions.",
        "",
        "## Definitions",
        "",
        "- Compute/non-communication time: clean median rank-max E2E multiplied by one minus the median direct-communication interval-union fraction on the critical rank. It includes hash partitioning and non-communication overhead.",
        "- Per-query compute fit: `C_q(N) = alpha_q * N^beta_q`.",
        "- CE bandwidth fit: `B(m) = m / (a + b*m)`, with `B` defined as per-GPU cross-fabric egress bandwidth.",
        "- Shuffle size: average off-rank send payload over all `N*(N-1)` directed rank pairs; receive and self records are excluded.",
        "- All-gather size: average root contribution after de-duplicating the same broadcast call observed on every rank.",
        "- Column time: `average_cross_bytes_per_gpu / B(average_per_peer_message_bytes)`.",
        "- Total: predicted compute + predicted shuffle + predicted broadcast/sampling all-gather. No overlap is assumed.",
        "",
        "## Fit summary",
        "",
        f"- Bandwidth fit: a = `{bandwidth_fit.a_seconds:.9g}` s, b = `{bandwidth_fit.b_seconds_per_byte:.9g}` s/byte.",
        f"- Fitted saturation: {bandwidth_fit.saturation_gbps:.1f} GB/s; log-space R² = {bandwidth_fit.r2_log:.4f}; MAPE = {bandwidth_fit.mape_percent:.1f}%.",
        f"- Compute fits: median training MAPE = {median_training_mape:.2f}%; median leave-one-GPU-count-out MAPE = {median_loo_mape:.2f}%.",
        "",
        "## Message-size projection validation",
        "",
        f"Message sizes are projected from the {args.message_anchor_gpus}-GPU traces and checked against the measured training-count traces.",
        "",
        "| GPUs | Kind | Aggregate byte ratio | Byte-weighted absolute error | Modeled-time ratio |",
        "|---:|:---|---:|---:|---:|",
    ]
    for row in message_validation_summary:
        lines.append(
            "| {gpus} | {kind} | {byte_ratio:.4f} | {byte_error:.2f}% | {time_ratio:.4f} |".format(
                gpus=int(row["gpus"]),
                kind=row["communication_kind"],
                byte_ratio=float(row["aggregate_projected_to_measured_bytes_ratio"]),
                byte_error=float(row["byte_weighted_absolute_percentage_error"]),
                time_ratio=float(
                    row["aggregate_projected_to_measured_modeled_time_ratio"]
                ),
            )
        )
    lines.extend(
        [
        "",
        "## Aggregate predictions",
        "",
        "| GPUs | Pred. compute (s) | Pred. shuffle (s) | Pred. all-gather (s) | Modeled total (s) | Measured legacy E2E (s) |",
        "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(aggregate_rows, key=lambda item: int(item["gpus"])):
        measured = (
            f"{float(row['measured_legacy_e2e_ms']) / 1000.0:.3f}"
            if row["measured_legacy_e2e_ms"] != ""
            else "—"
        )
        lines.append(
            "| {gpus} | {compute:.3f} | {shuffle:.3f} | {allgather:.3f} | {total:.3f} | {measured} |".format(
                gpus=int(row["gpus"]),
                compute=float(row["predicted_compute_ms"]) / 1000.0,
                shuffle=float(row["predicted_ce_shuffle_ms"]) / 1000.0,
                allgather=float(row["predicted_ce_allgather_ms"]) / 1000.0,
                total=float(row["modeled_compute_plus_ce_communication_ms"]) / 1000.0,
                measured=measured,
            )
        )
    lines.extend(
        [
            "",
            "## Calibration coverage and interpretation",
            "",
            f"The CE curve is calibrated over {calibrated_min / 1024.0:.0f} KiB to {calibrated_max / (1024.0**2):.0f} MiB. The formula is evaluated outside that interval, but those values are extrapolations.",
            "",
            "| GPUs | Below range | Above range | Outside-range share of modeled communication time |",
            "|---:|---:|---:|---:|",
        ]
    )
    for gpus in args.prediction_gpu_counts:
        rows = [row for row in message_columns if int(row["gpus"]) == gpus]
        below = [
            row
            for row in rows
            if 0.0 < float(row["average_per_peer_message_bytes"]) < calibrated_min
        ]
        above = [
            row
            for row in rows
            if float(row["average_per_peer_message_bytes"]) > calibrated_max
        ]
        total_time = sum(float(row["predicted_time_ms"]) for row in rows)
        outside_time = sum(float(row["predicted_time_ms"]) for row in below + above)
        share = 100.0 * outside_time / total_time if total_time > 0.0 else 0.0
        lines.append(f"| {gpus} | {len(below)} | {len(above)} | {share:.1f}% |")
    lines.extend(
        [
            "",
            "The modeled communication uses the CE-aligned bandwidth curve, while the comparison line is measured `legacy_cta64` NCCL end-to-end time. Their difference is therefore not a same-backend prediction error.",
            f"Message sizes are anchored at {args.message_anchor_gpus} GPUs. Shuffle messages scale as `(anchor/N)^2`; all-gather contributions scale as `anchor/N`.",
            f"Compute values above {max(args.training_gpu_counts)} GPUs and all topology behavior beyond the measured bandwidth campaigns are extrapolations.",
            "Gather-to-root, all-reduce, and unclassified communication are excluded. Their raw-record coverage is preserved in `distributed_tpch_strong_scaling_message_coverage.csv`.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python3 tools/visualize/model_distributed_tpch_strong_scaling.py",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.message_anchor_gpus not in args.training_gpu_counts:
            raise ValueError("message anchor must be one of the training GPU counts")
        if args.message_anchor_gpus not in args.prediction_gpu_counts:
            raise ValueError("message anchor must be one of the prediction GPU counts")
        compute_samples = load_compute_samples(
            args.results_dir,
            args.scale,
            args.training_gpu_counts,
            args.timing_arm,
        )
        compute_fits = fit_compute_models(compute_samples)
        bandwidth_samples, bandwidth_fits, calibrated_min, calibrated_max = (
            load_bandwidth_model(args.bandwidth_campaign)
        )
        pooled_fit = next(fit for fit in bandwidth_fits if fit.scope == "pooled")
        measured_message_columns, message_coverage = load_message_columns(
            args.results_dir,
            args.scale,
            args.training_gpu_counts,
            args.message_arm,
            pooled_fit,
            calibrated_min,
            calibrated_max,
        )
        message_columns = project_message_columns(
            measured_message_columns,
            anchor_gpus=args.message_anchor_gpus,
            prediction_gpu_counts=args.prediction_gpu_counts,
            bandwidth_fit=pooled_fit,
            calibrated_min=calibrated_min,
            calibrated_max=calibrated_max,
        )
        message_validation, message_validation_summary = validate_message_projection(
            measured_message_columns,
            message_columns,
            args.training_gpu_counts,
        )
        query_rows, aggregate_rows = assemble_predictions(
            compute_samples,
            compute_fits,
            message_columns,
            args.prediction_gpu_counts,
            args.scale,
        )

        output = args.output_dir
        write_csv(output / "distributed_tpch_strong_scaling_compute_samples.csv", compute_samples)
        write_csv(output / "distributed_tpch_strong_scaling_compute_fits.csv", compute_fits)
        write_csv(output / "distributed_tpch_strong_scaling_busbw_samples.csv", bandwidth_samples)
        write_csv(output / "distributed_tpch_strong_scaling_busbw_fits.csv", bandwidth_fits)
        write_csv(output / "distributed_tpch_strong_scaling_message_columns.csv", message_columns)
        write_csv(
            output / "distributed_tpch_strong_scaling_measured_message_columns.csv",
            measured_message_columns,
        )
        write_csv(output / "distributed_tpch_strong_scaling_message_coverage.csv", message_coverage)
        write_csv(
            output / "distributed_tpch_strong_scaling_message_projection_validation.csv",
            message_validation,
        )
        write_csv(
            output / "distributed_tpch_strong_scaling_message_projection_validation_summary.csv",
            message_validation_summary,
        )
        write_csv(output / "distributed_tpch_strong_scaling_query_predictions.csv", query_rows)
        write_csv(output / "distributed_tpch_strong_scaling_aggregate_predictions.csv", aggregate_rows)
        write_metadata(
            output / "distributed_tpch_strong_scaling_model.json",
            args,
            pooled_fit,
            calibrated_min,
            calibrated_max,
        )
        write_report(
            output / "distributed_tpch_strong_scaling_model.md",
            args,
            pooled_fit,
            compute_fits,
            message_columns,
            message_validation_summary,
            aggregate_rows,
            calibrated_min,
            calibrated_max,
        )

        plot_bandwidth_fit(bandwidth_samples, bandwidth_fits, output)
        plot_compute_fits(compute_samples, compute_fits, output)
        plot_aggregate_model(aggregate_rows, output)
        plot_model_validation(query_rows, output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
