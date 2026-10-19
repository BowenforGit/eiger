#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot the distribution of NCCL API payload sizes from a message campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.size": 13.0,
            "axes.titlesize": 17.0,
            "axes.labelsize": 15.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 13.0,
            "ytick.labelsize": 13.0,
            "legend.fontsize": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator
except ModuleNotFoundError as error:
    sys.exit(f"error: matplotlib and numpy are required: {error}")


FIGURE_SIZE = (7.4, 4.8)
BAR_COLOR = "#7A8793"
MEDIAN_COLOR = "#0072B2"
MEAN_COLOR = "#D55E00"
RANGE_COLOR = "#555555"
API_LABELS = {
    "ncclSend": "NCCL send",
    "ncclRecv": "NCCL receive",
    "ncclBroadcast": "NCCL broadcast",
    "ncclAllReduce": "NCCL all-reduce",
}
API_STEMS = {
    "ncclSend": "nccl_send",
    "ncclRecv": "nccl_recv",
    "ncclBroadcast": "nccl_broadcast",
    "ncclAllReduce": "nccl_allreduce",
}


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=(
            repository
            / "results"
            / "distributed_tpch"
            / "sf1000"
            / "8gpu_128p_message_size"
        ),
        help="completed distributed TPC-H message campaign",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            repository
            / "results"
            / "distributed_tpch"
            / "figures"
            / "presentation"
        ),
        help="parent output directory; PNG and PDF use separate subdirectories",
    )
    parser.add_argument(
        "--api",
        default="ncclSend",
        choices=("ncclSend", "ncclRecv", "ncclBroadcast", "ncclAllReduce"),
        help="NCCL API whose call-level payload distribution is plotted",
    )
    return parser.parse_args()


def attempt_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("attempt-"))
    except ValueError as error:
        raise ValueError(f"invalid attempt directory: {path}") from error


def usable_messages_csv(query_root: Path) -> Path:
    for attempt in sorted(
        query_root.glob("attempt-*"), key=attempt_number, reverse=True
    ):
        result_path = attempt / "result.json"
        messages_path = attempt / "messages.csv"
        if not result_path.is_file() or not messages_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        accepted = result.get("status") == "pass" or (
            result.get("status") == "fail"
            and result.get("failure_kind") == "verification"
            and result.get("measurements_complete") is True
        )
        if accepted and messages_path.stat().st_size:
            return messages_path
    raise ValueError(f"no usable messages.csv attempt under {query_root}")


def load_payloads(campaign: Path, api: str) -> tuple[list[int], set[str], set[str]]:
    query_roots = sorted(
        (campaign / "arms" / "legacy" / "messages").glob("q*")
    )
    if not query_roots:
        raise ValueError(f"no Legacy message cells found under {campaign}")

    payloads: list[int] = []
    queries: set[str] = set()
    ranks: set[str] = set()
    required = {"query", "ranks", "api", "payload_bytes"}
    for query_root in query_roots:
        path = usable_messages_csv(query_root)
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path} is missing columns: {', '.join(sorted(missing))}"
                )
            for row in reader:
                if row["api"] != api:
                    continue
                payload = int(row["payload_bytes"])
                if payload <= 0:
                    raise ValueError(f"non-positive payload in {path}: {payload}")
                payloads.append(payload)
                queries.add(row["query"])
                ranks.add(row["ranks"])
    if not payloads:
        raise ValueError(f"no {api} rows found under {campaign}")
    return payloads, queries, ranks


def binary_size(value: float) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.1f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value:.0f} B"


def campaign_scale(campaign: Path) -> int:
    manifest_path = campaign / "manifest" / "campaign.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        scale = int(manifest["configuration"]["arguments"]["scale"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read scale factor from {manifest_path}") from error
    if scale <= 0:
        raise ValueError(f"invalid scale factor in {manifest_path}: {scale}")
    return scale


def plot(
    payloads: list[int],
    api: str,
    scale: int,
    gpus: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    values = np.asarray(payloads, dtype=np.float64)
    minimum = float(values.min())
    maximum = float(values.max())
    median = float(np.median(values))
    mean = float(np.mean(values))

    lower_power = math.floor(math.log2(minimum))
    upper_power = math.ceil(math.log2(maximum))
    bins = np.geomspace(2.0**lower_power, 2.0**upper_power, 49)
    weights = np.full(values.shape, 100.0 / values.size)

    figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    axis.hist(
        values,
        bins=bins,
        weights=weights,
        color=BAR_COLOR,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.95,
    )
    axis.axvline(
        minimum,
        color=RANGE_COLOR,
        linewidth=1.0,
        linestyle=":",
        label=f"Min: {binary_size(minimum)}",
    )
    axis.axvline(
        median,
        color=MEDIAN_COLOR,
        linewidth=1.25,
        linestyle="-",
        label=f"Median: {binary_size(median)}",
    )
    axis.axvline(
        mean,
        color=MEAN_COLOR,
        linewidth=1.25,
        linestyle="--",
        label=f"Mean: {binary_size(mean)}",
    )
    axis.axvline(
        maximum,
        color=RANGE_COLOR,
        linewidth=1.0,
        linestyle="-.",
        label=f"Max: {binary_size(maximum)}",
    )

    axis.set_xscale("log", base=2)
    tick_values = [1, 1024, 1024**2, 1024**3]
    axis.set_xticks(tick_values, ["1 B", "1 KiB", "1 MiB", "1 GiB"])
    axis.set_xlim(2.0**lower_power, 2.0**upper_power)
    axis.set_xlabel(f"{API_LABELS[api]} payload size")
    axis.set_ylabel("Share of API calls (%)")
    axis.set_title(
        f"TPC-H {API_LABELS[api]} Message Sizes — SF={scale}, {gpus} GPUs",
        pad=13,
    )
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D5D5D5", linewidth=0.55, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.03,
        0.96,
        f"n = {len(payloads):,}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12.0,
        color="#4D4D4D",
    )
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.84),
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#CCCCCC",
        borderpad=0.35,
        handlelength=2.0,
    )

    stem = f"tpch_sf{scale}_{gpus}gpu_{API_STEMS[api]}_message_size_distribution"
    png_path = output_dir / "png" / f"{stem}.png"
    pdf_path = output_dir / "pdf" / f"{stem}.pdf"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": (
            f"TPC-H SF{scale} {API_LABELS[api]} message-size distribution"
        ),
        "Subject": f"Call-level NCCL API payload sizes across {gpus} GPUs",
        "Creator": "plot_distributed_tpch_message_sizes.py",
    }
    figure.savefig(png_path, dpi=300, facecolor="white", metadata=metadata)
    figure.savefig(pdf_path, facecolor="white", metadata=metadata)
    plt.close(figure)
    return png_path, pdf_path


def main() -> int:
    args = parse_args()
    try:
        campaign = args.campaign.resolve()
        payloads, queries, ranks = load_payloads(campaign, args.api)
        if len(ranks) != 1:
            raise ValueError(
                f"expected one rank-count value in {campaign}, found {sorted(ranks)}"
            )
        gpus = int(next(iter(ranks)))
        scale = campaign_scale(campaign)
        png_path, pdf_path = plot(
            payloads,
            args.api,
            scale,
            gpus,
            args.out_dir.resolve(),
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    print(
        f"{args.api}: n={len(payloads):,}, queries={len(queries)}, "
        f"ranks={','.join(sorted(ranks))}, mean={statistics.fmean(payloads):,.3f} B, "
        f"median={statistics.median(payloads):,.3f} B, "
        f"min={min(payloads):,} B, max={max(payloads):,} B"
    )
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
