#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot measured and perfectly linear compute-time scaling for every query.

Two 4x6 figures are produced.  SF=1000 uses its 8-GPU compute attribution as
the reference, while SF=3000 uses its 24-GPU compute attribution.  The ideal
time at N GPUs is T_ref * N_ref / N.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import plot_distributed_tpch_gpu_scaling_aggregate_breakdown as scaling
from plot_distributed_tpch_aggregate_breakdown import (
    QUERIES,
    REPO_ROOT,
    QueryBreakdown,
    plt,
)


ARM = "legacy_cta64"
SCALE_CONFIGURATIONS = {
    1000: (8, 16, 24, 32),
    3000: (24, 32),
}
REFERENCE_GPUS = {1000: 8, 3000: 24}
FIGURE_STEM = "legacy_cta64_per_query_compute_scaling"
MEASURED_COLOR = "#4C78A8"
IDEAL_COLOR = "#5F6368"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "distributed_tpch",
        help="Distributed TPC-H results root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "distributed_tpch"
        / "figures"
        / "presentation",
        help="Output root; figures are written to pdf/ and png/",
    )
    return parser.parse_args()


def validate(rows: list[QueryBreakdown]) -> dict[tuple[int, int, int], QueryBreakdown]:
    lookup = {
        (row.scale_factor, row.gpus, row.query): row for row in rows
    }
    expected = {
        (scale_factor, gpus, query)
        for scale_factor, gpu_counts in SCALE_CONFIGURATIONS.items()
        for gpus in gpu_counts
        for query in QUERIES
    }
    actual = set(lookup)
    if actual != expected:
        raise ValueError(
            "unexpected query/configuration coverage: "
            f"{sorted(actual ^ expected)}"
        )
    return lookup


def write_points_csv(
    lookup: dict[tuple[int, int, int], QueryBreakdown], output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{FIGURE_STEM}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scale_factor",
                "query",
                "gpus",
                "reference_gpus",
                "reference_compute_ms",
                "measured_compute_ms",
                "ideal_linear_compute_ms",
            ),
        )
        writer.writeheader()
        for scale_factor, gpu_counts in SCALE_CONFIGURATIONS.items():
            reference_gpus = REFERENCE_GPUS[scale_factor]
            for query in QUERIES:
                reference = lookup[(scale_factor, reference_gpus, query)]
                for gpus in gpu_counts:
                    measured = lookup[(scale_factor, gpus, query)]
                    writer.writerow(
                        {
                            "scale_factor": scale_factor,
                            "query": query,
                            "gpus": gpus,
                            "reference_gpus": reference_gpus,
                            "reference_compute_ms": (
                                f"{reference.computation_ms:.9f}"
                            ),
                            "measured_compute_ms": (
                                f"{measured.computation_ms:.9f}"
                            ),
                            "ideal_linear_compute_ms": (
                                f"{reference.computation_ms * reference_gpus / gpus:.9f}"
                            ),
                        }
                    )
    print(f"wrote {path}")
    return path


def plot_scale(
    lookup: dict[tuple[int, int, int], QueryBreakdown],
    scale_factor: int,
    output_dir: Path,
) -> None:
    gpu_counts = SCALE_CONFIGURATIONS[scale_factor]
    reference_gpus = REFERENCE_GPUS[scale_factor]
    fig, axes = plt.subplots(4, 6, figsize=(20.0, 13.0))
    fig.patch.set_facecolor("white")

    for ax, query in zip(axes.flat, QUERIES, strict=False):
        reference = lookup[(scale_factor, reference_gpus, query)]
        measured = [
            lookup[(scale_factor, gpus, query)].computation_ms
            for gpus in gpu_counts
        ]
        ideal = [
            reference.computation_ms * reference_gpus / gpus
            for gpus in gpu_counts
        ]
        maximum = max(measured + ideal)

        ax.set_facecolor("white")
        ax.plot(
            gpu_counts,
            ideal,
            color=IDEAL_COLOR,
            linewidth=1.8,
            linestyle=(0, (4, 2.5)),
            marker="D",
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=1.1,
            zorder=3,
        )
        ax.plot(
            gpu_counts,
            measured,
            color=MEASURED_COLOR,
            linewidth=2.0,
            marker="o",
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4,
        )

        ax.set_title(f"Q{query}", fontsize=11.5, fontweight="bold", pad=5)
        ax.set_xticks(gpu_counts)
        ax.tick_params(axis="x", labelsize=8.0, pad=2)
        ax.tick_params(axis="y", labelsize=7.5)
        margin = max(1.5, (max(gpu_counts) - min(gpu_counts)) * 0.10)
        ax.set_xlim(min(gpu_counts) - margin, max(gpu_counts) + margin)
        ax.set_ylim(0.0, maximum * 1.13 if maximum > 0.0 else 1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.grid(
            axis="y",
            color="#D8D8D8",
            linewidth=0.65,
            linestyle=(0, (2, 3)),
            zorder=0,
        )
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")

    for ax in axes.flat[len(QUERIES) :]:
        ax.set_visible(False)

    handles = (
        Line2D(
            [0],
            [0],
            color=MEASURED_COLOR,
            linewidth=2.4,
            marker="o",
            markersize=6,
            label="Measured compute",
        ),
        Line2D(
            [0],
            [0],
            color=IDEAL_COLOR,
            linewidth=2.1,
            linestyle=(0, (4, 2.5)),
            marker="D",
            markersize=5,
            markerfacecolor="white",
            label="Perfect linear scaling",
        ),
    )
    fig.suptitle(
        f"TPC-H Per-Query Compute-Time Scaling — SF={scale_factor}",
        fontsize=20,
        y=0.985,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.953),
        ncol=2,
        frameon=False,
        fontsize=12,
        handlelength=2.5,
        columnspacing=2.0,
    )
    fig.supylabel("Compute time (ms)", fontsize=14, x=0.012)
    fig.supxlabel("Number of GPUs", fontsize=13, y=0.018)
    fig.subplots_adjust(
        left=0.045,
        right=0.995,
        bottom=0.065,
        top=0.905,
        wspace=0.28,
        hspace=0.42,
    )

    stem = f"{FIGURE_STEM}_sf{scale_factor}"
    for suffix in ("pdf", "png"):
        directory = output_dir / suffix
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    query_rows, _ = scaling.collect(results_dir)
    lookup = validate(query_rows)
    write_points_csv(lookup, output_dir)
    for scale_factor in SCALE_CONFIGURATIONS:
        plot_scale(lookup, scale_factor, output_dir)


if __name__ == "__main__":
    main()
