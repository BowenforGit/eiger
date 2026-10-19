#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot compute versus dominant-memory throughput by kernel family and arm.

Each point represents one selected rank-0 NCU target-kernel launch. The x
coordinate is the maximum of its DRAM, L2, and L1 percent-of-peak throughput
metrics; marker shape identifies which memory level supplied that maximum. The
y coordinate is SM percent-of-peak throughput, and darker points ran longer.

The fixed 3 x 4 family layout, axes, and duration color normalization are shared
across every campaign passed to one invocation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 19,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import plot_distributed_tpch_compute_memory_throughput as throughput_plot
import plot_distributed_tpch_instruction_roofline as common
import plot_distributed_tpch_instruction_roofline_by_family as family_layout


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SharedPlotParameters:
    axis_min_pct: float
    axis_max_pct: float
    major_tick_spacing_pct: float
    duration_min_ms: float
    duration_max_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        action="append",
        required=True,
        type=Path,
        help="Campaign root; repeat once per arm to share axes and color scale",
    )
    return parser.parse_args()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def family_for_slot(
    dataset: throughput_plot.CampaignDataset, slot: tuple[str, ...]
) -> str | None:
    available = {point.kernel_family for point in dataset.points}
    present = [family for family in slot if family in available]
    if len(present) > 1:
        raise ValueError(f"multiple families found for panel slot: {present}")
    return present[0] if present else None


def summarize_family(
    dataset: throughput_plot.CampaignDataset,
    family: str,
    panel_slot: int,
) -> dict[str, object]:
    points = [point for point in dataset.points if point.kernel_family == family]
    if not points:
        raise ValueError(f"{dataset.arm}: empty family {family}")
    level_counts = Counter(point.dominant_memory_level for point in points)
    row: dict[str, object] = {
        "arm": dataset.arm,
        "arm_label": dataset.display_label,
        "panel_slot": panel_slot,
        "kernel_family": family,
        "display_label": family_layout.FAMILY_LABELS[family],
        "launch_count": len(points),
    }
    for level, label, _metric, _marker in throughput_plot.MEMORY_LEVELS:
        count = level_counts[level]
        row[f"dominant_{level}_count"] = count
        row[f"dominant_{level}_percent"] = 100.0 * count / len(points)
    row.update(
        compute_throughput_min_pct=min(
            point.compute_throughput_pct for point in points
        ),
        compute_throughput_median_pct=statistics.median(
            point.compute_throughput_pct for point in points
        ),
        compute_throughput_max_pct=max(
            point.compute_throughput_pct for point in points
        ),
        dominant_memory_throughput_min_pct=min(
            point.memory_throughput_pct for point in points
        ),
        dominant_memory_throughput_median_pct=statistics.median(
            point.memory_throughput_pct for point in points
        ),
        dominant_memory_throughput_max_pct=max(
            point.memory_throughput_pct for point in points
        ),
        duration_min_ms=min(point.duration_ms for point in points),
        duration_median_ms=statistics.median(point.duration_ms for point in points),
        duration_max_ms=max(point.duration_ms for point in points),
    )
    return row


def panel_summaries(
    dataset: throughput_plot.CampaignDataset,
) -> list[dict[str, object]]:
    rows = []
    for slot_index, slot in enumerate(family_layout.PANEL_SLOTS, start=1):
        family = family_for_slot(dataset, slot)
        if family is not None:
            rows.append(summarize_family(dataset, family, slot_index))
    if sum(int(row["launch_count"]) for row in rows) != len(dataset.points):
        raise ValueError(f"{dataset.arm}: panel counts do not cover every point")
    return rows


def fact_sheet_validation(
    dataset: throughput_plot.CampaignDataset,
) -> dict[str, object]:
    _, summary_path = family_layout.fact_sheet_paths(dataset.campaign, dataset.arm)
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing kernel fact-sheet summary: {summary_path}")
    validation = json.loads(summary_path.read_text())["validation"]
    expected_target_count = int(validation["target_kernel_count"])
    expected_excluded_count = int(validation["excluded_capture_count"])
    expected_raw_count = int(validation["row_count"])
    if len(dataset.points) != expected_target_count:
        raise ValueError(
            f"{dataset.arm}: {len(dataset.points)} target launches, "
            f"expected {expected_target_count}"
        )
    if len(dataset.excluded) != expected_excluded_count:
        raise ValueError(
            f"{dataset.arm}: {len(dataset.excluded)} excluded launches, "
            f"expected {expected_excluded_count}"
        )
    if len(dataset.points) + len(dataset.excluded) != expected_raw_count:
        raise ValueError(f"{dataset.arm}: raw launch count disagrees with fact sheet")
    return {
        "fact_sheet_row_count": expected_raw_count,
        "target_kernel_count": expected_target_count,
        "excluded_capture_count": expected_excluded_count,
    }


def validate_dataset(dataset: throughput_plot.CampaignDataset) -> dict[str, object]:
    throughput_validation = throughput_plot.validate_dataset(dataset)
    inventory_validation = fact_sheet_validation(dataset)

    allowed_families = {family for slot in family_layout.PANEL_SLOTS for family in slot}
    unexpected = sorted(
        {point.kernel_family for point in dataset.points} - allowed_families
    )
    if unexpected:
        raise ValueError(f"{dataset.arm}: unexpected kernel families: {unexpected}")

    family_counts = Counter(point.kernel_family for point in dataset.points)
    for slot in family_layout.PANEL_SLOTS:
        present = [family for family in slot if family_counts[family]]
        if len(present) > 1:
            raise ValueError(
                f"{dataset.arm}: alternate families collide in one slot: {present}"
            )
    panels = panel_summaries(dataset)
    return {
        **throughput_validation,
        **inventory_validation,
        "panel_count": len(panels),
        "panel_launch_count": sum(int(row["launch_count"]) for row in panels),
    }


def shared_plot_parameters(
    datasets: list[throughput_plot.CampaignDataset],
) -> SharedPlotParameters:
    if not datasets:
        raise ValueError("at least one campaign is required")
    durations = [point.duration_ms for dataset in datasets for point in dataset.points]
    axis_max = throughput_plot.common_axis_limit(datasets)
    major_spacing = 20.0 if axis_max <= 120.0 else 25.0
    duration_max = max(durations)
    duration_min = min(durations)
    if math.isclose(duration_min, duration_max):
        duration_max = duration_min * 1.01
    return SharedPlotParameters(
        axis_min_pct=0.0,
        axis_max_pct=axis_max,
        major_tick_spacing_pct=major_spacing,
        duration_min_ms=duration_min,
        duration_max_ms=duration_max,
    )


def configure_axis(ax: plt.Axes, shared: SharedPlotParameters) -> None:
    ax.set_xlim(shared.axis_min_pct, shared.axis_max_pct)
    ax.set_ylim(shared.axis_min_pct, shared.axis_max_pct)
    major_spacing = shared.major_tick_spacing_pct
    ax.xaxis.set_major_locator(MultipleLocator(major_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(major_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(major_spacing / 2.0))
    ax.yaxis.set_minor_locator(MultipleLocator(major_spacing / 2.0))
    ax.set_facecolor("#FCFCFC")
    ax.grid(
        axis="both",
        which="major",
        color="#D0D0D0",
        linewidth=0.9,
        linestyle=(0, (2, 3)),
        zorder=0,
    )
    ax.grid(
        axis="both",
        which="minor",
        color="#E8E8E8",
        linewidth=0.55,
        linestyle=(0, (1, 4)),
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#303030")
    for spine in ax.spines.values():
        spine.set_color("#777777")


def render_arm(
    dataset: throughput_plot.CampaignDataset,
    shared: SharedPlotParameters,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    norm = LogNorm(
        vmin=shared.duration_min_ms,
        vmax=shared.duration_max_ms,
    )
    fig, axes = plt.subplots(
        family_layout.PANEL_ROWS,
        family_layout.PANEL_COLUMNS,
        figsize=(23.5, 16.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    fig.patch.set_facecolor("white")
    visible_axes = []
    points_by_family = {
        family: [point for point in dataset.points if point.kernel_family == family]
        for family in {point.kernel_family for point in dataset.points}
    }

    for ax, slot in zip(axes.flat, family_layout.PANEL_SLOTS, strict=True):
        family = family_for_slot(dataset, slot)
        if family is None:
            ax.set_visible(False)
            continue
        visible_axes.append(ax)
        configure_axis(ax, shared)
        points = points_by_family[family]
        for level, _label, _metric, marker in throughput_plot.MEMORY_LEVELS:
            level_points = sorted(
                (point for point in points if point.dominant_memory_level == level),
                key=lambda point: point.duration_ms,
            )
            if not level_points:
                continue
            ax.scatter(
                [point.memory_throughput_pct for point in level_points],
                [point.compute_throughput_pct for point in level_points],
                c=[point.duration_ms for point in level_points],
                cmap=common.DURATION_CMAP,
                norm=norm,
                marker=marker,
                s=86,
                edgecolors="#FFFFFF",
                linewidths=0.6,
                alpha=0.88,
                zorder=4,
            )
        ax.set_title(
            f"{family_layout.FAMILY_LABELS[family]} (n={len(points)})",
            pad=10,
            fontsize=15,
            fontweight="bold",
        )

    fig.suptitle(
        "TPC-H Kernel Throughput by Kernel Family — "
        f"SF1000, 8 GPUs, {dataset.display_label}",
        fontsize=24,
        fontweight="bold",
    )
    fig.supxlabel("Dominant memory throughput (% of peak)", fontsize=19)
    fig.supylabel("SM compute throughput (% of peak)", fontsize=19)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="#466B8C",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8.5,
            label=label,
        )
        for _level, label, _metric, marker in throughput_plot.MEMORY_LEVELS
    ]
    visible_axes[0].legend(
        handles=legend_handles,
        title="Most utilized memory level",
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#AAAAAA",
        framealpha=0.92,
        fontsize=11.5,
        title_fontsize=11.5,
    )

    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=common.DURATION_CMAP),
        ax=visible_axes,
        location="right",
        pad=0.012,
        fraction=0.02,
        aspect=42,
    )
    colorbar.set_label(
        "Kernel duration (ms, log scale; darker = longer)",
        fontsize=17,
        labelpad=13,
    )
    colorbar.ax.tick_params(labelsize=13)
    fig.get_layout_engine().set(
        w_pad=0.075,
        h_pad=0.09,
        wspace=0.045,
        hspace=0.11,
    )

    outputs = []
    stem = f"compute_memory_throughput_by_family_{dataset.arm}"
    for suffix in ("png", "pdf"):
        output = output_dir / f"{stem}.{suffix}"
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        outputs.append(output)
        print(f"wrote {output}")
    plt.close(fig)
    return outputs


def write_outputs(
    dataset: throughput_plot.CampaignDataset,
    validation: dict[str, object],
    shared: SharedPlotParameters,
    all_datasets: list[throughput_plot.CampaignDataset],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"compute_memory_throughput_by_family_{dataset.arm}"
    points_path = output_dir / f"{stem}_points.csv"
    panels_path = output_dir / f"{stem}_panels.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    common.write_csv(points_path, throughput_plot.point_rows(dataset.points))
    panels = panel_summaries(dataset)
    common.write_csv(panels_path, panels)

    summary = {
        "campaign": str(dataset.campaign),
        "arm": dataset.arm,
        "arm_label": dataset.display_label,
        "plot": asdict(shared),
        "shared_with_arms": [item.arm for item in all_datasets],
        "inventory": {
            "raw_profiled_launches": len(dataset.points) + len(dataset.excluded),
            "plotted_target_launches": len(dataset.points),
            "excluded_non_target_captures": len(dataset.excluded),
            "panel_count": len(panels),
        },
        "dominant_memory_level_counts": dict(
            sorted(
                Counter(point.dominant_memory_level for point in dataset.points).items()
            )
        ),
        "panel_slots": [list(slot) for slot in family_layout.PANEL_SLOTS],
        "panels": panels,
        "metrics": {
            "compute_throughput": throughput_plot.COMPUTE_METRIC,
            "dram_throughput": throughput_plot.LEVEL_METRICS["dram"],
            "l2_throughput": throughput_plot.LEVEL_METRICS["l2"],
            "l1_throughput": throughput_plot.LEVEL_METRICS["l1"],
            "duration": throughput_plot.DURATION_METRIC,
        },
        "encoding": {
            "x": "max(dram_throughput, l2_throughput, l1_throughput)",
            "y": throughput_plot.COMPUTE_METRIC,
            "marker": "argmax(dram, l2, l1); ties use DRAM, L2, L1 priority",
            "marker_shapes": throughput_plot.LEVEL_MARKERS,
            "color": (
                f"{throughput_plot.DURATION_METRIC}; logarithmic, darker is longer"
            ),
        },
        "validation": validation,
        "capture_caveat": (
            "Rank-0 NCU replay with clock control disabled; throughput "
            "percentages and duration are time-derived isolated measurements, "
            "not contention-realistic all-GPU-active measurements."
        ),
        "denominator_caveat": (
            "DRAM, L2, and SM use pct_of_peak_sustained_elapsed; L1 uses "
            "pct_of_peak_sustained_active, exactly as requested."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")

    command_separator = " " + chr(92) + "\n  "
    command = command_separator.join(
        f"--campaign {relative_or_absolute(item.campaign)}" for item in all_datasets
    )
    readme = f"""# Compute versus dominant-memory throughput by family: {dataset.display_label}

This directory contains the fixed 3 x 4 family layout used by the instruction
roofline figures. Corresponding panels, axes, and duration colors are directly
comparable across the listed arms.

## Reproduce

```bash
python3 tools/visualize/plot_distributed_tpch_compute_memory_throughput_by_family.py \\
  {command}
```

## Encodings

- x: maximum of the DRAM, L2, and L1 percent-of-peak throughput counters.
- y: `{throughput_plot.COMPUTE_METRIC}`.
- marker: the memory level providing the x-axis maximum (DRAM circle, L2
  square, L1 triangle); exact ties use DRAM, then L2, then L1.
- color: `{throughput_plot.DURATION_METRIC}` on one logarithmic monochromatic
  scale shared across the listed arms; darker kernels ran longer.
- axes: {shared.axis_min_pct:g}--{shared.axis_max_pct:g}% on both dimensions.

Only validated target launches are plotted. In particular, the `count` family
accepts only symbols containing `{common.DESIRED_COUNT_SYMBOL}`.

DRAM, L2, and SM use `pct_of_peak_sustained_elapsed`; L1 uses the requested
`pct_of_peak_sustained_active` denominator.

## Capture caveat

These reports profile rank 0 with NCU kernel replay and `--clock-control none`.
Throughput percentages and durations are isolated, time-derived measurements
and are not contention-realistic all-GPU-active results.
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme)
    print(f"wrote {readme_path}")


def main() -> None:
    args = parse_args()
    campaigns = [campaign.resolve() for campaign in args.campaign]
    if len(set(campaigns)) != len(campaigns):
        raise ValueError("the same campaign was passed more than once")
    datasets = [throughput_plot.load_campaign(campaign) for campaign in campaigns]
    if len({dataset.arm for dataset in datasets}) != len(datasets):
        raise ValueError("each campaign must represent a different arm")
    validations = {dataset.arm: validate_dataset(dataset) for dataset in datasets}
    shared = shared_plot_parameters(datasets)

    for dataset in datasets:
        output_dir = (
            dataset.campaign / "analysis" / "compute_memory_throughput_by_family"
        )
        render_arm(dataset, shared, output_dir)
        write_outputs(
            dataset,
            validations[dataset.arm],
            shared,
            datasets,
            output_dir,
        )
        print(
            f"{dataset.arm}: plotted {len(dataset.points)} target launches in "
            f"{len(panel_summaries(dataset))} family panels"
        )


if __name__ == "__main__":
    main()
