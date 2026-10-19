#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot L1 and L2 sector hit rates against duration in separate figures.

Each validated kernel occurrence contributes one marker to the L1 figure and
one marker to the L2 figure. The fixed 3 x 4 family layout and axis ranges are
shared across both cache levels and every campaign passed to one invocation.
"""

from __future__ import annotations

import argparse
import csv
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
from matplotlib.ticker import FuncFormatter, MultipleLocator

import plot_distributed_tpch_instruction_roofline as common
import plot_distributed_tpch_instruction_roofline_by_family as family_layout


REPO_ROOT = Path(__file__).resolve().parents[2]
L1_HIT_RATE_METRIC = "l1tex__t_sector_hit_rate.pct"
L2_HIT_RATE_METRIC = "lts__t_sector_hit_rate.pct"
DURATION_METRIC = "gpu__time_duration.sum"
SERIES = (
    ("l1", "L1 hit rate", "o", "#38678B"),
    ("l2", "L2 hit rate", "s", "#D27A32"),
)


@dataclass(frozen=True)
class HitRatePoint:
    occurrence_id: str
    query: int
    launch_id: int
    family: str
    kernel_name: str
    l1_sector_hit_rate_pct: float
    l2_sector_hit_rate_pct: float
    duration_ms: float
    source_csv: str


@dataclass(frozen=True)
class ArmDataset:
    campaign: Path
    arm: str
    arm_label: str
    points: tuple[HitRatePoint, ...]
    fact_sheet_row_count: int
    excluded_capture_count: int
    capture_caveat: str


@dataclass(frozen=True)
class SharedPlotParameters:
    x_min_pct: float
    x_max_pct: float
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
        help="Campaign root; repeat once per arm to share axes",
    )
    return parser.parse_args()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_dataset(campaign: Path) -> ArmDataset:
    campaign = campaign.resolve()
    if not campaign.is_dir():
        raise FileNotFoundError(f"campaign directory does not exist: {campaign}")
    arm = common.infer_arm(campaign)
    csv_path, summary_path = family_layout.fact_sheet_paths(campaign, arm)
    if not csv_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "missing kernel fact sheet; run the fact-sheet exporter first: "
            f"{csv_path}"
        )

    summary = json.loads(summary_path.read_text())
    validation = summary["validation"]
    points = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not family_layout.parse_bool(row["target_kernel_match"]):
                continue
            points.append(
                HitRatePoint(
                    occurrence_id=row["occurrence_id"],
                    query=int(row["query"]),
                    launch_id=int(row["ncu_launch_id"]),
                    family=row["kernel_family"],
                    kernel_name=row["kernel_name"],
                    l1_sector_hit_rate_pct=float(row["l1_sector_hit_rate_pct"]),
                    l2_sector_hit_rate_pct=float(row["l2_sector_hit_rate_pct"]),
                    duration_ms=float(row["kernel_duration_ms"]),
                    source_csv=row["source_csv"],
                )
            )

    dataset = ArmDataset(
        campaign=campaign,
        arm=arm,
        arm_label=str(summary["arm_label"]),
        points=tuple(points),
        fact_sheet_row_count=int(validation["row_count"]),
        excluded_capture_count=int(validation["excluded_capture_count"]),
        capture_caveat=str(summary["capture_caveat"]),
    )
    validate_dataset(dataset, int(validation["target_kernel_count"]))
    return dataset


def family_for_slot(dataset: ArmDataset, slot: tuple[str, ...]) -> str | None:
    available = {point.family for point in dataset.points}
    present = [family for family in slot if family in available]
    if len(present) > 1:
        raise ValueError(f"multiple families found for panel slot: {present}")
    return present[0] if present else None


def validate_dataset(dataset: ArmDataset, expected_target_count: int) -> None:
    if len(dataset.points) != expected_target_count:
        raise ValueError(
            f"{dataset.arm}: loaded {len(dataset.points)} target launches, "
            f"expected {expected_target_count}"
        )
    if not dataset.points:
        raise ValueError(f"{dataset.arm}: no target kernel launches")
    for point in dataset.points:
        if not math.isfinite(point.duration_ms) or point.duration_ms <= 0:
            raise ValueError(f"{dataset.arm}: invalid kernel duration")
        for name, hit_rate in (
            ("L1", point.l1_sector_hit_rate_pct),
            ("L2", point.l2_sector_hit_rate_pct),
        ):
            if not math.isfinite(hit_rate) or not 0.0 <= hit_rate <= 100.0:
                raise ValueError(f"{dataset.arm}: invalid {name} hit rate {hit_rate}")

    allowed_families = {family for slot in family_layout.PANEL_SLOTS for family in slot}
    unexpected = sorted({point.family for point in dataset.points} - allowed_families)
    if unexpected:
        raise ValueError(f"{dataset.arm}: unexpected kernel families: {unexpected}")
    family_counts = Counter(point.family for point in dataset.points)
    for slot in family_layout.PANEL_SLOTS:
        present = [family for family in slot if family_counts[family]]
        if len(present) > 1:
            raise ValueError(
                f"{dataset.arm}: alternate families collide in one slot: {present}"
            )

    count_points = [point for point in dataset.points if point.family == "count"]
    if not count_points:
        raise ValueError(f"{dataset.arm}: no target count kernel")
    if any(
        common.DESIRED_COUNT_SYMBOL not in point.kernel_name for point in count_points
    ):
        raise ValueError(f"{dataset.arm}: non-target count kernel reached plot data")


def shared_plot_parameters(datasets: list[ArmDataset]) -> SharedPlotParameters:
    if not datasets:
        raise ValueError("at least one arm dataset is required")
    durations = [point.duration_ms for dataset in datasets for point in dataset.points]
    duration_min = common.power_of_ten_floor(min(durations) * 0.8)
    duration_max = common.power_of_ten_ceil(max(durations) * 1.1)
    return SharedPlotParameters(
        x_min_pct=-2.0,
        x_max_pct=102.0,
        major_tick_spacing_pct=20.0,
        duration_min_ms=duration_min,
        duration_max_ms=duration_max,
    )


def summarize_family(
    dataset: ArmDataset,
    family: str,
    panel_slot: int,
) -> dict[str, object]:
    points = [point for point in dataset.points if point.family == family]
    if not points:
        raise ValueError(f"{dataset.arm}: empty family {family}")
    return {
        "arm": dataset.arm,
        "arm_label": dataset.arm_label,
        "panel_slot": panel_slot,
        "kernel_family": family,
        "display_label": family_layout.FAMILY_LABELS[family],
        "launch_count": len(points),
        "rendered_marker_count_per_figure": len(points),
        "rendered_marker_count_across_both_figures": 2 * len(points),
        "l1_sector_hit_rate_min_pct": min(
            point.l1_sector_hit_rate_pct for point in points
        ),
        "l1_sector_hit_rate_median_pct": statistics.median(
            point.l1_sector_hit_rate_pct for point in points
        ),
        "l1_sector_hit_rate_max_pct": max(
            point.l1_sector_hit_rate_pct for point in points
        ),
        "l2_sector_hit_rate_min_pct": min(
            point.l2_sector_hit_rate_pct for point in points
        ),
        "l2_sector_hit_rate_median_pct": statistics.median(
            point.l2_sector_hit_rate_pct for point in points
        ),
        "l2_sector_hit_rate_max_pct": max(
            point.l2_sector_hit_rate_pct for point in points
        ),
        "duration_min_ms": min(point.duration_ms for point in points),
        "duration_median_ms": statistics.median(point.duration_ms for point in points),
        "duration_max_ms": max(point.duration_ms for point in points),
    }


def panel_summaries(dataset: ArmDataset) -> list[dict[str, object]]:
    rows = []
    for slot_index, slot in enumerate(family_layout.PANEL_SLOTS, start=1):
        family = family_for_slot(dataset, slot)
        if family is not None:
            rows.append(summarize_family(dataset, family, slot_index))
    if sum(int(row["launch_count"]) for row in rows) != len(dataset.points):
        raise ValueError(f"{dataset.arm}: panel counts do not cover every point")
    return rows


def configure_axis(ax: plt.Axes, shared: SharedPlotParameters) -> None:
    ax.set_yscale("log")
    ax.set_xlim(shared.x_min_pct, shared.x_max_pct)
    ax.set_ylim(shared.duration_min_ms, shared.duration_max_ms)
    spacing = shared.major_tick_spacing_pct
    ax.xaxis.set_major_locator(MultipleLocator(spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(spacing / 2.0))
    ax.yaxis.set_major_formatter(FuncFormatter(common.format_log_tick))
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
    dataset: ArmDataset,
    shared: SharedPlotParameters,
    output_dir: Path,
    level: str,
) -> list[Path]:
    series_by_level = {
        key: (label, marker, color) for key, label, marker, color in SERIES
    }
    if level not in series_by_level:
        raise ValueError(f"unknown cache level: {level}")
    level_label, marker, color = series_by_level[level]
    output_dir.mkdir(parents=True, exist_ok=True)
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
        family: sorted(
            (point for point in dataset.points if point.family == family),
            key=lambda point: point.duration_ms,
        )
        for family in {point.family for point in dataset.points}
    }

    for ax, slot in zip(axes.flat, family_layout.PANEL_SLOTS, strict=True):
        family = family_for_slot(dataset, slot)
        if family is None:
            ax.set_visible(False)
            continue
        visible_axes.append(ax)
        configure_axis(ax, shared)
        points = points_by_family[family]
        values = [
            (
                point.l1_sector_hit_rate_pct
                if level == "l1"
                else point.l2_sector_hit_rate_pct
            )
            for point in points
        ]
        ax.scatter(
            values,
            [point.duration_ms for point in points],
            marker=marker,
            s=86,
            facecolors=color,
            edgecolors="#FFFFFF",
            linewidths=0.65,
            alpha=0.76,
            zorder=4,
        )
        ax.set_title(
            f"{family_layout.FAMILY_LABELS[family]} (n={len(points)})",
            pad=10,
            fontsize=15,
            fontweight="bold",
        )

    fig.suptitle(
        f"TPC-H {level_label} vs. Kernel Duration by Family — "
        f"SF1000, 8 GPUs, {dataset.arm_label}",
        fontsize=24,
        fontweight="bold",
    )
    fig.supxlabel(f"{level_label} (%)", fontsize=19)
    fig.supylabel("Kernel duration (ms, log scale)", fontsize=19)
    fig.get_layout_engine().set(
        w_pad=0.075,
        h_pad=0.09,
        wspace=0.045,
        hspace=0.11,
    )

    outputs = []
    stem = f"{level}_hit_rate_vs_duration_by_family_{dataset.arm}"
    for suffix in ("png", "pdf"):
        output = output_dir / f"{stem}.{suffix}"
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        outputs.append(output)
        print(f"wrote {output}")
    plt.close(fig)
    return outputs


def point_rows(dataset: ArmDataset) -> list[dict[str, object]]:
    return [
        asdict(point)
        for point in sorted(
            dataset.points, key=lambda item: (item.query, item.launch_id)
        )
    ]


def write_outputs(
    dataset: ArmDataset,
    shared: SharedPlotParameters,
    all_datasets: list[ArmDataset],
    output_dir: Path,
) -> None:
    stem = f"l1_l2_hit_rate_vs_duration_by_family_{dataset.arm}"
    points_path = output_dir / f"{stem}_points.csv"
    panels_path = output_dir / f"{stem}_panels.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    common.write_csv(points_path, point_rows(dataset))
    panels = panel_summaries(dataset)
    common.write_csv(panels_path, panels)

    summary = {
        "campaign": str(dataset.campaign),
        "arm": dataset.arm,
        "arm_label": dataset.arm_label,
        "plot": asdict(shared),
        "shared_with_arms": [item.arm for item in all_datasets],
        "inventory": {
            "fact_sheet_rows": dataset.fact_sheet_row_count,
            "plotted_target_launches": len(dataset.points),
            "figure_count": 2,
            "rendered_markers_per_figure": len(dataset.points),
            "rendered_markers_across_both_figures": 2 * len(dataset.points),
            "excluded_non_target_captures": dataset.excluded_capture_count,
            "panel_count": len(panels),
        },
        "metrics": {
            "l1_sector_hit_rate": L1_HIT_RATE_METRIC,
            "l2_sector_hit_rate": L2_HIT_RATE_METRIC,
            "duration": DURATION_METRIC,
        },
        "encoding": {
            "y": "kernel duration (ms, logarithmic)",
            "figures": {
                level: {
                    "x": f"{label} (%)",
                    "marker": marker,
                    "color": color,
                    "markers_per_kernel_occurrence": 1,
                }
                for level, label, marker, color in SERIES
            },
        },
        "panel_slots": [list(slot) for slot in family_layout.PANEL_SLOTS],
        "panels": panels,
        "capture_caveat": dataset.capture_caveat,
        "hit_rate_caveat": (
            "Hit rates come from NCU kernel replay. Replayed mutating kernels "
            "can make cache-traffic ratios indicative rather than exact."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")

    command_separator = " " + chr(92) + "\n  "
    command = command_separator.join(
        f"--campaign {relative_or_absolute(item.campaign)}" for item in all_datasets
    )
    readme = f"""# L1/L2 hit rate versus duration by family: {dataset.arm_label}

This directory contains two separate figures using the fixed 3 x 4 family
layout: L1 hit rate versus duration and L2 hit rate versus duration. Every
kernel occurrence contributes one marker to each figure.

## Reproduce

```bash
python3 tools/visualize/plot_distributed_tpch_cache_hit_rate_duration_by_family.py \\
  {command}
```

## Metrics

- L1 hit rate: `{L1_HIT_RATE_METRIC}`.
- L2 hit rate: `{L2_HIT_RATE_METRIC}`.
- Duration: `{DURATION_METRIC}`, converted from ns to ms by the fact-sheet
  exporter and plotted on a logarithmic y-axis.

The hit-rate and duration axes are identical across both figures, all panels,
and all listed arms. Only validated target launches are plotted; the `count`
family accepts only symbols containing `{common.DESIRED_COUNT_SYMBOL}`.

## Capture caveat

{dataset.capture_caveat} Durations are isolated replay measurements rather than
contention-realistic all-GPU-active timings. Replayed mutating kernels can also
make cache-traffic ratios indicative rather than exact.
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme)
    print(f"wrote {readme_path}")


def main() -> None:
    args = parse_args()
    campaigns = [campaign.resolve() for campaign in args.campaign]
    if len(set(campaigns)) != len(campaigns):
        raise ValueError("the same campaign was passed more than once")
    datasets = [load_dataset(campaign) for campaign in campaigns]
    if len({dataset.arm for dataset in datasets}) != len(datasets):
        raise ValueError("each campaign must represent a different arm")
    shared = shared_plot_parameters(datasets)

    for dataset in datasets:
        output_dir = (
            dataset.campaign / "analysis" / "l1_l2_hit_rate_vs_duration_by_family"
        )
        for level, _label, _marker, _color in SERIES:
            render_arm(dataset, shared, output_dir, level)
        write_outputs(dataset, shared, datasets, output_dir)
        print(
            f"{dataset.arm}: plotted {len(dataset.points)} target launches in "
            f"each of two {len(panel_summaries(dataset))}-panel figures"
        )


if __name__ == "__main__":
    main()
