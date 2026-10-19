#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot one instruction-roofline panel per NCU kernel family and arm.

The input is the validated per-occurrence NCU fact sheet produced by
``export_distributed_tpch_ncu_kernel_fact_sheet.py``.  Every panel in an arm
uses the arm-wide hardware roof.  Axis limits and duration color normalization
are shared across all campaigns passed to one invocation, so corresponding
panels and colors are directly comparable between arms.
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
        "legend.fontsize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import plot_distributed_tpch_instruction_roofline as common


REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_ROWS = 3
PANEL_COLUMNS = 4

# A slot may name alternate implementations used by the two arms.  Keeping the
# alternatives in one slot makes the arm figures spatially comparable.
PANEL_SLOTS = (
    ("evaluate_expr_kernel", "kernel_binary_op_vec"),
    ("DeviceSelectSweepKernel",),
    ("DeviceRadixSortOnesweepKernel",),
    ("DeviceHistogramSweepKernel",),
    ("fill_sequence",),
    ("insert_if_n",),
    ("count",),
    ("retrieve",),
    ("compute_groupid_kernel",),
    ("compute_aggregation",),
    ("compute_aggregation_block",),
    ("join_copartitions_arr",),
)

FAMILY_LABELS = {
    "evaluate_expr_kernel": "Expression evaluation",
    "kernel_binary_op_vec": "Vector binary operation",
    "DeviceSelectSweepKernel": "Select sweep",
    "DeviceRadixSortOnesweepKernel": "Radix-sort onesweep",
    "DeviceHistogramSweepKernel": "Histogram sweep",
    "fill_sequence": "Fill sequence",
    "insert_if_n": "Hash-table insert",
    "count": "Hash-table count",
    "retrieve": "Hash-table retrieve",
    "compute_groupid_kernel": "Group ID",
    "compute_aggregation": "Aggregation",
    "compute_aggregation_block": "Block aggregation",
    "join_copartitions_arr": "Join copartitions",
}


@dataclass(frozen=True)
class FamilyPoint:
    query: int
    launch_id: int
    family: str
    kernel_name: str
    instruction_intensity: float
    instruction_throughput: float
    duration_ms: float
    bound_verdict: str


@dataclass(frozen=True)
class ArmDataset:
    campaign: Path
    arm: str
    arm_label: str
    points: tuple[FamilyPoint, ...]
    instruction_roof: float
    memory_roof: float
    ridge: float
    fact_sheet_row_count: int
    excluded_capture_count: int
    capture_caveat: str


@dataclass(frozen=True)
class SharedPlotParameters:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
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


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid Boolean value in fact sheet: {value!r}")


def fact_sheet_paths(campaign: Path, arm: str) -> tuple[Path, Path]:
    directory = campaign / "analysis" / "kernel_fact_sheet"
    return (
        directory / f"kernel_fact_sheet_{arm}.csv",
        directory / f"kernel_fact_sheet_{arm}_summary.json",
    )


def load_dataset(campaign: Path) -> ArmDataset:
    campaign = campaign.resolve()
    if not campaign.is_dir():
        raise FileNotFoundError(f"campaign directory does not exist: {campaign}")
    arm = common.infer_arm(campaign)
    csv_path, summary_path = fact_sheet_paths(campaign, arm)
    if not csv_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "missing kernel fact sheet; run "
            "export_distributed_tpch_ncu_kernel_fact_sheet.py first: "
            f"{csv_path}"
        )

    summary = json.loads(summary_path.read_text())
    roofs = summary["roof_parameters"]
    validation = summary["validation"]
    points = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not parse_bool(row["target_kernel_match"]):
                continue
            points.append(
                FamilyPoint(
                    query=int(row["query"]),
                    launch_id=int(row["ncu_launch_id"]),
                    family=row["kernel_family"],
                    kernel_name=row["kernel_name"],
                    instruction_intensity=float(
                        row["instruction_intensity_inst_per_dram_sector"]
                    ),
                    instruction_throughput=float(
                        row["instruction_throughput_inst_per_sm_cycle"]
                    ),
                    duration_ms=float(row["kernel_duration_ms"]),
                    bound_verdict=row["bound_verdict"],
                )
            )

    dataset = ArmDataset(
        campaign=campaign,
        arm=arm,
        arm_label=summary["arm_label"],
        points=tuple(points),
        instruction_roof=float(roofs["instruction_roof"]),
        memory_roof=float(roofs["memory_roof"]),
        ridge=float(roofs["ridge"]),
        fact_sheet_row_count=int(validation["row_count"]),
        excluded_capture_count=int(validation["excluded_capture_count"]),
        capture_caveat=summary["capture_caveat"],
    )
    validate_dataset(dataset, int(validation["target_kernel_count"]))
    return dataset


def validate_dataset(dataset: ArmDataset, expected_target_count: int) -> None:
    if not dataset.points:
        raise ValueError(f"{dataset.arm}: no target kernel launches")
    if len(dataset.points) != expected_target_count:
        raise ValueError(
            f"{dataset.arm}: loaded {len(dataset.points)} target launches, "
            f"expected {expected_target_count}"
        )
    if not all(
        math.isfinite(value) and value > 0
        for point in dataset.points
        for value in (
            point.instruction_intensity,
            point.instruction_throughput,
            point.duration_ms,
        )
    ):
        raise ValueError(f"{dataset.arm}: nonpositive or nonfinite plot value")

    allowed_families = {family for slot in PANEL_SLOTS for family in slot}
    unexpected = sorted({point.family for point in dataset.points} - allowed_families)
    if unexpected:
        raise ValueError(f"{dataset.arm}: unexpected kernel families: {unexpected}")

    family_counts = Counter(point.family for point in dataset.points)
    for slot in PANEL_SLOTS:
        present = [family for family in slot if family_counts[family]]
        if len(present) > 1:
            raise ValueError(
                f"{dataset.arm}: multiple alternate families occupy one slot: {present}"
            )

    bad_count = [
        point
        for point in dataset.points
        if point.family == "count"
        and common.DESIRED_COUNT_SYMBOL not in point.kernel_name
    ]
    if bad_count:
        raise ValueError(f"{dataset.arm}: non-target count kernel reached plot data")
    if family_counts["count"] == 0:
        raise ValueError(f"{dataset.arm}: no target count kernel")

    bad_verdicts = []
    for point in dataset.points:
        expected = (
            "memory" if point.instruction_intensity < dataset.ridge else "compute"
        )
        if point.bound_verdict != expected:
            bad_verdicts.append((point.query, point.launch_id))
    if bad_verdicts:
        raise ValueError(
            f"{dataset.arm}: {len(bad_verdicts)} bound verdicts disagree with ridge"
        )


def shared_plot_parameters(datasets: list[ArmDataset]) -> SharedPlotParameters:
    if not datasets:
        raise ValueError("at least one arm dataset is required")
    points = [point for dataset in datasets for point in dataset.points]
    intensities = [point.instruction_intensity for point in points]
    throughputs = [point.instruction_throughput for point in points]
    durations = [point.duration_ms for point in points]
    x_min = common.power_of_ten_floor(min(intensities) * 0.8)
    x_max = common.power_of_ten_ceil(
        max(max(intensities) * 1.25, max(dataset.ridge for dataset in datasets) * 2)
    )
    y_min = common.power_of_ten_floor(min(throughputs) * 0.75)
    y_max = common.power_of_ten_ceil(
        max(
            max(throughputs) * 1.15,
            max(dataset.instruction_roof for dataset in datasets) * 1.1,
        )
    )
    duration_min = min(durations)
    duration_max = max(durations)
    if math.isclose(duration_min, duration_max):
        duration_max = duration_min * 1.01
    return SharedPlotParameters(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        duration_min_ms=duration_min,
        duration_max_ms=duration_max,
    )


def family_for_slot(dataset: ArmDataset, slot: tuple[str, ...]) -> str | None:
    available = {point.family for point in dataset.points}
    present = [family for family in slot if family in available]
    if len(present) > 1:
        raise ValueError(f"multiple families found for panel slot: {present}")
    return present[0] if present else None


def summarize_family(
    dataset: ArmDataset, family: str, panel_slot: int
) -> dict[str, object]:
    points = [point for point in dataset.points if point.family == family]
    if not points:
        raise ValueError(f"{dataset.arm}: empty family {family}")
    memory_count = sum(point.bound_verdict == "memory" for point in points)
    compute_count = len(points) - memory_count
    return {
        "arm": dataset.arm,
        "arm_label": dataset.arm_label,
        "panel_slot": panel_slot,
        "kernel_family": family,
        "display_label": FAMILY_LABELS[family],
        "launch_count": len(points),
        "memory_bound_count": memory_count,
        "compute_bound_count": compute_count,
        "memory_bound_percent": 100.0 * memory_count / len(points),
        "compute_bound_percent": 100.0 * compute_count / len(points),
        "duration_min_ms": min(point.duration_ms for point in points),
        "duration_median_ms": statistics.median(point.duration_ms for point in points),
        "duration_max_ms": max(point.duration_ms for point in points),
        "instruction_intensity_min": min(
            point.instruction_intensity for point in points
        ),
        "instruction_intensity_max": max(
            point.instruction_intensity for point in points
        ),
        "instruction_throughput_min": min(
            point.instruction_throughput for point in points
        ),
        "instruction_throughput_max": max(
            point.instruction_throughput for point in points
        ),
    }


def panel_summaries(dataset: ArmDataset) -> list[dict[str, object]]:
    rows = []
    for slot_index, slot in enumerate(PANEL_SLOTS, start=1):
        family = family_for_slot(dataset, slot)
        if family is not None:
            rows.append(summarize_family(dataset, family, slot_index))
    if sum(int(row["launch_count"]) for row in rows) != len(dataset.points):
        raise ValueError(f"{dataset.arm}: panel counts do not cover every point")
    return rows


def configure_axis(ax: plt.Axes, shared: SharedPlotParameters) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(shared.x_min, shared.x_max)
    ax.set_ylim(shared.y_min, shared.y_max)
    ax.xaxis.set_major_formatter(FuncFormatter(common.format_log_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(common.format_log_tick))
    ax.set_facecolor("#FCFCFC")
    ax.grid(
        which="major",
        color="#CDCDCD",
        linewidth=0.9,
        linestyle=(0, (2, 3)),
        zorder=0,
    )
    ax.grid(
        which="minor",
        color="#E7E7E7",
        linewidth=0.55,
        linestyle=(0, (1, 4)),
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#303030")
    for spine in ax.spines.values():
        spine.set_color("#777777")


def draw_roofs(ax: plt.Axes, dataset: ArmDataset, shared: SharedPlotParameters) -> None:
    memory_x_end = min(dataset.ridge, shared.x_max)
    if memory_x_end > shared.x_min:
        ax.plot(
            [shared.x_min, memory_x_end],
            [
                dataset.memory_roof * shared.x_min,
                dataset.memory_roof * memory_x_end,
            ],
            color="#202020",
            linewidth=2.15,
            solid_capstyle="round",
            zorder=6,
        )
    instruction_x_start = max(dataset.ridge, shared.x_min)
    if instruction_x_start < shared.x_max:
        ax.plot(
            [instruction_x_start, shared.x_max],
            [dataset.instruction_roof, dataset.instruction_roof],
            color="#202020",
            linewidth=2.15,
            solid_capstyle="round",
            zorder=6,
        )
    ax.axvline(
        dataset.ridge,
        color="#606060",
        linewidth=1.25,
        linestyle=(0, (5, 4)),
        zorder=3,
    )


def render_arm(
    dataset: ArmDataset,
    shared: SharedPlotParameters,
    output_dir: Path,
    *,
    title_prefix: str = "TPC-H Instruction Rooflines by Kernel Family",
    x_axis_label: str = (
        "Instruction intensity (executed warp instructions / DRAM sector)"
    ),
    bound_label: str = "Memory-bound",
    roof_legend_label: str = "DRAM + instruction roofs",
    output_stem: str | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    norm = LogNorm(
        vmin=shared.duration_min_ms,
        vmax=shared.duration_max_ms,
    )
    fig, axes = plt.subplots(
        PANEL_ROWS,
        PANEL_COLUMNS,
        figsize=(23.5, 16.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    fig.patch.set_facecolor("white")
    family_points = {
        family: sorted(
            (point for point in dataset.points if point.family == family),
            key=lambda point: point.duration_ms,
        )
        for family in {point.family for point in dataset.points}
    }
    summary_by_slot = {int(row["panel_slot"]): row for row in panel_summaries(dataset)}

    visible_axes = []
    for slot_index, (ax, slot) in enumerate(
        zip(axes.flat, PANEL_SLOTS, strict=True), start=1
    ):
        family = family_for_slot(dataset, slot)
        if family is None:
            ax.set_visible(False)
            continue
        visible_axes.append(ax)
        configure_axis(ax, shared)
        draw_roofs(ax, dataset, shared)
        points = family_points[family]
        ax.scatter(
            [point.instruction_intensity for point in points],
            [point.instruction_throughput for point in points],
            c=[point.duration_ms for point in points],
            cmap=common.DURATION_CMAP,
            norm=norm,
            marker="o",
            s=43,
            edgecolors="#FFFFFF",
            linewidths=0.6,
            alpha=0.88,
            zorder=4,
        )
        row = summary_by_slot[slot_index]
        ax.set_title(
            f"{row['display_label']} (n={row['launch_count']})\n"
            f"{bound_label} {row['memory_bound_percent']:.1f}%\n"
            f"Compute-bound {row['compute_bound_percent']:.1f}%",
            pad=10,
            fontsize=14.5,
            fontweight="bold",
            linespacing=1.18,
        )

    fig.suptitle(
        f"{title_prefix} — SF1000, 8 GPUs, {dataset.arm_label}",
        fontsize=24,
        fontweight="bold",
    )
    fig.supxlabel(x_axis_label, fontsize=19)
    fig.supylabel(
        "Instruction throughput (executed warp instructions / SM cycle)",
        fontsize=19,
    )
    visible_axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#202020",
                linewidth=2.15,
                label=roof_legend_label,
            ),
            Line2D(
                [0],
                [0],
                color="#606060",
                linewidth=1.25,
                linestyle=(0, (5, 4)),
                label="Ridge point",
            ),
        ],
        loc="lower right",
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#BBBBBB",
        fontsize=11.5,
        handlelength=2.6,
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
    fig.get_layout_engine().set(w_pad=0.075, h_pad=0.09, wspace=0.045, hspace=0.11)

    outputs = []
    stem = output_stem or f"instruction_roofline_by_family_{dataset.arm}"
    for suffix in ("png", "pdf"):
        output = output_dir / f"{stem}.{suffix}"
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        outputs.append(output)
        print(f"wrote {output}")
    plt.close(fig)
    return outputs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_metadata(
    dataset: ArmDataset,
    shared: SharedPlotParameters,
    output_dir: Path,
    all_datasets: list[ArmDataset],
) -> None:
    rows = panel_summaries(dataset)
    csv_path = output_dir / f"instruction_roofline_by_family_{dataset.arm}_panels.csv"
    json_path = (
        output_dir / f"instruction_roofline_by_family_{dataset.arm}_summary.json"
    )
    write_csv(csv_path, rows)
    summary = {
        "campaign": str(dataset.campaign),
        "arm": dataset.arm,
        "arm_label": dataset.arm_label,
        "plot": asdict(shared),
        "shared_with_arms": [item.arm for item in all_datasets],
        "roof_parameters": {
            "instruction_roof_inst_per_cycle": dataset.instruction_roof,
            "memory_roof_dram_sectors_per_sm_cycle": dataset.memory_roof,
            "ridge_instruction_intensity": dataset.ridge,
        },
        "inventory": {
            "fact_sheet_rows": dataset.fact_sheet_row_count,
            "plotted_target_launches": len(dataset.points),
            "excluded_non_target_captures": dataset.excluded_capture_count,
            "panel_count": len(rows),
        },
        "panel_slots": [list(slot) for slot in PANEL_SLOTS],
        "panels": rows,
        "capture_caveat": dataset.capture_caveat,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {json_path}")

    command_separator = " " + chr(92) + "\n  "
    command = command_separator.join(
        f"--campaign {relative_or_absolute(item.campaign)}" for item in all_datasets
    )
    readme = f"""# Instruction rooflines by kernel family: {dataset.arm_label}

This directory contains a fixed 3 x 4 panel layout with one roofline per
kernel family. The alternate scalar and vector expression implementations use
the same panel slot across arms. The absent join-copartition slot is hidden in
the Baseline figure.

## Reproduce

```bash
python3 tools/visualize/plot_distributed_tpch_instruction_roofline_by_family.py \\
  {command}
```

## Definitions

- Each point is one validated target-kernel occurrence from the NCU fact sheet.
- Instruction intensity = `smsp__inst_executed.sum / dram__sectors.sum`.
- Instruction throughput = `smsp__inst_executed.sum / sm__cycles_elapsed.avg`.
- Kernel duration comes from `gpu__time_duration.sum` and is shown on one
  logarithmic monochromatic scale shared by all listed arms.
- Every panel in this figure uses the arm-wide instruction roof
  ({dataset.instruction_roof:.6g} instructions/cycle), DRAM roof
  ({dataset.memory_roof:.6g} sectors/SM cycle), and ridge
  ({dataset.ridge:.6g} instructions/sector).
- Axes are shared across every panel and every listed arm.

Only rows with `target_kernel_match=True` are plotted. In particular, the
`count` family accepts only symbols containing
`{common.DESIRED_COUNT_SYMBOL}`.

## Capture caveat

{dataset.capture_caveat} The duration colors are therefore useful for comparing
these isolated captures, not as kernel shares of end-to-end query latency.
"""
    (output_dir / "README.md").write_text(readme)
    print(f"wrote {output_dir / 'README.md'}")


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
        output_dir = dataset.campaign / "analysis" / "instruction_roofline_by_family"
        render_arm(dataset, shared, output_dir)
        write_metadata(dataset, shared, output_dir, datasets)
        print(
            f"{dataset.arm}: plotted {len(dataset.points)} target launches in "
            f"{len(panel_summaries(dataset))} family panels"
        )


if __name__ == "__main__":
    main()
