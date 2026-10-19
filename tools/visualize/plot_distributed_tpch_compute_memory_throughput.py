#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot per-kernel compute throughput against dominant memory throughput.

Each point represents one selected rank-0 NCU kernel launch. The x coordinate
is the maximum of its DRAM, L2, and L1 percent-of-peak throughput metrics; the
marker shape identifies which memory level supplied that maximum. The y
coordinate is SM percent-of-peak throughput, and darker points ran longer.

The reports were captured with rank-0 kernel replay and ``--clock-control
none``. Their time-derived throughput percentages and durations describe
isolated execution rather than contention-realistic all-GPU execution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator
except ModuleNotFoundError as error:
    sys.exit(f"error: matplotlib is required: {error}")

import plot_distributed_tpch_instruction_roofline as common


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPUTE_METRIC = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DURATION_METRIC = "gpu__time_duration.sum"
MEMORY_LEVELS = (
    (
        "dram",
        "DRAM",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "o",
    ),
    ("l2", "L2", "lts__throughput.avg.pct_of_peak_sustained_elapsed", "s"),
    ("l1", "L1", "l1tex__throughput.avg.pct_of_peak_sustained_active", "^"),
)
REQUIRED_METRICS = {
    COMPUTE_METRIC: "%",
    DURATION_METRIC: "ns",
    **{metric: "%" for _key, _label, metric, _marker in MEMORY_LEVELS},
}
LEVEL_LABELS = {key: label for key, label, _metric, _marker in MEMORY_LEVELS}
LEVEL_METRICS = {key: metric for key, _label, metric, _marker in MEMORY_LEVELS}
LEVEL_MARKERS = {key: marker for key, _label, _metric, marker in MEMORY_LEVELS}
TIE_PRIORITY = tuple(key for key, _label, _metric, _marker in MEMORY_LEVELS)


@dataclass(frozen=True)
class ThroughputPoint:
    query: int
    launch_id: int
    kernel_family: str
    kernel_name: str
    step_index: str
    step_label: str
    phase_category: str
    phase_name: str
    process_id: str
    device: str
    context: str
    stream: str
    block_size: str
    grid_size: str
    compute_throughput_pct: float
    dram_throughput_pct: float
    l2_throughput_pct: float
    l1_throughput_pct: float
    memory_throughput_pct: float
    dominant_memory_level: str
    dominant_memory_metric: str
    dominant_level_tie_count: int
    duration_ns: float
    duration_ms: float
    source_csv: str


@dataclass(frozen=True)
class CampaignDataset:
    campaign: Path
    arm: str
    display_label: str
    attempts: list[common.SelectedAttempt]
    points: list[ThroughputPoint]
    excluded: list[common.ExcludedLaunch]
    launches_read_by_query: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        action="append",
        type=Path,
        required=True,
        help="Campaign root; repeat for comparable figures with shared scales",
    )
    return parser.parse_args()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def dominant_memory_level(values: dict[str, float]) -> tuple[str, float, int]:
    maximum = max(values.values())
    tied = [level for level in TIE_PRIORITY if values[level] == maximum]
    return tied[0], maximum, len(tied)


def build_point(
    launch: common.RawLaunch, campaign: Path
) -> tuple[ThroughputPoint | None, common.ExcludedLaunch | None]:
    family, reason = common.classify_kernel(launch.kernel_name)
    duration_ns = launch.metrics[DURATION_METRIC]
    relative_source = str(launch.source_csv.relative_to(campaign))
    if family is None:
        return None, common.ExcludedLaunch(
            query=launch.query,
            launch_id=launch.launch_id,
            kernel_name=launch.kernel_name,
            reason=str(reason),
            duration_ns=duration_ns,
            source_csv=relative_source,
        )

    compute = launch.metrics[COMPUTE_METRIC]
    memory_values = {
        level: launch.metrics[metric] for level, metric in LEVEL_METRICS.items()
    }
    invalid = [
        name
        for name, value in {
            "compute_throughput": compute,
            "dram_throughput": memory_values["dram"],
            "l2_throughput": memory_values["l2"],
            "l1_throughput": memory_values["l1"],
            "duration_ns": duration_ns,
        }.items()
        if value < 0 or (name == "duration_ns" and value == 0)
    ]
    if invalid:
        return None, common.ExcludedLaunch(
            query=launch.query,
            launch_id=launch.launch_id,
            kernel_name=launch.kernel_name,
            reason="invalid_" + "_".join(invalid),
            duration_ns=duration_ns,
            source_csv=relative_source,
        )

    level, memory_throughput, tie_count = dominant_memory_level(memory_values)
    step_index, step_label, phase_category, phase_name = common.parse_nvtx_context(
        launch.nvtx_context
    )
    return (
        ThroughputPoint(
            query=launch.query,
            launch_id=launch.launch_id,
            kernel_family=family,
            kernel_name=launch.kernel_name,
            step_index=step_index,
            step_label=step_label,
            phase_category=phase_category,
            phase_name=phase_name,
            process_id=launch.process_id,
            device=launch.device,
            context=launch.context,
            stream=launch.stream,
            block_size=launch.block_size,
            grid_size=launch.grid_size,
            compute_throughput_pct=compute,
            dram_throughput_pct=memory_values["dram"],
            l2_throughput_pct=memory_values["l2"],
            l1_throughput_pct=memory_values["l1"],
            memory_throughput_pct=memory_throughput,
            dominant_memory_level=level,
            dominant_memory_metric=LEVEL_METRICS[level],
            dominant_level_tie_count=tie_count,
            duration_ns=duration_ns,
            duration_ms=duration_ns / 1e6,
            source_csv=relative_source,
        ),
        None,
    )


def load_campaign(campaign: Path) -> CampaignDataset:
    campaign = campaign.resolve()
    if not campaign.is_dir():
        raise FileNotFoundError(f"campaign directory does not exist: {campaign}")
    arm = common.infer_arm(campaign)
    display_label = common.display_label_for_arm(arm)
    attempts = common.select_attempts(campaign, arm)
    points: list[ThroughputPoint] = []
    excluded: list[common.ExcludedLaunch] = []
    launches_read_by_query: dict[int, int] = {}
    print(f"selected {len(attempts)} passing Q1-Q22 attempts for {arm}")
    for selected in attempts:
        launches = common.extract_launches(selected, REQUIRED_METRICS)
        launches_read_by_query[selected.query] = len(launches)
        for launch in launches:
            point, rejected = build_point(launch, campaign)
            if point is not None:
                points.append(point)
            if rejected is not None:
                excluded.append(rejected)
        print(
            f"{arm} Q{selected.query:02d}: read {len(launches)} launches from "
            f"{selected.metrics_csv.name}"
        )
    return CampaignDataset(
        campaign=campaign,
        arm=arm,
        display_label=display_label,
        attempts=attempts,
        points=points,
        excluded=excluded,
        launches_read_by_query=launches_read_by_query,
    )


def validate_dataset(dataset: CampaignDataset) -> dict[str, object]:
    if not dataset.points:
        raise ValueError(f"no plot points for {dataset.arm}")
    count_points = [point for point in dataset.points if point.kernel_family == "count"]
    if not count_points:
        raise ValueError(f"no {common.DESIRED_COUNT_SYMBOL} kernel for {dataset.arm}")
    if any(
        common.DESIRED_COUNT_SYMBOL not in point.kernel_name for point in count_points
    ):
        raise ValueError("non-open_addressing count kernel reached plot data")
    for point in dataset.points:
        values = {
            "dram": point.dram_throughput_pct,
            "l2": point.l2_throughput_pct,
            "l1": point.l1_throughput_pct,
        }
        level, maximum, tie_count = dominant_memory_level(values)
        if (
            level != point.dominant_memory_level
            or maximum != point.memory_throughput_pct
            or tie_count != point.dominant_level_tie_count
        ):
            raise ValueError("inconsistent dominant-memory classification")
    return {
        "minimum_compute_throughput_pct": min(
            point.compute_throughput_pct for point in dataset.points
        ),
        "maximum_compute_throughput_pct": max(
            point.compute_throughput_pct for point in dataset.points
        ),
        "minimum_memory_throughput_pct": min(
            point.memory_throughput_pct for point in dataset.points
        ),
        "maximum_memory_throughput_pct": max(
            point.memory_throughput_pct for point in dataset.points
        ),
        "dominant_level_tie_launch_count": sum(
            point.dominant_level_tie_count > 1 for point in dataset.points
        ),
    }


def common_axis_limit(datasets: list[CampaignDataset]) -> float:
    maximum = max(
        max(
            max(point.memory_throughput_pct, point.compute_throughput_pct)
            for point in dataset.points
        )
        for dataset in datasets
    )
    return max(100.0, math.ceil(maximum / 10.0) * 10.0)


def render_plot(
    dataset: CampaignDataset,
    duration_min_ms: float,
    duration_max_ms: float,
    axis_limit: float,
    output_dir: Path,
) -> list[Path]:
    norm_max = duration_max_ms
    if math.isclose(duration_min_ms, duration_max_ms):
        norm_max = duration_min_ms * 1.01
    norm = LogNorm(vmin=duration_min_ms, vmax=norm_max)

    fig, ax = plt.subplots(figsize=(12.8, 8.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFC")
    for level, label, _metric, marker in MEMORY_LEVELS:
        level_points = sorted(
            (point for point in dataset.points if point.dominant_memory_level == level),
            key=lambda point: point.duration_ns,
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
            s=54,
            edgecolors="#FFFFFF",
            linewidths=0.7,
            alpha=0.87,
            zorder=4,
            label=label,
        )

    ax.set_xlim(0.0, axis_limit)
    ax.set_ylim(0.0, axis_limit)
    major_spacing = 20.0 if axis_limit <= 120.0 else 25.0
    ax.xaxis.set_major_locator(MultipleLocator(major_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(major_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(major_spacing / 2.0))
    ax.yaxis.set_minor_locator(MultipleLocator(major_spacing / 2.0))
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
    ax.set_xlabel("Dominant memory throughput (% of peak)")
    ax.set_ylabel("SM compute throughput (% of peak)")
    ax.set_title(
        f"TPC-H Kernel Throughput — SF1000, 8 GPUs, {dataset.display_label}",
        pad=17,
    )
    for spine in ax.spines.values():
        spine.set_color("#777777")
    ax.tick_params(axis="both", colors="#333333")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="#466B8C",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=9,
            label=label,
        )
        for _level, label, _metric, marker in MEMORY_LEVELS
    ]
    legend = ax.legend(
        handles=legend_handles,
        title="Most utilized memory level",
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#AAAAAA",
        framealpha=0.92,
        fontsize=13,
        title_fontsize=14,
    )
    legend.set_zorder(8)

    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=common.DURATION_CMAP),
        ax=ax,
        pad=0.018,
        fraction=0.045,
    )
    colorbar.set_label(
        "Kernel duration (ms, log scale; darker = longer)",
        fontsize=15,
        labelpad=12,
    )
    colorbar.ax.tick_params(labelsize=13)
    fig.tight_layout()

    outputs = []
    stem = f"compute_memory_throughput_{dataset.arm}"
    for suffix in ("png", "pdf"):
        output = output_dir / f"{stem}.{suffix}"
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        outputs.append(output)
        print(f"wrote {output}")
    plt.close(fig)
    return outputs


def point_rows(points: list[ThroughputPoint]) -> list[dict[str, object]]:
    return [
        asdict(point)
        for point in sorted(points, key=lambda point: (point.query, point.launch_id))
    ]


def write_artifacts(
    dataset: CampaignDataset,
    validation: dict[str, object],
    duration_min_ms: float,
    duration_max_ms: float,
    axis_limit: float,
) -> None:
    output_dir = dataset.campaign / "analysis" / "compute_memory_throughput"
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "compute_memory_throughput_points.csv"
    excluded_path = output_dir / "compute_memory_throughput_excluded_launches.csv"
    summary_path = output_dir / "compute_memory_throughput_summary.json"
    common.write_csv(points_path, point_rows(dataset.points))
    common.write_csv(excluded_path, [asdict(row) for row in dataset.excluded])

    count_points = [point for point in dataset.points if point.kernel_family == "count"]
    summary = {
        "campaign": str(dataset.campaign),
        "arm": dataset.arm,
        "display_label": dataset.display_label,
        "queries": list(common.QUERIES),
        "selected_attempts": {
            f"Q{attempt.query:02d}": str(attempt.attempt)
            for attempt in dataset.attempts
        },
        "launches_read_by_query": {
            f"Q{query:02d}": count
            for query, count in sorted(dataset.launches_read_by_query.items())
        },
        "plotted_launch_count": len(dataset.points),
        "excluded_launch_count": len(dataset.excluded),
        "plotted_kernel_families": dict(
            sorted(Counter(point.kernel_family for point in dataset.points).items())
        ),
        "dominant_memory_level_counts": dict(
            sorted(
                Counter(point.dominant_memory_level for point in dataset.points).items()
            )
        ),
        "excluded_reasons": dict(
            sorted(Counter(row.reason for row in dataset.excluded).items())
        ),
        "count_kernel_rule": {
            "required_symbol_substring": common.DESIRED_COUNT_SYMBOL,
            "plotted_launch_count": len(count_points),
            "queries": sorted({point.query for point in count_points}),
            "unique_symbols": sorted({point.kernel_name for point in count_points}),
        },
        "metrics": {
            "compute_throughput": COMPUTE_METRIC,
            "dram_throughput": LEVEL_METRICS["dram"],
            "l2_throughput": LEVEL_METRICS["l2"],
            "l1_throughput": LEVEL_METRICS["l1"],
            "duration": DURATION_METRIC,
        },
        "encoding": {
            "x": "max(dram_throughput, l2_throughput, l1_throughput)",
            "y": COMPUTE_METRIC,
            "marker": "argmax(dram, l2, l1); ties use DRAM, L2, L1 priority",
            "marker_shapes": LEVEL_MARKERS,
            "color": f"{DURATION_METRIC}; logarithmic, darker is longer",
            "shared_duration_range_ms": [duration_min_ms, duration_max_ms],
            "shared_axis_range_pct": [0.0, axis_limit],
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

    command_campaign = relative_or_absolute(dataset.campaign)
    readme = f"""# Compute vs. dominant-memory throughput: {dataset.display_label}

This directory was generated from the passing Q1-Q22 NCU attempts in
`{command_campaign}` for arm `{dataset.arm}`.

Each point is one selected rank-0 NCU kernel launch. Its x coordinate is the
maximum of the DRAM, L2, and L1 percent-of-peak throughput counters; marker
shape identifies that memory level. Its y coordinate is SM percent-of-peak
throughput, and darker color represents longer duration on a logarithmic scale.

## Metrics

- Compute: `{COMPUTE_METRIC}`
- DRAM: `{LEVEL_METRICS['dram']}`
- L2: `{LEVEL_METRICS['l2']}`
- L1: `{LEVEL_METRICS['l1']}`
- Duration: `{DURATION_METRIC}`

The strict count matcher accepts only symbols containing
`{common.DESIRED_COUNT_SYMBOL}`. Exact memory-throughput ties use the priority
DRAM, then L2, then L1. DRAM, L2, and SM use an elapsed-cycle denominator;
L1 uses the requested active-cycle denominator.

The figures generated together use a shared duration range of
{duration_min_ms:.6g}--{duration_max_ms:.6g} ms and a shared axis range of
0--{axis_limit:.6g}%.

## Capture caveat

These reports profile rank 0 with NCU kernel replay and `--clock-control none`.
Throughput percentages and durations are isolated, time-derived measurements
and are not contention-realistic all-GPU-active results.
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme)
    print(f"wrote {readme_path}")
    render_plot(
        dataset,
        duration_min_ms,
        duration_max_ms,
        axis_limit,
        output_dir,
    )


def main() -> None:
    args = parse_args()
    datasets = [load_campaign(campaign) for campaign in args.campaign]
    if len({dataset.campaign for dataset in datasets}) != len(datasets):
        raise ValueError("the same campaign was passed more than once")
    validations = [validate_dataset(dataset) for dataset in datasets]
    durations = [point.duration_ms for dataset in datasets for point in dataset.points]
    duration_min_ms = min(durations)
    duration_max_ms = max(durations)
    axis_limit = common_axis_limit(datasets)
    for dataset, validation in zip(datasets, validations, strict=True):
        write_artifacts(
            dataset,
            validation,
            duration_min_ms,
            duration_max_ms,
            axis_limit,
        )
        print(
            f"{dataset.arm}: plotted {len(dataset.points)} launches; "
            f"excluded {len(dataset.excluded)}; dominant levels="
            f"{dict(Counter(point.dominant_memory_level for point in dataset.points))}"
        )


if __name__ == "__main__":
    main()
