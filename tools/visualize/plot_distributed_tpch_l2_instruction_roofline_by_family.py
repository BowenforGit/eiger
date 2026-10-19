#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot L2 read-hit instruction rooflines from existing NCU reports.

Each point uses executed warp instructions per read-hit L2 sector on the x-axis
and executed warp instructions per average SM cycle on the y-axis. The read-hit
counter counts sectors of read data actually served by L2. The roof is
architectural: NCU reports reveal the LTS-instance count and the LTS-to-SM cycle
ratio, while the GB200 sustained read-hit sector rate is supplied explicitly as
an architectural input (2 sectors per LTS instance per LTS cycle by default).
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

import plot_distributed_tpch_instruction_roofline as common
import plot_distributed_tpch_instruction_roofline_by_family as family_plot


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_L2_READ_HIT_SECTORS_PER_LTS_CYCLE = 2.0
DEFAULT_ROOF_MIN_DURATION_US = 100.0

L2_REQUIRED_METRICS = {
    "gpu__time_duration.sum": "ns",
    "launch__sm_count": "SM",
    "lts__cycles_elapsed.avg": "cycle",
    "lts__t_sectors.avg": "sector",
    "lts__t_sectors_op_read_lookup_hit.sum": "sector",
    "lts__t_sectors.sum": "sector",
    "sm__cycles_elapsed.avg": "cycle",
    "smsp__inst_executed.sum": "inst",
}


@dataclass(frozen=True)
class RawL2Point:
    query: int
    launch_id: int
    family: str
    kernel_name: str
    instructions: float
    l2_total_sectors: float
    l2_total_sectors_avg: float
    l2_read_hit_sectors: float
    sm_cycles_elapsed: float
    lts_cycles_elapsed: float
    duration_ns: float
    sm_count: int
    inferred_lts_instances: float
    source_csv: str


@dataclass(frozen=True)
class L2Point(family_plot.FamilyPoint):
    instructions: float
    l2_total_sectors: float
    l2_total_sectors_avg: float
    l2_read_hit_sectors: float
    sm_cycles_elapsed: float
    lts_cycles_elapsed: float
    sm_count: int
    inferred_lts_instances: float
    achieved_l2_read_hit_sectors_per_sm_cycle: float
    lts_to_sm_cycle_ratio: float
    source_csv: str


@dataclass(frozen=True)
class L2RoofParameters:
    sm_count: int
    lts_instance_count: int
    l2_read_hit_sectors_per_lts_cycle: float
    lts_to_sm_cycle_ratio_median: float
    lts_to_sm_cycle_ratio_relative_mad: float
    calibration_launch_count: int
    calibration_min_duration_us: float
    instruction_roof_inst_per_cycle: float
    l2_read_hit_roof_sectors_per_sm_cycle: float
    ridge_instruction_per_l2_read_hit_sector: float


@dataclass(frozen=True)
class L2ArmDataset:
    campaign: Path
    arm: str
    arm_label: str
    points: tuple[L2Point, ...]
    instruction_roof: float
    memory_roof: float
    ridge: float
    fact_sheet_row_count: int
    excluded_capture_count: int
    capture_caveat: str
    roofs: L2RoofParameters
    excluded_reason_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        action="append",
        required=True,
        type=Path,
        help="Campaign root; repeat once per arm to share axes and color scale",
    )
    parser.add_argument(
        "--l2-read-hit-sectors-per-lts-cycle",
        type=float,
        default=DEFAULT_L2_READ_HIT_SECTORS_PER_LTS_CYCLE,
        help=(
            "Architectural sustained L2 read-hit rate per LTS instance " "(default: 2)"
        ),
    )
    parser.add_argument(
        "--roof-min-duration-us",
        type=float,
        default=DEFAULT_ROOF_MIN_DURATION_US,
        help="Minimum launch duration for LTS/SM cycle-ratio calibration",
    )
    return parser.parse_args()


def fact_sheet_summary(campaign: Path, arm: str) -> dict[str, object]:
    _, summary_path = family_plot.fact_sheet_paths(campaign, arm)
    if not summary_path.is_file():
        raise FileNotFoundError(
            "missing kernel fact-sheet summary; run the fact-sheet exporter first: "
            f"{summary_path}"
        )
    return json.loads(summary_path.read_text())


def integer_metric(value: float, name: str) -> int:
    rounded = round(value)
    if not math.isclose(value, rounded, abs_tol=1e-8):
        raise ValueError(f"{name} is not integral: {value}")
    return int(rounded)


def raw_l2_point(launch: common.RawLaunch, campaign: Path, family: str) -> RawL2Point:
    metrics = launch.metrics
    positive_metrics = (
        "gpu__time_duration.sum",
        "launch__sm_count",
        "lts__cycles_elapsed.avg",
        "lts__t_sectors.avg",
        "lts__t_sectors_op_read_lookup_hit.sum",
        "lts__t_sectors.sum",
        "sm__cycles_elapsed.avg",
        "smsp__inst_executed.sum",
    )
    if any(metrics[name] <= 0 for name in positive_metrics):
        raise ValueError(
            f"Q{launch.query:02d} launch {launch.launch_id}: nonpositive L2 metric"
        )
    instances = metrics["lts__t_sectors.sum"] / metrics["lts__t_sectors.avg"]
    return RawL2Point(
        query=launch.query,
        launch_id=launch.launch_id,
        family=family,
        kernel_name=launch.kernel_name,
        instructions=metrics["smsp__inst_executed.sum"],
        l2_total_sectors=metrics["lts__t_sectors.sum"],
        l2_total_sectors_avg=metrics["lts__t_sectors.avg"],
        l2_read_hit_sectors=metrics["lts__t_sectors_op_read_lookup_hit.sum"],
        sm_cycles_elapsed=metrics["sm__cycles_elapsed.avg"],
        lts_cycles_elapsed=metrics["lts__cycles_elapsed.avg"],
        duration_ns=metrics["gpu__time_duration.sum"],
        sm_count=integer_metric(metrics["launch__sm_count"], "launch__sm_count"),
        inferred_lts_instances=instances,
        source_csv=str(launch.source_csv.relative_to(campaign)),
    )


def derive_l2_roofs(
    points: list[RawL2Point],
    sectors_per_lts_cycle: float,
    minimum_duration_us: float,
) -> L2RoofParameters:
    if sectors_per_lts_cycle <= 0 or minimum_duration_us <= 0:
        raise ValueError("L2 roof inputs must be positive")
    if not points:
        raise ValueError("cannot derive an L2 roof without target launches")

    sm_counts = {point.sm_count for point in points}
    if len(sm_counts) != 1:
        raise ValueError(f"inconsistent launch__sm_count values: {sm_counts}")
    sm_count = sm_counts.pop()

    rounded_lts_counts = {round(point.inferred_lts_instances) for point in points}
    if len(rounded_lts_counts) != 1:
        raise ValueError(
            f"inconsistent inferred LTS-instance counts: {rounded_lts_counts}"
        )
    lts_instance_count = rounded_lts_counts.pop()
    maximum_instance_rounding_error = max(
        abs(point.inferred_lts_instances - lts_instance_count) for point in points
    )
    if maximum_instance_rounding_error > 0.1:
        raise ValueError(
            "LTS-instance inference is unstable: maximum rounding error "
            f"{maximum_instance_rounding_error:.6g}"
        )

    calibration = [
        point for point in points if point.duration_ns >= minimum_duration_us * 1000.0
    ]
    if len(calibration) < 5:
        raise ValueError("fewer than five launches qualify for L2 roof calibration")
    ratios = [
        point.lts_cycles_elapsed / point.sm_cycles_elapsed for point in calibration
    ]
    ratio_median = statistics.median(ratios)
    ratio_relative_mad = (
        statistics.median(abs(value - ratio_median) for value in ratios) / ratio_median
    )
    if ratio_relative_mad > 0.05:
        raise ValueError(
            "LTS-to-SM cycle conversion is unstable: relative MAD="
            f"{ratio_relative_mad:.4f}"
        )

    instruction_roof = common.SMSPS_PER_SM * sm_count
    l2_roof = lts_instance_count * sectors_per_lts_cycle * ratio_median
    return L2RoofParameters(
        sm_count=sm_count,
        lts_instance_count=lts_instance_count,
        l2_read_hit_sectors_per_lts_cycle=sectors_per_lts_cycle,
        lts_to_sm_cycle_ratio_median=ratio_median,
        lts_to_sm_cycle_ratio_relative_mad=ratio_relative_mad,
        calibration_launch_count=len(calibration),
        calibration_min_duration_us=minimum_duration_us,
        instruction_roof_inst_per_cycle=instruction_roof,
        l2_read_hit_roof_sectors_per_sm_cycle=l2_roof,
        ridge_instruction_per_l2_read_hit_sector=instruction_roof / l2_roof,
    )


def build_l2_point(point: RawL2Point, roofs: L2RoofParameters) -> L2Point:
    intensity = point.instructions / point.l2_read_hit_sectors
    throughput = point.instructions / point.sm_cycles_elapsed
    achieved_l2_rate = point.l2_read_hit_sectors / point.sm_cycles_elapsed
    verdict = (
        "memory"
        if intensity < roofs.ridge_instruction_per_l2_read_hit_sector
        else "compute"
    )
    return L2Point(
        query=point.query,
        launch_id=point.launch_id,
        family=point.family,
        kernel_name=point.kernel_name,
        instruction_intensity=intensity,
        instruction_throughput=throughput,
        duration_ms=point.duration_ns / 1e6,
        bound_verdict=verdict,
        instructions=point.instructions,
        l2_total_sectors=point.l2_total_sectors,
        l2_total_sectors_avg=point.l2_total_sectors_avg,
        l2_read_hit_sectors=point.l2_read_hit_sectors,
        sm_cycles_elapsed=point.sm_cycles_elapsed,
        lts_cycles_elapsed=point.lts_cycles_elapsed,
        sm_count=point.sm_count,
        inferred_lts_instances=point.inferred_lts_instances,
        achieved_l2_read_hit_sectors_per_sm_cycle=achieved_l2_rate,
        lts_to_sm_cycle_ratio=(point.lts_cycles_elapsed / point.sm_cycles_elapsed),
        source_csv=point.source_csv,
    )


def validate_dataset(
    dataset: L2ArmDataset, expected_target_count: int
) -> dict[str, float]:
    family_plot.validate_dataset(dataset, expected_target_count)
    max_instruction_fraction = max(
        point.instruction_throughput / dataset.instruction_roof
        for point in dataset.points
    )
    max_l2_fraction = max(
        point.achieved_l2_read_hit_sectors_per_sm_cycle / dataset.memory_roof
        for point in dataset.points
    )
    if max_instruction_fraction > 1.01:
        raise ValueError(
            f"{dataset.arm}: point exceeds instruction roof: "
            f"{max_instruction_fraction:.4f}x"
        )
    if max_l2_fraction > 1.01:
        raise ValueError(
            f"{dataset.arm}: point exceeds architectural L2 read-hit roof: "
            f"{max_l2_fraction:.4f}x"
        )
    return {
        "maximum_instruction_roof_fraction": max_instruction_fraction,
        "maximum_l2_read_hit_roof_fraction": max_l2_fraction,
    }


def load_dataset(
    campaign: Path,
    sectors_per_lts_cycle: float,
    minimum_duration_us: float,
) -> tuple[L2ArmDataset, dict[str, float]]:
    campaign = campaign.resolve()
    if not campaign.is_dir():
        raise FileNotFoundError(f"campaign directory does not exist: {campaign}")
    arm = common.infer_arm(campaign)
    summary = fact_sheet_summary(campaign, arm)
    expected = summary["validation"]

    raw_points = []
    excluded_reasons: Counter[str] = Counter()
    raw_launch_count = 0
    for selected in common.select_attempts(campaign, arm):
        launches = common.extract_launches(selected, L2_REQUIRED_METRICS)
        raw_launch_count += len(launches)
        for launch in launches:
            family, reason = common.classify_kernel(launch.kernel_name)
            if family is None:
                excluded_reasons[str(reason)] += 1
                continue
            raw_points.append(raw_l2_point(launch, campaign, family))

    if raw_launch_count != int(expected["row_count"]):
        raise ValueError(
            f"{arm}: raw launch count {raw_launch_count} does not match fact sheet "
            f"{expected['row_count']}"
        )
    if len(raw_points) != int(expected["target_kernel_count"]):
        raise ValueError(
            f"{arm}: target launch count {len(raw_points)} does not match fact sheet "
            f"{expected['target_kernel_count']}"
        )
    if sum(excluded_reasons.values()) != int(expected["excluded_capture_count"]):
        raise ValueError(f"{arm}: excluded-capture count does not match fact sheet")

    roofs = derive_l2_roofs(
        raw_points,
        sectors_per_lts_cycle=sectors_per_lts_cycle,
        minimum_duration_us=minimum_duration_us,
    )
    points = tuple(build_l2_point(point, roofs) for point in raw_points)
    dataset = L2ArmDataset(
        campaign=campaign,
        arm=arm,
        arm_label=str(summary["arm_label"]),
        points=points,
        instruction_roof=roofs.instruction_roof_inst_per_cycle,
        memory_roof=roofs.l2_read_hit_roof_sectors_per_sm_cycle,
        ridge=roofs.ridge_instruction_per_l2_read_hit_sector,
        fact_sheet_row_count=raw_launch_count,
        excluded_capture_count=sum(excluded_reasons.values()),
        capture_caveat=str(summary["capture_caveat"]),
        roofs=roofs,
        excluded_reason_counts=dict(sorted(excluded_reasons.items())),
    )
    validation = validate_dataset(dataset, int(expected["target_kernel_count"]))
    return dataset, validation


def l2_panel_rows(dataset: L2ArmDataset) -> list[dict[str, object]]:
    rows = []
    for source in family_plot.panel_summaries(dataset):
        row = dict(source)
        row["l2_read_hit_bound_count"] = row.pop("memory_bound_count")
        row["l2_read_hit_bound_percent"] = row.pop("memory_bound_percent")
        row["instruction_intensity_l2_read_hit_sector_min"] = row.pop(
            "instruction_intensity_min"
        )
        row["instruction_intensity_l2_read_hit_sector_max"] = row.pop(
            "instruction_intensity_max"
        )
        rows.append(row)
    return rows


def point_rows(dataset: L2ArmDataset) -> list[dict[str, object]]:
    rows = []
    for point in sorted(dataset.points, key=lambda item: (item.query, item.launch_id)):
        row = asdict(point)
        active_roof = min(
            dataset.instruction_roof,
            dataset.memory_roof * point.instruction_intensity,
        )
        row.update(
            limiting_roof=(
                "l2_read_hit" if point.bound_verdict == "memory" else "instruction"
            ),
            active_roof_instruction_per_sm_cycle=active_roof,
            fraction_of_active_roof=point.instruction_throughput / active_roof,
            fraction_of_l2_read_hit_roof=(
                point.achieved_l2_read_hit_sectors_per_sm_cycle / dataset.memory_roof
            ),
        )
        rows.append(row)
    return rows


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


def write_outputs(
    dataset: L2ArmDataset,
    validation: dict[str, float],
    shared: family_plot.SharedPlotParameters,
    all_datasets: list[L2ArmDataset],
    output_dir: Path,
) -> None:
    stem = f"l2_instruction_roofline_by_family_{dataset.arm}"
    points_path = output_dir / f"{stem}_points.csv"
    panels_path = output_dir / f"{stem}_panels.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    write_csv(points_path, point_rows(dataset))
    panels = l2_panel_rows(dataset)
    write_csv(panels_path, panels)

    summary = {
        "campaign": str(dataset.campaign),
        "arm": dataset.arm,
        "arm_label": dataset.arm_label,
        "plot": asdict(shared),
        "shared_with_arms": [item.arm for item in all_datasets],
        "roof_parameters": asdict(dataset.roofs),
        "roof_model": {
            "l2_sector_bytes": 32,
            "intensity_denominator": ("lts__t_sectors_op_read_lookup_hit.sum"),
            "architectural_input": (
                "L2 read-hit sectors per LTS instance per LTS cycle"
            ),
            "architectural_input_source": (
                "GB200/Nsight Compute peak-sustained metric property"
            ),
            "cycle_conversion": (
                "median(lts__cycles_elapsed.avg / sm__cycles_elapsed.avg)"
            ),
        },
        "inventory": {
            "raw_profiled_launches": dataset.fact_sheet_row_count,
            "plotted_target_launches": len(dataset.points),
            "excluded_non_target_captures": dataset.excluded_capture_count,
            "panel_count": len(panels),
        },
        "excluded_reason_counts": dataset.excluded_reason_counts,
        "validation": validation,
        "panels": panels,
        "capture_caveat": dataset.capture_caveat,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")

    separator = " " + chr(92) + "\n  "
    campaign_args = separator.join(
        f"--campaign {relative_or_absolute(item.campaign)}" for item in all_datasets
    )
    readme = f"""# L2 read-hit instruction rooflines by kernel family: {dataset.arm_label}

This directory contains a fixed 3 x 4 panel layout with one L2 read-hit
instruction roofline per target kernel family. Axes and the logarithmic duration
color scale are shared across the listed arms.

## Reproduce

```bash
python3 tools/visualize/plot_distributed_tpch_l2_instruction_roofline_by_family.py \\
  {campaign_args}
```

## Definitions

- L2 read-hit sector count: `lts__t_sectors_op_read_lookup_hit.sum` (one
  sector is 32 bytes). This counts read sectors whose requested data was
  present in L2.
- L2 read-hit instruction intensity: `smsp__inst_executed.sum /
  lts__t_sectors_op_read_lookup_hit.sum`.
- Instruction throughput: `smsp__inst_executed.sum / sm__cycles_elapsed.avg`.
- Instruction roof: 4 SMSPs/SM x {dataset.roofs.sm_count} SMs =
  {dataset.instruction_roof:.6g} instructions/SM cycle.
- NCU count aggregation identifies {dataset.roofs.lts_instance_count} LTS
  instances from the total-sector counters `lts__t_sectors.sum /
  lts__t_sectors.avg`; these are used only for topology inference.
- Architectural L2 read-hit rate:
  {dataset.roofs.l2_read_hit_sectors_per_lts_cycle:.6g} sectors per LTS
  instance per LTS cycle.
- Median LTS/SM cycle ratio: {dataset.roofs.lts_to_sm_cycle_ratio_median:.8g},
  using {dataset.roofs.calibration_launch_count} target launches lasting at
  least {dataset.roofs.calibration_min_duration_us:g} us.
- L2 read-hit roof: {dataset.memory_roof:.8g} sectors/SM cycle.
- Ridge: {dataset.ridge:.8g} instructions/L2 read-hit sector.
- Duration: `gpu__time_duration.sum`; darker points are longer.

Only target launches are plotted, and the `count` family accepts only symbols
containing `{common.DESIRED_COUNT_SYMBOL}`.

## Interpretation caveat

The roof is hybrid: the LTS-instance count and clock-domain conversion come
from these NCU reports, while the
{dataset.roofs.l2_read_hit_sectors_per_lts_cycle:g} read-hit
sectors/LTS-cycle term is an explicit GB200 architectural input. The hardware
rate is retained from the total-sector plot and interpreted as the read-hit
delivery capability. The composite `lts__throughput` percentage is deliberately
not used as an L2-sector peak.

{dataset.capture_caveat} Duration colors are isolated replay measurements, not
kernel shares of end-to-end query latency. Replayed mutating kernels can also
make data-movement counters indicative rather than exact.
"""
    (output_dir / "README.md").write_text(readme)
    print(f"wrote {output_dir / 'README.md'}")


def main() -> None:
    args = parse_args()
    campaigns = [campaign.resolve() for campaign in args.campaign]
    if len(set(campaigns)) != len(campaigns):
        raise ValueError("the same campaign was passed more than once")
    loaded = [
        load_dataset(
            campaign,
            sectors_per_lts_cycle=args.l2_read_hit_sectors_per_lts_cycle,
            minimum_duration_us=args.roof_min_duration_us,
        )
        for campaign in campaigns
    ]
    datasets = [item[0] for item in loaded]
    validations = {item[0].arm: item[1] for item in loaded}
    if len({dataset.arm for dataset in datasets}) != len(datasets):
        raise ValueError("each campaign must represent a different arm")
    shared = family_plot.shared_plot_parameters(datasets)

    for dataset in datasets:
        output_dir = dataset.campaign / "analysis" / "l2_instruction_roofline_by_family"
        family_plot.render_arm(
            dataset,
            shared,
            output_dir,
            title_prefix="TPC-H L2 Read-Hit Instruction Rooflines by Kernel Family",
            x_axis_label=(
                "L2 read-hit instruction intensity "
                "(executed warp instructions / read-hit sector)"
            ),
            bound_label="L2-bound",
            roof_legend_label="L2 read-hit + instruction roofs",
            output_stem=f"l2_instruction_roofline_by_family_{dataset.arm}",
        )
        write_outputs(
            dataset,
            validations[dataset.arm],
            shared,
            datasets,
            output_dir,
        )
        print(
            f"{dataset.arm}: plotted {len(dataset.points)} target launches in "
            f"{len(family_plot.panel_summaries(dataset))} family panels; "
            f"L2 read-hit roof={dataset.memory_roof:.6g}, "
            f"ridge={dataset.ridge:.6g}"
        )


if __name__ == "__main__":
    main()
