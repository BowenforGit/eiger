#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Build an instruction roofline from distributed TPC-H NCU reports.

Each point represents one rank-0 NCU kernel launch.  The x coordinate is
executed warp instructions per DRAM sector, the y coordinate is executed warp
instructions per elapsed SM cycle, and point color is NCU kernel duration.

The memory roof is reconstructed from NCU's achieved DRAM-sector rate and its
percent-of-sustained-peak metric, then expressed in the SM-cycle domain used by
the instruction roof.  The reports were captured with rank-0 kernel replay and
``--clock-control none``; their durations are not contention-realistic 8-GPU
query contributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
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
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.ticker import FuncFormatter
except ModuleNotFoundError as error:
    sys.exit(f"error: matplotlib is required: {error}")


REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES = tuple(range(1, 23))
ARM_DISPLAY_LABELS = {
    "legacy_cta64": "Baseline",
    "synthesized_cta64": "Optimized Computation",
}
DESIRED_COUNT_SYMBOL = "open_addressing_ns::count<"
INSTRUCTIONS_PER_SMSP_CYCLE = 1.0
SMSPS_PER_SM = 4
DURATION_CMAP = LinearSegmentedColormap.from_list(
    "duration_blue",
    ("#DCEAF5", "#0B2C4D"),
)

REQUIRED_METRICS = {
    "dram__cycles_elapsed.avg": "cycle",
    "dram__sectors.sum": "sector",
    "dram__sectors_read.sum": "sector",
    "dram__sectors_write.sum": "sector",
    "dram__throughput.sum.pct_of_peak_sustained_elapsed": "%",
    "gpu__time_duration.sum": "ns",
    "launch__sm_count": "SM",
    "sm__cycles_elapsed.avg": "cycle",
    "smsp__inst_executed.sum": "inst",
}

# Match longer/specific names before their prefixes.  Count is deliberately
# absent: it is accepted only through DESIRED_COUNT_SYMBOL below.
NON_COUNT_KERNEL_FAMILIES = (
    ("compute_aggregation_block", "compute_aggregation_block"),
    ("compute_groupid_kernel", "compute_groupid_kernel"),
    ("DeviceRadixSortOnesweepKernel", "DeviceRadixSortOnesweepKernel"),
    ("DeviceHistogramSweepKernel", "DeviceHistogramSweepKernel"),
    ("DeviceSelectSweepKernel", "DeviceSelectSweepKernel"),
    ("join_copartitions_arr", "join_copartitions_arr"),
    ("kernel_binary_op_vec", "kernel_binary_op_vec"),
    ("compute_aggregation", "compute_aggregation"),
    ("evaluate_expr_kernel", "evaluate_expr_kernel"),
    ("insert_if_n", "insert_if_n"),
    ("fill_sequence", "fill_sequence"),
    ("retrieve", "retrieve"),
)

STEP_PATTERN = re.compile(r"<default domain>:step\[(\d+)\]:([^:\"]+)")
PHASE_PATTERN = re.compile(r"<default domain>:phase:([^:\"]+):([^:\"]+)")


@dataclass(frozen=True)
class SelectedAttempt:
    query: int
    attempt: Path
    metrics_csv: Path
    validation_json: Path


@dataclass
class RawLaunch:
    query: int
    launch_id: int
    attempt: Path
    source_csv: Path
    kernel_name: str
    process_id: str
    device: str
    context: str
    stream: str
    block_size: str
    grid_size: str
    nvtx_context: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class RooflinePoint:
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
    instructions: float
    dram_sectors: float
    dram_read_sectors: float
    dram_write_sectors: float
    sm_cycles_elapsed: float
    dram_cycles_elapsed: float
    dram_throughput_pct_of_peak: float
    duration_ns: float
    duration_ms: float
    sm_count: int
    instruction_intensity: float
    instruction_throughput: float
    achieved_dram_sectors_per_sm_cycle: float
    inferred_peak_dram_sectors_per_dram_cycle: float
    inferred_peak_dram_sectors_per_sm_cycle: float
    source_csv: str


@dataclass(frozen=True)
class ExcludedLaunch:
    query: int
    launch_id: int
    kernel_name: str
    reason: str
    duration_ns: float
    source_csv: str


@dataclass(frozen=True)
class RoofParameters:
    sm_count: int
    instruction_roof_inst_per_cycle: float
    memory_roof_sector_per_sm_cycle: float
    dram_peak_sector_per_dram_cycle_median: float
    dram_to_sm_cycle_ratio_median: float
    memory_roof_relative_mad: float
    calibration_launch_count: int
    calibration_min_duration_us: float
    calibration_min_dram_pct: float

    @property
    def ridge_point_instruction_per_sector(self) -> float:
        return (
            self.instruction_roof_inst_per_cycle / self.memory_roof_sector_per_sm_cycle
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        required=True,
        help="Campaign root containing status.csv and arms/",
    )
    parser.add_argument(
        "--arm",
        help="Arm to plot (default: infer the sole passing NCU arm)",
    )
    parser.add_argument(
        "--label",
        help="Presentation label in the title (default: known arm label or arm name)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: CAMPAIGN/analysis/instruction_roofline)",
    )
    parser.add_argument(
        "--roof-min-duration-us",
        type=float,
        default=100.0,
        help="Minimum duration for memory-roof calibration launches",
    )
    parser.add_argument(
        "--roof-min-dram-pct",
        type=float,
        default=1.0,
        help="Minimum DRAM sustained-peak percentage for roof calibration",
    )
    parser.add_argument(
        "--annotate-longest",
        type=int,
        default=0,
        help="Annotate this many longest launches (default: 0)",
    )
    return parser.parse_args()


def query_number(value: str) -> int:
    match = re.fullmatch(r"[Qq]?(\d+)", value.strip())
    if match is None:
        raise ValueError(f"invalid query value in status.csv: {value!r}")
    return int(match.group(1))


def infer_arm(campaign: Path) -> str:
    status_path = campaign / "status.csv"
    if not status_path.is_file():
        raise FileNotFoundError(f"missing campaign status file: {status_path}")

    with status_path.open(newline="") as stream:
        arms = {
            row["arm"]
            for row in csv.DictReader(stream)
            if row.get("stage") == "ncu_profiles"
            and row.get("status") == "pass"
            and row.get("arm")
        }
    if len(arms) != 1:
        raise ValueError(
            "could not infer a unique passing NCU arm from status.csv; "
            f"found {sorted(arms)}; pass --arm explicitly"
        )
    return arms.pop()


def display_label_for_arm(arm: str) -> str:
    return ARM_DISPLAY_LABELS.get(arm, arm.replace("_", " "))


def select_attempts(campaign: Path, arm: str) -> list[SelectedAttempt]:
    status_path = campaign / "status.csv"
    if not status_path.is_file():
        raise FileNotFoundError(f"missing campaign status file: {status_path}")

    selected: dict[int, Path] = {}
    with status_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("arm") != arm or row.get("stage") != "ncu_profiles":
                continue
            if row.get("status") != "pass":
                continue
            query = query_number(row["query"])
            attempt = Path(row["attempt"])
            if not attempt.is_absolute():
                attempt = campaign / attempt
            elif not attempt.is_dir() and "arms" in attempt.parts:
                # Campaigns may be archived below an additional directory
                # after collection, leaving absolute attempt paths stale.
                arms_index = attempt.parts.index("arms")
                attempt = campaign.joinpath(*attempt.parts[arms_index:])
            selected[query] = attempt.resolve()

    expected = set(QUERIES)
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise ValueError(
            f"expected passing Q1-Q22 attempts; missing={missing}, extra={extra}"
        )

    attempts = []
    for query in QUERIES:
        attempt = selected[query]
        metrics_csv = attempt / f"q{query:02d}_ncu_metrics.csv"
        validation_json = attempt / f"q{query:02d}_ncu_validation.json"
        if not metrics_csv.is_file() or metrics_csv.stat().st_size == 0:
            raise FileNotFoundError(f"missing NCU metrics CSV: {metrics_csv}")
        if not validation_json.is_file():
            raise FileNotFoundError(f"missing NCU validation JSON: {validation_json}")
        validation = json.loads(validation_json.read_text())
        if validation.get("status") != "pass":
            raise ValueError(f"NCU validation did not pass: {validation_json}")
        if validation.get("missing_requested_metrics"):
            raise ValueError(f"missing requested metrics: {validation_json}")
        if validation.get("missing_launch_metrics"):
            raise ValueError(f"missing launch metrics: {validation_json}")
        attempts.append(
            SelectedAttempt(
                query=query,
                attempt=attempt,
                metrics_csv=metrics_csv,
                validation_json=validation_json,
            )
        )
    return attempts


def metric_value(raw: str, metric: str, source: Path) -> float:
    try:
        value = float(raw.replace(",", ""))
    except ValueError as error:
        raise ValueError(f"{source}: non-numeric value {raw!r} for {metric}") from error
    if not math.isfinite(value):
        raise ValueError(f"{source}: non-finite value for {metric}")
    return value


def extract_launches(
    selected: SelectedAttempt,
    required_metrics: dict[str, str] | None = None,
) -> list[RawLaunch]:
    if required_metrics is None:
        required_metrics = REQUIRED_METRICS
    launches: dict[int, RawLaunch] = {}
    with selected.metrics_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"empty NCU CSV: {selected.metrics_csv}")
        nvtx_fields = [
            field
            for field in reader.fieldnames
            if field.startswith("thread Domain:Push/Pop_Range:")
        ]
        if len(nvtx_fields) != 1:
            raise ValueError(
                f"{selected.metrics_csv}: expected one thread NVTX column, "
                f"found {len(nvtx_fields)}"
            )
        nvtx_field = nvtx_fields[0]

        for row in reader:
            launch_id = int(row["ID"])
            kernel_name = row["Kernel Name"]
            launch = launches.get(launch_id)
            if launch is None:
                launch = RawLaunch(
                    query=selected.query,
                    launch_id=launch_id,
                    attempt=selected.attempt,
                    source_csv=selected.metrics_csv,
                    kernel_name=kernel_name,
                    process_id=row["Process ID"],
                    device=row["Device"],
                    context=row["Context"],
                    stream=row["Stream"],
                    block_size=row["Block Size"],
                    grid_size=row["Grid Size"],
                    nvtx_context=row[nvtx_field],
                    metrics={},
                )
                launches[launch_id] = launch
            elif launch.kernel_name != kernel_name:
                raise ValueError(
                    f"{selected.metrics_csv}: ID {launch_id} has multiple "
                    "kernel names"
                )

            metric = row["Metric Name"]
            if metric not in required_metrics:
                continue
            expected_unit = required_metrics[metric]
            if row["Metric Unit"] != expected_unit:
                raise ValueError(
                    f"{selected.metrics_csv}: {metric} unit is "
                    f"{row['Metric Unit']!r}, expected {expected_unit!r}"
                )
            if metric in launch.metrics:
                raise ValueError(
                    f"{selected.metrics_csv}: duplicate {metric} for "
                    f"launch ID {launch_id}"
                )
            launch.metrics[metric] = metric_value(
                row["Metric Value"], metric, selected.metrics_csv
            )

    if not launches:
        raise ValueError(f"no launches found in {selected.metrics_csv}")
    for launch in launches.values():
        missing = sorted(set(required_metrics) - set(launch.metrics))
        if missing:
            raise ValueError(
                f"{selected.metrics_csv}: launch ID {launch.launch_id} "
                f"is missing metrics {missing}"
            )
    return [launches[index] for index in sorted(launches)]


def classify_kernel(kernel_name: str) -> tuple[str | None, str | None]:
    if DESIRED_COUNT_SYMBOL in kernel_name:
        return "count", None
    for family, token in NON_COUNT_KERNEL_FAMILIES:
        if token in kernel_name:
            return family, None
    if "::count<" in kernel_name:
        return None, "non_target_count_symbol"
    if "counting_iterator" in kernel_name:
        return None, "incidental_counting_iterator_match"
    return None, "unmatched_capture"


def parse_nvtx_context(context: str) -> tuple[str, str, str, str]:
    step_match = STEP_PATTERN.search(context)
    phase_match = PHASE_PATTERN.search(context)
    step_index = step_match.group(1) if step_match else ""
    step_label = step_match.group(2) if step_match else ""
    phase_category = phase_match.group(1) if phase_match else ""
    phase_name = phase_match.group(2) if phase_match else ""
    return step_index, step_label, phase_category, phase_name


def build_point(
    launch: RawLaunch, campaign: Path
) -> tuple[RooflinePoint | None, ExcludedLaunch | None]:
    family, reason = classify_kernel(launch.kernel_name)
    duration_ns = launch.metrics["gpu__time_duration.sum"]
    relative_source = str(launch.source_csv.relative_to(campaign))
    if family is None:
        return None, ExcludedLaunch(
            query=launch.query,
            launch_id=launch.launch_id,
            kernel_name=launch.kernel_name,
            reason=str(reason),
            duration_ns=duration_ns,
            source_csv=relative_source,
        )

    instructions = launch.metrics["smsp__inst_executed.sum"]
    dram_sectors = launch.metrics["dram__sectors.sum"]
    dram_read = launch.metrics["dram__sectors_read.sum"]
    dram_write = launch.metrics["dram__sectors_write.sum"]
    sm_cycles = launch.metrics["sm__cycles_elapsed.avg"]
    dram_cycles = launch.metrics["dram__cycles_elapsed.avg"]
    dram_pct = launch.metrics["dram__throughput.sum.pct_of_peak_sustained_elapsed"]
    sm_count_value = launch.metrics["launch__sm_count"]
    sm_count = int(round(sm_count_value))

    positive_values = {
        "instructions": instructions,
        "dram_sectors": dram_sectors,
        "sm_cycles": sm_cycles,
        "dram_cycles": dram_cycles,
        "duration_ns": duration_ns,
        "sm_count": sm_count_value,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        return None, ExcludedLaunch(
            query=launch.query,
            launch_id=launch.launch_id,
            kernel_name=launch.kernel_name,
            reason="nonpositive_" + "_".join(invalid),
            duration_ns=duration_ns,
            source_csv=relative_source,
        )
    if not math.isclose(sm_count_value, sm_count, abs_tol=1e-9):
        raise ValueError(f"{launch.source_csv}: non-integral SM count {sm_count_value}")
    if not math.isclose(
        dram_sectors, dram_read + dram_write, rel_tol=1e-9, abs_tol=1.0
    ):
        raise ValueError(
            f"{launch.source_csv}: launch {launch.launch_id} DRAM sectors "
            "do not equal read + write sectors"
        )

    step_index, step_label, phase_category, phase_name = parse_nvtx_context(
        launch.nvtx_context
    )
    instruction_intensity = instructions / dram_sectors
    instruction_throughput = instructions / sm_cycles
    achieved_dram_rate = dram_sectors / sm_cycles
    if dram_pct > 0:
        peak_dram_per_dram_cycle = dram_sectors / dram_cycles / (dram_pct / 100.0)
        peak_dram_per_sm_cycle = achieved_dram_rate / (dram_pct / 100.0)
    else:
        peak_dram_per_dram_cycle = math.nan
        peak_dram_per_sm_cycle = math.nan

    return (
        RooflinePoint(
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
            instructions=instructions,
            dram_sectors=dram_sectors,
            dram_read_sectors=dram_read,
            dram_write_sectors=dram_write,
            sm_cycles_elapsed=sm_cycles,
            dram_cycles_elapsed=dram_cycles,
            dram_throughput_pct_of_peak=dram_pct,
            duration_ns=duration_ns,
            duration_ms=duration_ns / 1e6,
            sm_count=sm_count,
            instruction_intensity=instruction_intensity,
            instruction_throughput=instruction_throughput,
            achieved_dram_sectors_per_sm_cycle=achieved_dram_rate,
            inferred_peak_dram_sectors_per_dram_cycle=(peak_dram_per_dram_cycle),
            inferred_peak_dram_sectors_per_sm_cycle=peak_dram_per_sm_cycle,
            source_csv=relative_source,
        ),
        None,
    )


def derive_roofs(
    points: list[RooflinePoint],
    minimum_duration_us: float,
    minimum_dram_pct: float,
) -> RoofParameters:
    if minimum_duration_us <= 0 or minimum_dram_pct <= 0:
        raise ValueError("memory-roof calibration thresholds must be positive")
    sm_counts = {point.sm_count for point in points}
    if len(sm_counts) != 1:
        raise ValueError(f"inconsistent launch__sm_count values: {sm_counts}")
    sm_count = sm_counts.pop()

    calibration = [
        point
        for point in points
        if point.duration_ns >= minimum_duration_us * 1000.0
        and point.dram_throughput_pct_of_peak >= minimum_dram_pct
        and math.isfinite(point.inferred_peak_dram_sectors_per_sm_cycle)
    ]
    if len(calibration) < 5:
        raise ValueError("fewer than five launches qualify for memory-roof calibration")
    peak_sm_rates = [
        point.inferred_peak_dram_sectors_per_sm_cycle for point in calibration
    ]
    peak_dram_rates = [
        point.inferred_peak_dram_sectors_per_dram_cycle for point in calibration
    ]
    cycle_ratios = [
        point.dram_cycles_elapsed / point.sm_cycles_elapsed for point in calibration
    ]
    memory_roof = statistics.median(peak_sm_rates)
    peak_dram = statistics.median(peak_dram_rates)
    cycle_ratio = statistics.median(cycle_ratios)
    relative_mad = (
        statistics.median(abs(value - memory_roof) for value in peak_sm_rates)
        / memory_roof
    )
    if relative_mad > 0.10:
        raise ValueError(
            f"memory-roof calibration is unstable: relative MAD={relative_mad:.3f}"
        )

    instruction_roof = INSTRUCTIONS_PER_SMSP_CYCLE * SMSPS_PER_SM * sm_count
    return RoofParameters(
        sm_count=sm_count,
        instruction_roof_inst_per_cycle=instruction_roof,
        memory_roof_sector_per_sm_cycle=memory_roof,
        dram_peak_sector_per_dram_cycle_median=peak_dram,
        dram_to_sm_cycle_ratio_median=cycle_ratio,
        memory_roof_relative_mad=relative_mad,
        calibration_launch_count=len(calibration),
        calibration_min_duration_us=minimum_duration_us,
        calibration_min_dram_pct=minimum_dram_pct,
    )


def validate_points(
    points: list[RooflinePoint], roofs: RoofParameters
) -> dict[str, float | int]:
    if not points:
        raise ValueError("no roofline points")
    bad_count = [
        point
        for point in points
        if point.kernel_family == "count"
        and DESIRED_COUNT_SYMBOL not in point.kernel_name
    ]
    if bad_count:
        raise ValueError("non-open_addressing count kernel reached plot data")
    if not any(point.kernel_family == "count" for point in points):
        raise ValueError("no validated open_addressing_ns::count kernel found")

    max_instruction_fraction = max(
        point.instruction_throughput / roofs.instruction_roof_inst_per_cycle
        for point in points
    )
    if max_instruction_fraction > 1.01:
        raise ValueError(
            "a point exceeds the instruction roof by more than 1%: "
            f"{max_instruction_fraction:.4f}x"
        )
    max_memory_fraction = max(
        point.achieved_dram_sectors_per_sm_cycle / roofs.memory_roof_sector_per_sm_cycle
        for point in points
    )
    return {
        "maximum_instruction_roof_fraction": max_instruction_fraction,
        "maximum_memory_roof_fraction": max_memory_fraction,
        "points_above_memory_roof": sum(
            point.achieved_dram_sectors_per_sm_cycle
            > roofs.memory_roof_sector_per_sm_cycle
            for point in points
        ),
    }


def bound_summary(points: list[RooflinePoint], ridge: float) -> dict[str, float | int]:
    if not points:
        raise ValueError("cannot classify an empty point set")
    memory_count = sum(point.instruction_intensity < ridge for point in points)
    compute_count = len(points) - memory_count
    return {
        "memory_bound_launch_count": memory_count,
        "compute_bound_launch_count": compute_count,
        "memory_bound_percent": 100.0 * memory_count / len(points),
        "compute_bound_percent": 100.0 * compute_count / len(points),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        print(f"wrote {path}")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def point_rows(
    points: list[RooflinePoint], roofs: RoofParameters
) -> list[dict[str, object]]:
    rows = []
    ridge = roofs.ridge_point_instruction_per_sector
    for point in sorted(points, key=lambda item: (item.query, item.launch_id)):
        row = asdict(point)
        memory_ceiling = (
            roofs.memory_roof_sector_per_sm_cycle * point.instruction_intensity
        )
        active_ceiling = min(memory_ceiling, roofs.instruction_roof_inst_per_cycle)
        row.update(
            limiting_roof=(
                "memory" if point.instruction_intensity < ridge else "instruction"
            ),
            active_roof_instruction_per_cycle=active_ceiling,
            fraction_of_active_roof=(point.instruction_throughput / active_ceiling),
            fraction_of_instruction_roof=(
                point.instruction_throughput / roofs.instruction_roof_inst_per_cycle
            ),
            fraction_of_memory_roof=(
                point.achieved_dram_sectors_per_sm_cycle
                / roofs.memory_roof_sector_per_sm_cycle
            ),
        )
        rows.append(row)
    return rows


def power_of_ten_floor(value: float) -> float:
    return 10.0 ** math.floor(math.log10(value))


def power_of_ten_ceil(value: float) -> float:
    return 10.0 ** math.ceil(math.log10(value))


def format_log_tick(value: float, _position: float) -> str:
    if value <= 0:
        return ""
    exponent = round(math.log10(value))
    if math.isclose(value, 10.0**exponent, rel_tol=1e-8):
        return rf"$10^{{{exponent}}}$"
    return ""


def annotation_label(point: RooflinePoint) -> str:
    return f"Q{point.query:02d} {point.kernel_family}"


def render_plot(
    points: list[RooflinePoint],
    roofs: RoofParameters,
    output_dir: Path,
    annotate_longest: int,
    arm: str,
    display_label: str,
) -> list[Path]:
    if annotate_longest < 0:
        raise ValueError("--annotate-longest must be nonnegative")
    intensities = [point.instruction_intensity for point in points]
    throughputs = [point.instruction_throughput for point in points]
    durations_ms = [point.duration_ms for point in points]
    ridge = roofs.ridge_point_instruction_per_sector
    bounds = bound_summary(points, ridge)

    x_min = power_of_ten_floor(min(intensities) * 0.8)
    x_max = power_of_ten_ceil(max(max(intensities) * 1.25, ridge * 2.0))
    y_min = power_of_ten_floor(min(throughputs) * 0.75)
    y_max = power_of_ten_ceil(
        max(max(throughputs) * 1.15, roofs.instruction_roof_inst_per_cycle * 1.1)
    )
    if x_min == x_max:
        x_max *= 10.0
    if y_min == y_max:
        y_max *= 10.0

    duration_min = min(durations_ms)
    duration_max = max(durations_ms)
    if math.isclose(duration_min, duration_max):
        duration_max = duration_min * 1.01
    norm = LogNorm(vmin=duration_min, vmax=duration_max)
    cmap = DURATION_CMAP

    fig, ax = plt.subplots(figsize=(12.8, 8.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFC")

    ordered = sorted(points, key=lambda point: point.duration_ns)
    ax.scatter(
        [point.instruction_intensity for point in ordered],
        [point.instruction_throughput for point in ordered],
        c=[point.duration_ms for point in ordered],
        cmap=cmap,
        norm=norm,
        marker="o",
        s=42,
        edgecolors="#FFFFFF",
        linewidths=0.65,
        alpha=0.86,
        zorder=4,
    )
    ax.axvline(
        ridge,
        color="#606060",
        linewidth=1.35,
        linestyle=(0, (5, 4)),
        zorder=3,
    )

    memory_x_end = min(ridge, x_max)
    if memory_x_end > x_min:
        ax.plot(
            [x_min, memory_x_end],
            [
                roofs.memory_roof_sector_per_sm_cycle * x_min,
                roofs.memory_roof_sector_per_sm_cycle * memory_x_end,
            ],
            color="#202020",
            linewidth=2.3,
            solid_capstyle="round",
            zorder=6,
        )
    instruction_x_start = max(ridge, x_min)
    if instruction_x_start < x_max:
        ax.plot(
            [instruction_x_start, x_max],
            [
                roofs.instruction_roof_inst_per_cycle,
                roofs.instruction_roof_inst_per_cycle,
            ],
            color="#202020",
            linewidth=2.3,
            solid_capstyle="round",
            zorder=6,
        )

    memory_label_x = 10.0 ** (
        0.75 * math.log10(x_min) + 0.25 * math.log10(memory_x_end)
    )
    memory_roof_annotation = ax.annotate(
        f"DRAM roof: {roofs.memory_roof_sector_per_sm_cycle:.1f} " "sectors/SM cycle",
        xy=(
            memory_label_x,
            roofs.memory_roof_sector_per_sm_cycle * memory_label_x,
        ),
        xytext=(3, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=13.0,
        color="#202020",
        rotation=0,
        rotation_mode="anchor",
        zorder=7,
    )
    label_y = 0.025
    for label_x, text in (
        (
            math.sqrt(x_min * ridge),
            f"Memory-bound: {bounds['memory_bound_percent']:.1f}%",
        ),
        (
            math.sqrt(ridge * x_max),
            f"Compute-bound: {bounds['compute_bound_percent']:.1f}%",
        ),
    ):
        ax.text(
            label_x,
            label_y,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=14.0,
            fontweight="bold",
            color="#303030",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 2.5,
            },
            zorder=8,
        )
    ax.annotate(
        f"Instruction roof: 4 × {roofs.sm_count} SMs = "
        f"{roofs.instruction_roof_inst_per_cycle:.0f} inst/cycle",
        xy=(
            math.sqrt(instruction_x_start * x_max),
            roofs.instruction_roof_inst_per_cycle,
        ),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=13.0,
        color="#202020",
        zorder=7,
    )

    for index, point in enumerate(
        sorted(points, key=lambda item: item.duration_ns, reverse=True)[
            :annotate_longest
        ]
    ):
        x_offset = 6 if index % 2 == 0 else -6
        horizontal_alignment = "left" if x_offset > 0 else "right"
        ax.annotate(
            annotation_label(point),
            (point.instruction_intensity, point.instruction_throughput),
            xytext=(x_offset, 5 + 4 * (index % 3)),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="bottom",
            fontsize=10.0,
            color="#303030",
            zorder=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_formatter(FuncFormatter(format_log_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(format_log_tick))
    ax.set_xlabel("Instruction intensity (executed warp instructions / DRAM sector)")
    ax.set_ylabel("Instruction throughput (executed warp instructions / SM cycle)")
    ax.set_title(
        f"TPC-H Instruction Roofline — SF1000, 8 GPUs, {display_label}",
        pad=17,
    )
    ax.grid(
        which="major",
        color="#D4D4D4",
        linewidth=0.8,
        linestyle=(0, (2, 3)),
        zorder=0,
    )
    ax.grid(
        which="minor",
        color="#ECECEC",
        linewidth=0.45,
        linestyle=(0, (1, 4)),
        zorder=0,
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#777777")
    ax.tick_params(axis="both", colors="#333333")

    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
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
    fig.canvas.draw()
    memory_label_y = roofs.memory_roof_sector_per_sm_cycle * memory_label_x
    angle_x_end = min(memory_x_end, memory_label_x * 1.5)
    angle_y_end = roofs.memory_roof_sector_per_sm_cycle * angle_x_end
    start_display = ax.transData.transform((memory_label_x, memory_label_y))
    end_display = ax.transData.transform((angle_x_end, angle_y_end))
    memory_roof_annotation.set_rotation(
        math.degrees(
            math.atan2(
                end_display[1] - start_display[1],
                end_display[0] - start_display[0],
            )
        )
    )

    outputs = []
    figure_stem = f"instruction_roofline_{arm}"
    for suffix in ("png", "pdf"):
        output = output_dir / f"{figure_stem}.{suffix}"
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        outputs.append(output)
        print(f"wrote {output}")
    plt.close(fig)
    return outputs


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_readme(
    path: Path,
    campaign: Path,
    points: list[RooflinePoint],
    excluded: list[ExcludedLaunch],
    roofs: RoofParameters,
    arm: str,
    display_label: str,
) -> None:
    command_campaign = relative_or_absolute(campaign)
    text = f"""# Instruction roofline: SF1000 8-GPU {display_label}

This directory was generated from the passing Q1-Q22 NCU attempts in
`{command_campaign}` for arm `{arm}`. Each plotted point is one rank-0 NCU launch.

## Reproduce

```bash
python3 tools/visualize/plot_distributed_tpch_instruction_roofline.py \\
  --campaign {command_campaign}
```

## Definitions

- Instruction count: `smsp__inst_executed.sum` (executed warp instructions).
- DRAM traffic: `dram__sectors.sum` (read plus write sectors).
- Instruction intensity: instructions / DRAM sectors.
- Instruction throughput: instructions / `sm__cycles_elapsed.avg`.
- Duration color: `gpu__time_duration.sum`; larger values use darker colors.
- Instruction roof: 4 SMSPs/SM x {roofs.sm_count} SMs =
  {roofs.instruction_roof_inst_per_cycle:.6g} instructions/SM cycle.
- Memory roof: {roofs.memory_roof_sector_per_sm_cycle:.6g} DRAM sectors/SM
  cycle. For each calibration launch it is reconstructed as
  `(dram__sectors.sum / sm__cycles_elapsed.avg) /
  (dram__throughput.sum.pct_of_peak_sustained_elapsed / 100)`, then the
  campaign median is used. The equivalent DRAM-domain peak is
  {roofs.dram_peak_sector_per_dram_cycle_median:.6g} sectors/DRAM cycle.
- Ridge point: {roofs.ridge_point_instruction_per_sector:.6g}
  instructions/sector.

The strict count matcher accepts only kernel symbols containing
`{DESIRED_COUNT_SYMBOL}`. It rejects other `count` symbols and incidental
`counting_iterator` template matches.

## Inventory and caveat

- Plotted launches: {len(points)}
- Excluded launches: {len(excluded)}
- Roof calibration launches: {roofs.calibration_launch_count}

These reports profile rank 0 with NCU kernel replay and `--clock-control none`.
Time-derived quantities, including the duration color, are isolated and can be
optimistic relative to an all-GPU-active execution. They must not be read as
per-kernel contributions to end-to-end 8-GPU query latency.
"""
    path.write_text(text)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    campaign = args.campaign.resolve()
    if not campaign.is_dir():
        raise FileNotFoundError(f"campaign directory does not exist: {campaign}")
    arm = args.arm or infer_arm(campaign)
    display_label = args.label or display_label_for_arm(arm)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else campaign / "analysis" / "instruction_roofline"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts = select_attempts(campaign, arm)
    print(f"selected {len(attempts)} passing Q1-Q22 attempts for {arm}")
    points = []
    excluded = []
    per_query_launches: dict[int, int] = {}
    for selected in attempts:
        launches = extract_launches(selected)
        per_query_launches[selected.query] = len(launches)
        for launch in launches:
            point, rejected = build_point(launch, campaign)
            if point is not None:
                points.append(point)
            if rejected is not None:
                excluded.append(rejected)
        print(
            f"Q{selected.query:02d}: read {len(launches)} launches from "
            f"{selected.metrics_csv.name}"
        )

    roofs = derive_roofs(
        points,
        minimum_duration_us=args.roof_min_duration_us,
        minimum_dram_pct=args.roof_min_dram_pct,
    )
    validation = validate_points(points, roofs)
    bounds = bound_summary(points, roofs.ridge_point_instruction_per_sector)

    points_path = output_dir / "instruction_roofline_points.csv"
    excluded_path = output_dir / "instruction_roofline_excluded_launches.csv"
    summary_path = output_dir / "instruction_roofline_summary.json"
    write_csv(points_path, point_rows(points, roofs))
    write_csv(excluded_path, [asdict(row) for row in excluded])

    count_points = [point for point in points if point.kernel_family == "count"]
    summary = {
        "campaign": str(campaign),
        "arm": arm,
        "display_label": display_label,
        "queries": list(QUERIES),
        "selected_attempts": {
            f"Q{attempt.query:02d}": str(attempt.attempt) for attempt in attempts
        },
        "launches_read_by_query": {
            f"Q{query:02d}": count
            for query, count in sorted(per_query_launches.items())
        },
        "plotted_launch_count": len(points),
        "excluded_launch_count": len(excluded),
        "plotted_kernel_families": dict(
            sorted(Counter(point.kernel_family for point in points).items())
        ),
        "excluded_reasons": dict(
            sorted(Counter(row.reason for row in excluded).items())
        ),
        "count_kernel_rule": {
            "required_symbol_substring": DESIRED_COUNT_SYMBOL,
            "plotted_launch_count": len(count_points),
            "queries": sorted({point.query for point in count_points}),
            "unique_symbols": sorted({point.kernel_name for point in count_points}),
        },
        "metrics": {
            "instruction_count": "smsp__inst_executed.sum",
            "dram_sectors": "dram__sectors.sum",
            "elapsed_cycles": "sm__cycles_elapsed.avg",
            "duration": "gpu__time_duration.sum",
            "sm_count": "launch__sm_count",
            "dram_cycles": "dram__cycles_elapsed.avg",
            "dram_peak_percentage": (
                "dram__throughput.sum.pct_of_peak_sustained_elapsed"
            ),
        },
        "formulas": {
            "instruction_intensity": ("smsp__inst_executed.sum / dram__sectors.sum"),
            "instruction_throughput": (
                "smsp__inst_executed.sum / sm__cycles_elapsed.avg"
            ),
            "inferred_peak_dram_sectors_per_dram_cycle": (
                "(dram__sectors.sum / dram__cycles_elapsed.avg) / "
                "(dram__throughput.sum.pct_of_peak_sustained_elapsed / 100)"
            ),
            "inferred_peak_dram_sectors_per_sm_cycle": (
                "(dram__sectors.sum / sm__cycles_elapsed.avg) / "
                "(dram__throughput.sum.pct_of_peak_sustained_elapsed / 100)"
            ),
            "memory_roof": (
                "median inferred_peak_dram_sectors_per_sm_cycle over "
                "calibration launches"
            ),
            "roof": "min(memory_roof * intensity, instruction_roof)",
        },
        "roof_parameters": {
            **asdict(roofs),
            "ridge_point_instruction_per_sector": (
                roofs.ridge_point_instruction_per_sector
            ),
        },
        "bound_classification": bounds,
        "validation": validation,
        "capture_caveat": (
            "Rank-0 NCU replay with clock control disabled; duration and "
            "other time-derived metrics are not contention-realistic "
            "all-GPU-active measurements."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")
    write_readme(
        output_dir / "README.md",
        campaign,
        points,
        excluded,
        roofs,
        arm,
        display_label,
    )
    render_plot(
        points,
        roofs,
        output_dir,
        args.annotate_longest,
        arm,
        display_label,
    )

    print(
        f"plotted {len(points)} launches; excluded {len(excluded)}; "
        f"count launches={len(count_points)}"
    )
    print(
        f"instruction roof={roofs.instruction_roof_inst_per_cycle:.3f} "
        "inst/cycle; memory roof="
        f"{roofs.memory_roof_sector_per_sm_cycle:.3f} sectors/SM-cycle; "
        f"ridge={roofs.ridge_point_instruction_per_sector:.3f} inst/sector"
    )


if __name__ == "__main__":
    main()
