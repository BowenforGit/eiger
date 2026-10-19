#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Export one NCU kernel-occurrence fact sheet per distributed TPC-H arm."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import plot_distributed_tpch_instruction_roofline as common


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOF_MIN_DURATION_US = 100.0
ROOF_MIN_DRAM_PCT = 1.0

STALL_METRICS = (
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct",
    "smsp__warp_issue_stalled_drain_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_misc_per_warp_active.pct",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_sleeping_per_warp_active.pct",
    "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
)

METRIC_COLUMNS = {
    "launch__waves_per_multiprocessor": "waves_per_sm",
    "launch__thread_count": "thread_count",
    "launch__sm_count": "sm_count",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block_static": "shared_mem_static_bytes_per_block",
    "launch__shared_mem_per_block_dynamic": "shared_mem_dynamic_bytes_per_block",
    "launch__shared_mem_per_block_driver": "shared_mem_driver_bytes_per_block",
    "smsp__inst_executed.sum": "warp_instructions_executed",
    "smsp__thread_inst_executed.sum": "thread_instructions_executed",
    "smsp__thread_inst_executed_per_inst_executed.pct": (
        "thread_instructions_per_warp_instruction_pct"
    ),
    "smsp__warps_launched.sum": "warps_launched",
    "smsp__warps_active.avg": "warps_active_avg",
    "smsp__warps_eligible.avg": "warps_eligible_avg",
    "sm__cycles_elapsed.avg": "sm_cycles_elapsed_avg",
    "dram__cycles_elapsed.avg": "dram_cycles_elapsed_avg",
    "dram__sectors.sum": "dram_sectors_total",
    "dram__sectors_read.sum": "dram_sectors_read",
    "dram__sectors_write.sum": "dram_sectors_write",
    "dram__bytes.sum": "dram_bytes_total",
    "dram__bytes_read.sum": "dram_bytes_read",
    "dram__bytes_write.sum": "dram_bytes_write",
    "l1tex__t_bytes.sum": "l1_bytes_total",
    "l1tex__t_sectors.sum": "l1_sectors_total",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": ("l1_global_load_sectors"),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": ("l1_global_store_sectors"),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum": ("l1_global_atomic_sectors"),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum": ("l1_global_reduction_sectors"),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum": (
        "l1_global_ldgsts_cache_access_sectors"
    ),
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": "l1_local_load_sectors",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": "l1_local_store_sectors",
    "lts__t_bytes.sum": "l2_bytes_total",
    "lts__t_sectors.sum": "l2_sectors_total",
    "lts__t_sectors_op_read.sum": "l2_read_sectors",
    "lts__t_sectors_op_write.sum": "l2_write_sectors",
    "l1tex__t_sector_hit_rate.pct": "l1_sector_hit_rate_pct",
    "lts__t_sector_hit_rate.pct": "l2_sector_hit_rate_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": ("compute_throughput_pct"),
    "l1tex__throughput.avg.pct_of_peak_sustained_active": ("l1_throughput_pct"),
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_throughput_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": ("dram_throughput_pct"),
}
for stall_metric in STALL_METRICS:
    reason = stall_metric.removeprefix("smsp__warp_issue_stalled_").removesuffix(
        "_per_warp_active.pct"
    )
    METRIC_COLUMNS[stall_metric] = f"stall_{reason}_pct"

REQUIRED_METRICS = {
    "launch__waves_per_multiprocessor": "",
    "launch__thread_count": "thread",
    "launch__sm_count": "SM",
    "launch__registers_per_thread": "register/thread",
    "launch__shared_mem_per_block_static": "byte/block",
    "launch__shared_mem_per_block_dynamic": "byte/block",
    "launch__shared_mem_per_block_driver": "byte/block",
    "smsp__inst_executed.sum": "inst",
    "smsp__thread_inst_executed.sum": "inst",
    "smsp__thread_inst_executed_per_inst_executed.pct": "%",
    "smsp__warps_launched.sum": "warp",
    "smsp__warps_active.avg": "warp",
    "smsp__warps_eligible.avg": "warp",
    "sm__cycles_elapsed.avg": "cycle",
    "dram__cycles_elapsed.avg": "cycle",
    "dram__sectors.sum": "sector",
    "dram__sectors_read.sum": "sector",
    "dram__sectors_write.sum": "sector",
    "dram__bytes.sum": "byte",
    "dram__bytes_read.sum": "byte",
    "dram__bytes_write.sum": "byte",
    "dram__throughput.sum.pct_of_peak_sustained_elapsed": "%",
    "l1tex__t_bytes.sum": "byte",
    "l1tex__t_sectors.sum": "sector",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "sector",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "sector",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum": "sector",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum": "sector",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum": "sector",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": "sector",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": "sector",
    "lts__t_bytes.sum": "byte",
    "lts__t_sectors.sum": "sector",
    "lts__t_sectors_op_read.sum": "sector",
    "lts__t_sectors_op_write.sum": "sector",
    "l1tex__t_sector_hit_rate.pct": "%",
    "lts__t_sector_hit_rate.pct": "%",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "l1tex__throughput.avg.pct_of_peak_sustained_active": "%",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "gpu__time_duration.sum": "ns",
    **{metric: "%" for metric in STALL_METRICS},
}

INTEGER_METRICS = {
    metric
    for metric, unit in REQUIRED_METRICS.items()
    if unit in {"byte", "byte/block", "sector", "inst", "thread", "SM"}
    and not metric.endswith(".avg")
}

PROVENANCE_FIELDS = [
    "occurrence_id",
    "campaign",
    "arm",
    "arm_label",
    "scale_factor",
    "topology",
    "node_count",
    "ranks",
    "gpus_per_node",
    "gpu_count",
    "hosts",
    "gpu_model",
    "compute_capability",
    "ncu_version",
    "ncu_capture_host",
    "capture_scope",
    "query",
    "step_index",
    "step_label",
    "phase_category",
    "phase_name",
    "nvtx_context",
    "attempt",
    "source_csv",
]

IDENTITY_FIELDS = [
    "ncu_launch_id",
    "kernel_family",
    "kernel_name",
    "target_kernel_match",
    "target_match_reason",
    "process_id",
    "process_name",
    "profile_host",
    "device",
    "context",
    "stream",
    "block_size",
    "grid_size",
]

DERIVED_FIELDS = [
    "kernel_duration_ms",
    "instruction_intensity_inst_per_dram_sector",
    "instruction_throughput_inst_per_sm_cycle",
    "achieved_dram_sectors_per_sm_cycle",
    "instruction_roof_inst_per_cycle",
    "memory_roof_dram_sectors_per_sm_cycle",
    "ridge_instruction_intensity",
    "bound_verdict",
    "fraction_of_active_roof",
    "dominant_memory_level",
    "dominant_memory_throughput_pct",
    "leading_stall_reason",
    "leading_stall_pct",
    "derivation_status",
]

FACT_SHEET_FIELDS = (
    PROVENANCE_FIELDS + IDENTITY_FIELDS + list(METRIC_COLUMNS.values()) + DERIVED_FIELDS
)


@dataclass
class FactLaunch:
    query: int
    launch_id: int
    attempt: Path
    source_csv: Path
    kernel_name: str
    process_id: str
    process_name: str
    profile_host: str
    device: str
    compute_capability: str
    context: str
    stream: str
    block_size: str
    grid_size: str
    nvtx_context: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class CampaignMetadata:
    campaign: Path
    arm: str
    arm_label: str
    scale_factor: int
    topology: str
    node_count: int
    ranks: int
    gpus_per_node: int
    gpu_count: int
    hosts: str
    gpu_model: str
    ncu_version: str
    ncu_capture_host: str


@dataclass(frozen=True)
class Roofs:
    sm_count: int
    instruction_roof: float
    memory_roof: float
    ridge: float
    calibration_launches: int
    memory_roof_relative_mad: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        action="append",
        type=Path,
        required=True,
        help="NCU campaign root; repeat to export multiple arms",
    )
    return parser.parse_args()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_version(text: str) -> str:
    match = re.search(r"^Version\s+([^\s]+)", text, flags=re.MULTILINE)
    return match.group(1) if match else text.strip().replace("\n", " ")


def campaign_metadata(campaign: Path, arm: str) -> CampaignMetadata:
    manifest = json.loads((campaign / "manifest" / "campaign.json").read_text())
    preflight = json.loads((campaign / "manifest" / "preflight.json").read_text())
    profile = json.loads((campaign / "manifest" / "ncu_profile.json").read_text())
    configuration = manifest["configuration"]
    topology = configuration["topology"]
    host_entries = topology["hosts"]
    local_platform = preflight["local_platform"]
    first_gpu = local_platform["gpu_inventory"].splitlines()[0].split(",")
    gpu_model = first_gpu[3].strip()
    ncu_capture_host = profile.get("selected", {}).get("host", "")
    return CampaignMetadata(
        campaign=campaign,
        arm=arm,
        arm_label=common.display_label_for_arm(arm),
        scale_factor=int(configuration["scale"]),
        topology=topology["label"],
        node_count=len(host_entries),
        ranks=int(topology["ranks"]),
        gpus_per_node=int(topology["gpus_per_node"]),
        gpu_count=sum(int(host["slots"]) for host in host_entries),
        hosts=",".join(host["name"] for host in host_entries),
        gpu_model=gpu_model,
        ncu_version=parse_version(local_platform["ncu_version"]),
        ncu_capture_host=ncu_capture_host,
    )


def extract_launches(selected: common.SelectedAttempt) -> list[FactLaunch]:
    launches: dict[int, FactLaunch] = {}
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

        for csv_row in reader:
            launch_id = int(csv_row["ID"])
            launch = launches.get(launch_id)
            if launch is None:
                launch = FactLaunch(
                    query=selected.query,
                    launch_id=launch_id,
                    attempt=selected.attempt,
                    source_csv=selected.metrics_csv,
                    kernel_name=csv_row["Kernel Name"],
                    process_id=csv_row["Process ID"],
                    process_name=csv_row["Process Name"],
                    profile_host=csv_row["Host Name"],
                    device=csv_row["Device"],
                    compute_capability=csv_row["CC"],
                    context=csv_row["Context"],
                    stream=csv_row["Stream"],
                    block_size=csv_row["Block Size"],
                    grid_size=csv_row["Grid Size"],
                    nvtx_context=csv_row[nvtx_field],
                    metrics={},
                )
                launches[launch_id] = launch
            elif launch.kernel_name != csv_row["Kernel Name"]:
                raise ValueError(
                    f"{selected.metrics_csv}: launch ID {launch_id} has "
                    "multiple kernel names"
                )

            metric = csv_row["Metric Name"]
            if metric not in REQUIRED_METRICS:
                continue
            expected_unit = REQUIRED_METRICS[metric]
            if csv_row["Metric Unit"] != expected_unit:
                raise ValueError(
                    f"{selected.metrics_csv}: {metric} unit is "
                    f"{csv_row['Metric Unit']!r}; expected {expected_unit!r}"
                )
            if metric in launch.metrics:
                raise ValueError(
                    f"{selected.metrics_csv}: duplicate {metric} for launch "
                    f"ID {launch_id}"
                )
            launch.metrics[metric] = common.metric_value(
                csv_row["Metric Value"], metric, selected.metrics_csv
            )

    if not launches:
        raise ValueError(f"no launches found in {selected.metrics_csv}")
    for launch in launches.values():
        missing = sorted(set(REQUIRED_METRICS) - set(launch.metrics))
        if missing:
            raise ValueError(
                f"{selected.metrics_csv}: launch ID {launch.launch_id} is "
                f"missing metrics {missing}"
            )
    return [launches[index] for index in sorted(launches)]


def target_classification(kernel_name: str) -> tuple[str, bool, str]:
    family, reason = common.classify_kernel(kernel_name)
    if family is None:
        return "", False, str(reason)
    return family, True, "target_kernel"


def valid_roof_values(launch: FactLaunch) -> bool:
    return (
        launch.metrics["smsp__inst_executed.sum"] > 0
        and launch.metrics["dram__sectors.sum"] > 0
        and launch.metrics["sm__cycles_elapsed.avg"] > 0
        and launch.metrics["gpu__time_duration.sum"] > 0
    )


def derive_roofs(launches: list[FactLaunch]) -> Roofs:
    target_launches = [
        launch
        for launch in launches
        if target_classification(launch.kernel_name)[1] and valid_roof_values(launch)
    ]
    if not target_launches:
        raise ValueError("no valid target launches for roof derivation")
    sm_counts = {
        int(round(launch.metrics["launch__sm_count"])) for launch in target_launches
    }
    if len(sm_counts) != 1:
        raise ValueError(f"inconsistent SM counts: {sorted(sm_counts)}")
    sm_count = sm_counts.pop()
    calibration = []
    for launch in target_launches:
        duration_us = launch.metrics["gpu__time_duration.sum"] / 1000.0
        dram_pct = launch.metrics["dram__throughput.sum.pct_of_peak_sustained_elapsed"]
        if duration_us < ROOF_MIN_DURATION_US or dram_pct < ROOF_MIN_DRAM_PCT:
            continue
        achieved_rate = (
            launch.metrics["dram__sectors.sum"]
            / launch.metrics["sm__cycles_elapsed.avg"]
        )
        calibration.append(achieved_rate / (dram_pct / 100.0))
    if len(calibration) < 5:
        raise ValueError("fewer than five memory-roof calibration launches")
    memory_roof = statistics.median(calibration)
    relative_mad = (
        statistics.median(abs(value - memory_roof) for value in calibration)
        / memory_roof
    )
    if relative_mad > 0.10:
        raise ValueError(
            f"unstable memory-roof calibration: relative MAD={relative_mad:.4f}"
        )
    instruction_roof = common.SMSPS_PER_SM * sm_count
    return Roofs(
        sm_count=sm_count,
        instruction_roof=instruction_roof,
        memory_roof=memory_roof,
        ridge=instruction_roof / memory_roof,
        calibration_launches=len(calibration),
        memory_roof_relative_mad=relative_mad,
    )


def metric_output_value(metric: str, value: float) -> int | float:
    if metric in INTEGER_METRICS and math.isclose(value, round(value), abs_tol=1e-9):
        return int(round(value))
    return value


def compact_float(value: float) -> float:
    return float(f"{value:.12g}")


def leading_stall(metrics: dict[str, float]) -> tuple[str, float]:
    metric = max(STALL_METRICS, key=lambda name: metrics[name])
    reason = metric.removeprefix("smsp__warp_issue_stalled_").removesuffix(
        "_per_warp_active.pct"
    )
    return reason, metrics[metric]


def build_row(
    launch: FactLaunch,
    metadata: CampaignMetadata,
    roofs: Roofs,
) -> dict[str, object]:
    family, target_match, target_reason = target_classification(launch.kernel_name)
    step_index, step_label, phase_category, phase_name = common.parse_nvtx_context(
        launch.nvtx_context
    )
    attempt = relative_or_absolute(launch.attempt)
    source_csv = relative_or_absolute(launch.source_csv)
    row: dict[str, object] = {
        "occurrence_id": (
            f"sf{metadata.scale_factor}_{metadata.gpu_count}gpu_{metadata.arm}_"
            f"q{launch.query:02d}_{launch.attempt.name}_ncu{launch.launch_id}"
        ),
        "campaign": relative_or_absolute(metadata.campaign),
        "arm": metadata.arm,
        "arm_label": metadata.arm_label,
        "scale_factor": metadata.scale_factor,
        "topology": metadata.topology,
        "node_count": metadata.node_count,
        "ranks": metadata.ranks,
        "gpus_per_node": metadata.gpus_per_node,
        "gpu_count": metadata.gpu_count,
        "hosts": metadata.hosts,
        "gpu_model": metadata.gpu_model,
        "compute_capability": launch.compute_capability,
        "ncu_version": metadata.ncu_version,
        "ncu_capture_host": metadata.ncu_capture_host,
        "capture_scope": "rank0_ncu_kernel_replay_clock_control_none",
        "query": launch.query,
        "step_index": step_index,
        "step_label": step_label,
        "phase_category": phase_category,
        "phase_name": phase_name,
        "nvtx_context": launch.nvtx_context,
        "attempt": attempt,
        "source_csv": source_csv,
        "ncu_launch_id": launch.launch_id,
        "kernel_family": family,
        "kernel_name": launch.kernel_name,
        "target_kernel_match": target_match,
        "target_match_reason": target_reason,
        "process_id": launch.process_id,
        "process_name": launch.process_name,
        "profile_host": launch.profile_host,
        "device": launch.device,
        "context": launch.context,
        "stream": launch.stream,
        "block_size": launch.block_size,
        "grid_size": launch.grid_size,
    }
    for metric, column in METRIC_COLUMNS.items():
        row[column] = metric_output_value(metric, launch.metrics[metric])

    duration_ns = launch.metrics["gpu__time_duration.sum"]
    instructions = launch.metrics["smsp__inst_executed.sum"]
    dram_sectors = launch.metrics["dram__sectors.sum"]
    sm_cycles = launch.metrics["sm__cycles_elapsed.avg"]
    row["kernel_duration_ms"] = round(duration_ns / 1e6, 9)
    if instructions <= 0 or dram_sectors <= 0 or sm_cycles <= 0:
        invalid = [
            name
            for name, value in (
                ("warp_instructions", instructions),
                ("dram_sectors", dram_sectors),
                ("sm_cycles", sm_cycles),
            )
            if value <= 0
        ]
        row.update(
            instruction_intensity_inst_per_dram_sector="",
            instruction_throughput_inst_per_sm_cycle="",
            achieved_dram_sectors_per_sm_cycle="",
            instruction_roof_inst_per_cycle=compact_float(roofs.instruction_roof),
            memory_roof_dram_sectors_per_sm_cycle=compact_float(roofs.memory_roof),
            ridge_instruction_intensity=compact_float(roofs.ridge),
            bound_verdict="unclassified",
            fraction_of_active_roof="",
            derivation_status="nonpositive_" + "_".join(invalid),
        )
    else:
        intensity = instructions / dram_sectors
        throughput = instructions / sm_cycles
        achieved_dram_rate = dram_sectors / sm_cycles
        active_roof = min(
            roofs.memory_roof * intensity,
            roofs.instruction_roof,
        )
        row.update(
            instruction_intensity_inst_per_dram_sector=compact_float(intensity),
            instruction_throughput_inst_per_sm_cycle=compact_float(throughput),
            achieved_dram_sectors_per_sm_cycle=compact_float(achieved_dram_rate),
            instruction_roof_inst_per_cycle=compact_float(roofs.instruction_roof),
            memory_roof_dram_sectors_per_sm_cycle=compact_float(roofs.memory_roof),
            ridge_instruction_intensity=compact_float(roofs.ridge),
            bound_verdict="memory" if intensity < roofs.ridge else "compute",
            fraction_of_active_roof=compact_float(throughput / active_roof),
            derivation_status="valid",
        )

    memory_throughputs = {
        "dram": launch.metrics[
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
        ],
        "l2": launch.metrics["lts__throughput.avg.pct_of_peak_sustained_elapsed"],
        "l1": launch.metrics["l1tex__throughput.avg.pct_of_peak_sustained_active"],
    }
    dominant_level = max(("dram", "l2", "l1"), key=memory_throughputs.get)
    row["dominant_memory_level"] = dominant_level
    row["dominant_memory_throughput_pct"] = memory_throughputs[dominant_level]
    stall_reason, stall_pct = leading_stall(launch.metrics)
    row["leading_stall_reason"] = stall_reason
    row["leading_stall_pct"] = stall_pct
    if set(row) != set(FACT_SHEET_FIELDS):
        missing = sorted(set(FACT_SHEET_FIELDS) - set(row))
        extra = sorted(set(row) - set(FACT_SHEET_FIELDS))
        raise ValueError(
            f"fact-sheet row schema mismatch: missing={missing}, extra={extra}"
        )
    return row


def validate_rows(
    launches: list[FactLaunch],
    rows: list[dict[str, object]],
    metadata: CampaignMetadata,
) -> dict[str, object]:
    if len(rows) != len(launches):
        raise ValueError("fact-sheet row count does not match raw launch count")
    occurrence_ids = [str(row["occurrence_id"]) for row in rows]
    if len(set(occurrence_ids)) != len(occurrence_ids):
        raise ValueError("duplicate occurrence IDs")
    if {int(row["query"]) for row in rows} != set(common.QUERIES):
        raise ValueError("fact sheet does not cover Q1-Q22")

    sector_mismatches = 0
    byte_mismatches = 0
    for row in rows:
        if not math.isclose(
            float(row["dram_sectors_total"]),
            float(row["dram_sectors_read"]) + float(row["dram_sectors_write"]),
            rel_tol=1e-9,
            abs_tol=1.0,
        ):
            sector_mismatches += 1
        if not math.isclose(
            float(row["dram_bytes_total"]),
            float(row["dram_bytes_read"]) + float(row["dram_bytes_write"]),
            rel_tol=1e-9,
            abs_tol=1.0,
        ):
            byte_mismatches += 1
    if sector_mismatches or byte_mismatches:
        raise ValueError(
            "DRAM total/component validation failed: "
            f"sector_mismatches={sector_mismatches}, byte_mismatches={byte_mismatches}"
        )

    target_count_rows = [row for row in rows if row["kernel_family"] == "count"]
    if not target_count_rows:
        raise ValueError("no target count kernels in fact sheet")
    if any(
        common.DESIRED_COUNT_SYMBOL not in str(row["kernel_name"])
        for row in target_count_rows
    ):
        raise ValueError("non-target count symbol classified as target count")
    return {
        "row_count": len(rows),
        "unique_occurrence_id_count": len(set(occurrence_ids)),
        "query_count": len({int(row["query"]) for row in rows}),
        "valid_derivation_count": sum(
            row["derivation_status"] == "valid" for row in rows
        ),
        "target_kernel_count": sum(bool(row["target_kernel_match"]) for row in rows),
        "excluded_capture_count": sum(
            not bool(row["target_kernel_match"]) for row in rows
        ),
        "target_count_kernel_count": len(target_count_rows),
        "dram_sector_total_mismatches": sector_mismatches,
        "dram_byte_total_mismatches": byte_mismatches,
        "scale_factor": metadata.scale_factor,
        "gpu_count": metadata.gpu_count,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FACT_SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def write_readme(
    path: Path,
    metadata: CampaignMetadata,
    roofs: Roofs,
    row_count: int,
) -> None:
    metric_lines = "\n".join(
        f"- `{column}` <- `{metric}` ({REQUIRED_METRICS[metric] or 'unitless'})"
        for metric, column in METRIC_COLUMNS.items()
    )
    campaign = relative_or_absolute(metadata.campaign)
    text = f"""# NCU kernel fact sheet: {metadata.arm_label}

This directory contains one row for every NCU-profiled launch in the passing
Q1-Q22 attempts of `{campaign}`. Rows are not dropped when the capture regex
matched an unintended kernel; inspect `target_kernel_match` and
`target_match_reason`.

## Inventory

- Rows: {row_count}
- SF: {metadata.scale_factor}
- Setup: {metadata.topology} ({metadata.gpu_count} {metadata.gpu_model} GPUs)
- Capture: rank 0, NCU kernel replay, clock control disabled
- Instruction roof: {roofs.instruction_roof:.12g} instructions/cycle
- DRAM roof: {roofs.memory_roof:.12g} sectors/SM cycle
- Ridge: {roofs.ridge:.12g} instructions/sector

## Roofline formulas

- Instruction intensity = `smsp__inst_executed.sum / dram__sectors.sum`
- Instruction throughput = `smsp__inst_executed.sum / sm__cycles_elapsed.avg`
- Bound verdict = memory when intensity < ridge, otherwise compute
- The memory roof is the median of
  `(dram__sectors.sum / sm__cycles_elapsed.avg) /
  (dram__throughput.sum.pct_of_peak_sustained_elapsed / 100)` over target
  launches lasting at least {ROOF_MIN_DURATION_US:g} us with at least
  {ROOF_MIN_DRAM_PCT:g}% DRAM throughput.

## Direct NCU metric columns

{metric_lines}

`kernel_duration_ms` is the only duration representation and is converted from
`gpu__time_duration.sum`. Traffic totals and their components are intentionally
retained together because a total is not derivable from either component alone.

## Capture caveat

Duration, throughput percentages, and stalls are time-derived from isolated
rank-0 NCU replay and are not contention-realistic all-GPU-active measurements.
Byte and instruction counts are more robust, although replayed mutating kernels
can still make data-movement counters indicative rather than exact.
"""
    path.write_text(text)
    print(f"wrote {path}")


def export_campaign(campaign: Path) -> tuple[Path, dict[str, object]]:
    campaign = campaign.resolve()
    if not campaign.is_dir():
        raise FileNotFoundError(f"campaign directory does not exist: {campaign}")
    arm = common.infer_arm(campaign)
    metadata = campaign_metadata(campaign, arm)
    attempts = common.select_attempts(campaign, arm)
    launches: list[FactLaunch] = []
    launches_by_query: dict[str, int] = {}
    for selected in attempts:
        query_launches = extract_launches(selected)
        launches.extend(query_launches)
        launches_by_query[f"Q{selected.query:02d}"] = len(query_launches)
        print(f"{arm} Q{selected.query:02d}: {len(query_launches)} launches")

    roofs = derive_roofs(launches)
    rows = [build_row(launch, metadata, roofs) for launch in launches]
    rows.sort(key=lambda row: (int(row["query"]), int(row["ncu_launch_id"])))
    validation = validate_rows(launches, rows, metadata)
    output_dir = campaign / "analysis" / "kernel_fact_sheet"
    csv_path = output_dir / f"kernel_fact_sheet_{arm}.csv"
    summary_path = output_dir / f"kernel_fact_sheet_{arm}_summary.json"
    write_csv(csv_path, rows)

    summary = {
        "campaign": str(campaign),
        "arm": arm,
        "arm_label": metadata.arm_label,
        "launches_by_query": launches_by_query,
        "kernel_family_counts": dict(
            sorted(Counter(str(row["kernel_family"]) for row in rows).items())
        ),
        "target_match_reason_counts": dict(
            sorted(Counter(str(row["target_match_reason"]) for row in rows).items())
        ),
        "bound_verdict_counts": dict(
            sorted(Counter(str(row["bound_verdict"]) for row in rows).items())
        ),
        "dominant_memory_level_counts": dict(
            sorted(Counter(str(row["dominant_memory_level"]) for row in rows).items())
        ),
        "roof_parameters": asdict(roofs),
        "validation": validation,
        "duration_metric": {
            "source": "gpu__time_duration.sum",
            "output_column": "kernel_duration_ms",
            "output_unit": "ms",
        },
        "capture_caveat": (
            "Rank-0 NCU kernel replay with clock control disabled; duration, "
            "throughput percentages, and stalls are isolated time-derived values."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")
    write_readme(output_dir / "README.md", metadata, roofs, len(rows))
    print(
        f"{arm}: exported {len(rows)} rows; "
        f"target={validation['target_kernel_count']}; "
        f"excluded={validation['excluded_capture_count']}"
    )
    return csv_path, summary


def main() -> None:
    args = parse_args()
    campaigns = [campaign.resolve() for campaign in args.campaign]
    if len(set(campaigns)) != len(campaigns):
        raise ValueError("the same campaign was passed more than once")
    for campaign in campaigns:
        export_campaign(campaign)


if __name__ == "__main__":
    main()
