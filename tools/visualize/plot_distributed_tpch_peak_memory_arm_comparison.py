#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Compare worst-case per-rank peak memory for baseline and optimized arms."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_distributed_tpch_peak_memory as memory_base


REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES = tuple(range(1, 23))
GIB = 1024**3
ARMS = (
    ("legacy_cta64", "Baseline", "#4C78A8"),
    ("synthesized_cta64", "Opt. comp", "#54A24B"),
)


@dataclass(frozen=True)
class Setting:
    scale_factor: int
    gpus: int
    arm: str
    label: str
    color: str

    @property
    def key(self) -> str:
        return f"sf{self.scale_factor}_{self.gpus}gpu_{self.arm}"


SETTINGS = tuple(
    Setting(scale_factor, gpus, arm, label, color)
    for scale_factor, gpus in ((1000, 8), (3000, 24))
    for arm, label, color in ARMS
)


@dataclass(frozen=True)
class SelectedMemory:
    setting: Setting
    query: int
    csv_path: Path
    result_status: str
    failure_kind: str


@dataclass(frozen=True)
class QueryPeak:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    query: int
    peak_rank: int
    total_live_peak_bytes: int
    total_live_peak_gib: float
    resident_current_bytes_at_peak_rank: int
    query_peak_bytes_at_peak_rank: int
    result_status: str
    failure_kind: str
    memory_csv: str


@dataclass(frozen=True)
class CampaignPeak:
    setting_key: str
    scale_factor: int
    gpus: int
    arm: str
    arm_label: str
    peak_query: int
    peak_rank: int
    total_live_peak_bytes: int
    total_live_peak_gib: float
    resident_current_gib_at_peak_rank: float
    query_peak_gib_at_peak_rank: float
    query_count: int
    min_query_peak_gib: float
    q1_query_peak_gib: float
    median_query_peak_gib: float
    mean_query_peak_gib: float
    q3_query_peak_gib: float


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


def query_root(results_dir: Path, setting: Setting, query: int) -> Path:
    return (
        results_dir
        / f"sf{setting.scale_factor}"
        / setting.arm
        / f"{setting.gpus}gpu_128p_memory"
        / "arms"
        / setting.arm
        / "memory"
        / f"q{query:02d}"
    )


def select_memory(
    results_dir: Path, setting: Setting, query: int
) -> SelectedMemory:
    root = query_root(results_dir, setting, query)
    attempts = sorted(
        root.glob("attempt-*"),
        key=memory_base.attempt_number,
        reverse=True,
    )
    for attempt in attempts:
        result_path = attempt / "result.json"
        memory_path = attempt / "memory.csv"
        if not result_path.is_file() or not memory_path.is_file():
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
        if not accepted or memory_path.stat().st_size == 0:
            continue
        return SelectedMemory(
            setting=setting,
            query=query,
            csv_path=memory_path,
            result_status=str(result["status"]),
            failure_kind=str(result.get("failure_kind") or ""),
        )
    raise ValueError(f"no usable memory.csv attempt under {root}")


def load_query_peak(
    selected: SelectedMemory, results_dir: Path
) -> QueryPeak:
    required = {
        "query",
        "scale",
        "ranks",
        "record_type",
        "rank",
        "resident_current_bytes",
        "query_peak_bytes",
        "total_live_peak_bytes",
        "total_max_rank",
    }
    with selected.csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{selected.csv_path} is missing columns: "
                f"{', '.join(sorted(missing))}"
            )
        rows = list(reader)

    setting = selected.setting
    rank_rows = [row for row in rows if row["record_type"] == "rank"]
    aggregate_rows = [row for row in rows if row["record_type"] == "aggregate"]
    if len(rank_rows) != setting.gpus:
        raise ValueError(
            f"{selected.csv_path}: expected {setting.gpus} rank rows, "
            f"found {len(rank_rows)}"
        )
    if len(aggregate_rows) != 1:
        raise ValueError(
            f"{selected.csv_path}: expected one aggregate row, "
            f"found {len(aggregate_rows)}"
        )
    expected_ranks = set(range(setting.gpus))
    actual_ranks = {int(row["rank"]) for row in rank_rows}
    if actual_ranks != expected_ranks:
        raise ValueError(f"{selected.csv_path}: incomplete rank coverage")
    for row in rows:
        if int(row["scale"]) != setting.scale_factor:
            raise ValueError(f"{selected.csv_path}: scale-factor mismatch")
        if int(row["ranks"]) != setting.gpus:
            raise ValueError(f"{selected.csv_path}: rank-count mismatch")
        if row["query"] != f"Q{selected.query}":
            raise ValueError(f"{selected.csv_path}: query mismatch")

    by_rank = {int(row["rank"]): row for row in rank_rows}
    computed_peak_rank = max(
        by_rank, key=lambda rank: int(by_rank[rank]["total_live_peak_bytes"])
    )
    computed_peak_bytes = int(
        by_rank[computed_peak_rank]["total_live_peak_bytes"]
    )
    aggregate = aggregate_rows[0]
    aggregate_peak_rank = int(aggregate["total_max_rank"])
    aggregate_peak_bytes = int(aggregate["total_live_peak_bytes"])
    if (aggregate_peak_rank, aggregate_peak_bytes) != (
        computed_peak_rank,
        computed_peak_bytes,
    ):
        raise ValueError(
            f"{selected.csv_path}: aggregate and rank peaks disagree"
        )
    peak_row = by_rank[computed_peak_rank]
    resident_current = int(peak_row["resident_current_bytes"])
    query_peak = int(peak_row["query_peak_bytes"])
    if resident_current + query_peak != aggregate_peak_bytes:
        raise ValueError(f"{selected.csv_path}: peak components do not sum")

    return QueryPeak(
        setting_key=setting.key,
        scale_factor=setting.scale_factor,
        gpus=setting.gpus,
        arm=setting.arm,
        query=selected.query,
        peak_rank=computed_peak_rank,
        total_live_peak_bytes=aggregate_peak_bytes,
        total_live_peak_gib=aggregate_peak_bytes / GIB,
        resident_current_bytes_at_peak_rank=resident_current,
        query_peak_bytes_at_peak_rank=query_peak,
        result_status=selected.result_status,
        failure_kind=selected.failure_kind,
        memory_csv=str(selected.csv_path.relative_to(results_dir)),
    )


def summarize_campaign(setting: Setting, rows: list[QueryPeak]) -> CampaignPeak:
    if {row.query for row in rows} != set(QUERIES):
        raise ValueError(f"{setting.key}: incomplete Q1-Q22 coverage")
    peak = max(rows, key=lambda row: (row.total_live_peak_bytes, -row.query))
    values = np.asarray([row.total_live_peak_gib for row in rows], dtype=float)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return CampaignPeak(
        setting_key=setting.key,
        scale_factor=setting.scale_factor,
        gpus=setting.gpus,
        arm=setting.arm,
        arm_label=setting.label,
        peak_query=peak.query,
        peak_rank=peak.peak_rank,
        total_live_peak_bytes=peak.total_live_peak_bytes,
        total_live_peak_gib=peak.total_live_peak_gib,
        resident_current_gib_at_peak_rank=(
            peak.resident_current_bytes_at_peak_rank / GIB
        ),
        query_peak_gib_at_peak_rank=peak.query_peak_bytes_at_peak_rank / GIB,
        query_count=len(rows),
        min_query_peak_gib=float(np.min(values)),
        q1_query_peak_gib=float(q1),
        median_query_peak_gib=float(median),
        mean_query_peak_gib=float(np.mean(values)),
        q3_query_peak_gib=float(q3),
    )


def plot(
    query_rows: list[QueryPeak],
    campaigns: list[CampaignPeak],
    output_dir: Path,
) -> None:
    campaign_lookup = {
        (row.scale_factor, row.gpus, row.arm): row for row in campaigns
    }
    sample_lookup = {
        (scale_factor, gpus, arm): [
            row.total_live_peak_gib
            for row in query_rows
            if row.scale_factor == scale_factor
            and row.gpus == gpus
            and row.arm == arm
        ]
        for scale_factor, gpus in ((1000, 8), (3000, 24))
        for arm, _, _ in ARMS
    }
    for key, values in sample_lookup.items():
        if len(values) != len(QUERIES):
            raise ValueError(f"{key}: expected 22 query peaks, found {len(values)}")
        campaign = campaign_lookup[key]
        if not np.isclose(np.mean(values), campaign.mean_query_peak_gib):
            raise ValueError(f"{key}: query samples and summary mean disagree")

    group_centers = {(1000, 8): 0.0, (3000, 24): 0.72}
    arm_offsets = {"legacy_cta64": -0.11, "synthesized_cta64": 0.11}
    violin_width = 0.205

    fig, ax = memory_base.plt.subplots(figsize=(4.8, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for scale_factor, gpus in ((1000, 8), (3000, 24)):
        center = group_centers[(scale_factor, gpus)]
        for arm, label, color in ARMS:
            position = center + arm_offsets[arm]
            values = np.asarray(
                sample_lookup[(scale_factor, gpus, arm)], dtype=float
            )
            parts = ax.violinplot(
                [values],
                positions=[position],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                points=160,
                bw_method="scott",
            )
            body = parts["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor("#555555")
            body.set_linewidth(0.65)
            body.set_alpha(0.86)

            q1, median, q3 = np.percentile(values, [25, 50, 75])
            mean = float(np.mean(values))
            ax.vlines(
                position,
                q1,
                q3,
                color="#303030",
                linewidth=2.2,
                zorder=5,
            )
            ax.scatter(
                [position],
                [median],
                marker="o",
                s=27,
                facecolor="white",
                edgecolor="#303030",
                linewidth=0.8,
                zorder=6,
            )
            ax.scatter(
                [position],
                [mean],
                marker="D",
                s=24,
                facecolor="#303030",
                edgecolor="white",
                linewidth=0.6,
                zorder=7,
            )
            ax.annotate(
                f"{mean:.1f}",
                xy=(position, mean),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#333333",
                zorder=8,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
            )

    maximum = max(row.total_live_peak_gib for row in query_rows)
    ax.set_ylim(0.0, maximum * 1.10)
    ax.set_xticks(
        [group_centers[(1000, 8)], group_centers[(3000, 24)]],
        ["SF=1000\n8 GPUs", "SF=3000\n24 GPUs"],
    )
    ax.set_ylabel("Per-query peak memory\non any rank (GiB)")
    ax.set_title("TPC-H Peak GPU Memory Usage", pad=50)
    ax.grid(
        axis="y",
        color="#D8D8D8",
        linewidth=0.8,
        linestyle=(0, (2, 3)),
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="both", colors="#333333")

    arm_handles = [
        Patch(facecolor=color, label=label) for _, label, color in ARMS
    ]
    mean_handle = Line2D(
        [],
        [],
        marker="D",
        linestyle="None",
        markersize=6,
        markerfacecolor="#303030",
        markeredgecolor="white",
        label="Mean",
    )
    median_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="#303030",
        label="Median",
    )
    iqr_handle = Line2D(
        [], [], color="#303030", linewidth=2.2, label="IQR"
    )
    blank_handle = Line2D([], [], linestyle="None", label="")
    # Three-column legends fill column-first.  This renders arms on the first
    # row and the mean/median/IQR key on the second.
    handles = [
        arm_handles[0],
        mean_handle,
        arm_handles[1],
        median_handle,
        blank_handle,
        iqr_handle,
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.55,
    )
    fig.subplots_adjust(left=0.22, right=0.99, bottom=0.18, top=0.76)

    stem = "distributed_tpch_peak_memory_arm_comparison"
    for suffix in ("pdf", "png"):
        directory = output_dir / suffix
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    memory_base.plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    selected = [
        select_memory(results_dir, setting, query)
        for setting in SETTINGS
        for query in QUERIES
    ]
    expected = len(SETTINGS) * len(QUERIES)
    if len(selected) != expected:
        raise ValueError(
            f"expected {expected} memory artifacts, found {len(selected)}"
        )
    query_rows = [load_query_peak(item, results_dir) for item in selected]
    campaigns = [
        summarize_campaign(
            setting,
            [row for row in query_rows if row.setting_key == setting.key],
        )
        for setting in SETTINGS
    ]
    memory_base.write_csv(
        output_dir / "distributed_tpch_peak_memory_arm_comparison_by_query.csv",
        [asdict(row) for row in query_rows],
    )
    memory_base.write_csv(
        output_dir / "distributed_tpch_peak_memory_arm_comparison_summary.csv",
        [asdict(row) for row in campaigns],
    )
    plot(query_rows, campaigns, output_dir)


if __name__ == "__main__":
    main()
