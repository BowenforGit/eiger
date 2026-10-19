#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot worst-case per-rank peak memory for distributed TPC-H campaigns."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    sys.exit(f"error: matplotlib is required: {error}")


REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES = tuple(range(1, 23))
GIB = 1024**3


@dataclass(frozen=True)
class Setting:
    scale_factor: int
    gpus: int

    @property
    def key(self) -> str:
        return f"sf{self.scale_factor}_{self.gpus}gpu_legacy_cta64"


SETTINGS = (
    Setting(1000, 8),
    Setting(1000, 16),
    Setting(1000, 24),
    Setting(1000, 32),
    Setting(3000, 24),
    Setting(3000, 32),
)

COLORS = {1000: "#4C78A8", 3000: "#F28E2B"}


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
    peak_query: int
    peak_rank: int
    total_live_peak_bytes: int
    total_live_peak_gib: float
    resident_current_gib_at_peak_rank: float
    query_peak_gib_at_peak_rank: float


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


def attempt_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("attempt-"))
    except ValueError as error:
        raise ValueError(f"invalid attempt directory: {path}") from error


def query_root(results_dir: Path, setting: Setting, query: int) -> Path:
    return (
        results_dir
        / f"sf{setting.scale_factor}"
        / "legacy_cta64"
        / f"{setting.gpus}gpu_128p_memory"
        / "arms"
        / "legacy_cta64"
        / "memory"
        / f"q{query:02d}"
    )


def select_memory(
    results_dir: Path, setting: Setting, query: int
) -> SelectedMemory:
    root = query_root(results_dir, setting, query)
    attempts = sorted(root.glob("attempt-*"), key=attempt_number, reverse=True)
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

    rank_rows = [row for row in rows if row["record_type"] == "rank"]
    aggregate_rows = [row for row in rows if row["record_type"] == "aggregate"]
    if len(rank_rows) != selected.setting.gpus:
        raise ValueError(
            f"{selected.csv_path}: expected {selected.setting.gpus} rank rows, "
            f"found {len(rank_rows)}"
        )
    if len(aggregate_rows) != 1:
        raise ValueError(
            f"{selected.csv_path}: expected one aggregate row, "
            f"found {len(aggregate_rows)}"
        )
    expected_ranks = set(range(selected.setting.gpus))
    actual_ranks = {int(row["rank"]) for row in rank_rows}
    if actual_ranks != expected_ranks:
        raise ValueError(f"{selected.csv_path}: incomplete rank coverage")
    for row in rows:
        if int(row["scale"]) != selected.setting.scale_factor:
            raise ValueError(f"{selected.csv_path}: scale-factor mismatch")
        if int(row["ranks"]) != selected.setting.gpus:
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
        setting_key=selected.setting.key,
        scale_factor=selected.setting.scale_factor,
        gpus=selected.setting.gpus,
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
    return CampaignPeak(
        setting_key=setting.key,
        scale_factor=setting.scale_factor,
        gpus=setting.gpus,
        peak_query=peak.query,
        peak_rank=peak.peak_rank,
        total_live_peak_bytes=peak.total_live_peak_bytes,
        total_live_peak_gib=peak.total_live_peak_gib,
        resident_current_gib_at_peak_rank=(
            peak.resident_current_bytes_at_peak_rank / GIB
        ),
        query_peak_gib_at_peak_rank=peak.query_peak_bytes_at_peak_rank / GIB,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot(campaigns: list[CampaignPeak], output_dir: Path) -> None:
    positions = {
        (1000, 8): 0.0,
        (1000, 16): 1.0,
        (1000, 24): 2.0,
        (1000, 32): 3.0,
        (3000, 24): 5.0,
        (3000, 32): 6.0,
    }
    width = 0.68

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for row in campaigns:
        x = positions[(row.scale_factor, row.gpus)]
        bar = ax.bar(
            x,
            row.total_live_peak_gib,
            width=width,
            color=COLORS[row.scale_factor],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )[0]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row.total_live_peak_gib,
            f"{row.total_live_peak_gib:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="#333333",
        )

    maximum = max(row.total_live_peak_gib for row in campaigns)
    ax.set_ylim(0.0, maximum * 1.16)
    ax.set_xticks(
        [positions[(row.scale_factor, row.gpus)] for row in campaigns],
        [str(row.gpus) for row in campaigns],
    )
    ax.text(
        1.5,
        -0.13,
        "SF=1000",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=13,
        color="#444444",
    )
    ax.text(
        5.5,
        -0.13,
        "SF=3000",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=13,
        color="#444444",
    )
    ax.axvline(4.0, color="#E2E2E2", linewidth=1.0, zorder=1)
    ax.set_xlabel("Number of GPUs", labelpad=48)
    ax.set_ylabel("Peak memory on any rank (GiB)")
    ax.set_title("TPC-H Peak GPU Memory Usage", pad=13)
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
    fig.tight_layout()

    stem = "distributed_tpch_peak_memory_legacy_cta64"
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
    selected = [
        select_memory(results_dir, setting, query)
        for setting in SETTINGS
        for query in QUERIES
    ]
    if len(selected) != len(SETTINGS) * len(QUERIES):
        raise ValueError("unexpected selected memory-artifact count")
    query_rows = [load_query_peak(item, results_dir) for item in selected]
    campaigns = [
        summarize_campaign(
            setting,
            [row for row in query_rows if row.setting_key == setting.key],
        )
        for setting in SETTINGS
    ]
    write_csv(
        output_dir / "distributed_tpch_peak_memory_by_query.csv",
        [asdict(row) for row in query_rows],
    )
    write_csv(
        output_dir / "distributed_tpch_peak_memory_summary.csv",
        [asdict(row) for row in campaigns],
    )
    plot(campaigns, output_dir)


if __name__ == "__main__":
    main()
