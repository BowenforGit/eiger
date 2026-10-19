# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


VISUALIZE_DIR = Path(__file__).resolve().parents[1]
if str(VISUALIZE_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZE_DIR))
SCRIPT = VISUALIZE_DIR / "plot_distributed_tpch_compute_memory_throughput_by_family.py"
SPEC = importlib.util.spec_from_file_location(
    "compute_memory_throughput_by_family", SCRIPT
)
assert SPEC and SPEC.loader
PLOTTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOTTER
SPEC.loader.exec_module(PLOTTER)


def make_point(
    *,
    family: str = "count",
    launch_id: int = 1,
    compute: float = 60.0,
    dram: float = 30.0,
    l2: float = 40.0,
    l1: float = 50.0,
    dominant_level: str = "l1",
    duration_ms: float = 1.0,
):
    values = {"dram": dram, "l2": l2, "l1": l1}
    return PLOTTER.throughput_plot.ThroughputPoint(
        query=1,
        launch_id=launch_id,
        kernel_family=family,
        kernel_name=(
            "void open_addressing_ns::count<0>(T1)"
            if family == "count"
            else f"void {family}<0>(T1)"
        ),
        step_index="1",
        step_label="Project",
        phase_category="compute",
        phase_name="local.Project",
        process_id="123",
        device="0",
        context="1",
        stream="16",
        block_size="(128, 1, 1)",
        grid_size="(10, 1, 1)",
        compute_throughput_pct=compute,
        dram_throughput_pct=dram,
        l2_throughput_pct=l2,
        l1_throughput_pct=l1,
        memory_throughput_pct=values[dominant_level],
        dominant_memory_level=dominant_level,
        dominant_memory_metric=PLOTTER.throughput_plot.LEVEL_METRICS[dominant_level],
        dominant_level_tie_count=1,
        duration_ns=duration_ms * 1e6,
        duration_ms=duration_ms,
        source_csv="q01/profile.csv",
    )


def make_dataset(points, arm: str = "legacy_cta64"):
    return PLOTTER.throughput_plot.CampaignDataset(
        campaign=Path(f"/campaign/{arm}"),
        arm=arm,
        display_label=(
            "Baseline" if arm == "legacy_cta64" else "Optimized Computation"
        ),
        attempts=[],
        points=list(points),
        excluded=[],
        launches_read_by_query={},
    )


class ComputeMemoryThroughputByFamilyTests(unittest.TestCase):
    def test_expression_implementations_share_first_panel_slot(self) -> None:
        baseline = make_dataset([make_point(family="evaluate_expr_kernel")])
        optimized = make_dataset(
            [make_point(family="kernel_binary_op_vec")],
            arm="synthesized_cta64",
        )
        slot = PLOTTER.family_layout.PANEL_SLOTS[0]
        self.assertEqual(
            PLOTTER.family_for_slot(baseline, slot), "evaluate_expr_kernel"
        )
        self.assertEqual(
            PLOTTER.family_for_slot(optimized, slot), "kernel_binary_op_vec"
        )

    def test_family_summary_counts_dominant_memory_levels(self) -> None:
        dataset = make_dataset(
            [
                make_point(launch_id=1, dominant_level="dram", dram=80.0),
                make_point(launch_id=2, dominant_level="l1", l1=90.0),
                make_point(launch_id=3, dominant_level="l1", l1=95.0),
            ]
        )
        row = PLOTTER.summarize_family(dataset, "count", panel_slot=7)
        self.assertEqual(row["launch_count"], 3)
        self.assertEqual(row["dominant_dram_count"], 1)
        self.assertEqual(row["dominant_l2_count"], 0)
        self.assertEqual(row["dominant_l1_count"], 2)
        self.assertAlmostEqual(row["dominant_l1_percent"], 200.0 / 3.0)

    def test_axes_and_duration_normalization_are_shared(self) -> None:
        baseline = make_dataset([make_point(compute=10.0, l1=20.0, duration_ms=0.1)])
        optimized = make_dataset(
            [make_point(compute=85.0, l1=106.02, duration_ms=10.0)],
            arm="synthesized_cta64",
        )
        shared = PLOTTER.shared_plot_parameters([baseline, optimized])
        self.assertEqual(shared.axis_min_pct, 0.0)
        self.assertEqual(shared.axis_max_pct, 110.0)
        self.assertEqual(shared.major_tick_spacing_pct, 20.0)
        self.assertEqual(shared.duration_min_ms, 0.1)
        self.assertEqual(shared.duration_max_ms, 10.0)


if __name__ == "__main__":
    unittest.main()
