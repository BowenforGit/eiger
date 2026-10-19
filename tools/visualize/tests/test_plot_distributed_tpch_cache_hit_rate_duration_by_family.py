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
SCRIPT = VISUALIZE_DIR / "plot_distributed_tpch_cache_hit_rate_duration_by_family.py"
SPEC = importlib.util.spec_from_file_location(
    "cache_hit_rate_duration_by_family", SCRIPT
)
assert SPEC and SPEC.loader
PLOTTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOTTER
SPEC.loader.exec_module(PLOTTER)


def make_point(
    *,
    family: str = "count",
    launch_id: int = 1,
    l1_hit_rate: float = 60.0,
    l2_hit_rate: float = 80.0,
    duration_ms: float = 1.0,
    kernel_name: str | None = None,
):
    return PLOTTER.HitRatePoint(
        occurrence_id=f"occurrence-{launch_id}",
        query=1,
        launch_id=launch_id,
        family=family,
        kernel_name=(
            kernel_name
            or (
                "void open_addressing_ns::count<0>(T1)"
                if family == "count"
                else f"void {family}<0>(T1)"
            )
        ),
        l1_sector_hit_rate_pct=l1_hit_rate,
        l2_sector_hit_rate_pct=l2_hit_rate,
        duration_ms=duration_ms,
        source_csv="q01/profile.csv",
    )


def make_dataset(points, arm: str = "legacy_cta64"):
    return PLOTTER.ArmDataset(
        campaign=Path(f"/campaign/{arm}"),
        arm=arm,
        arm_label="Baseline" if arm == "legacy_cta64" else "Optimized Computation",
        points=tuple(points),
        fact_sheet_row_count=len(points),
        excluded_capture_count=0,
        capture_caveat="isolated replay",
    )


class CacheHitRateDurationByFamilyTests(unittest.TestCase):
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

    def test_summary_keeps_l1_and_l2_statistics_separate(self) -> None:
        dataset = make_dataset(
            [
                make_point(
                    launch_id=1,
                    l1_hit_rate=20.0,
                    l2_hit_rate=70.0,
                    duration_ms=0.1,
                ),
                make_point(
                    launch_id=2,
                    l1_hit_rate=40.0,
                    l2_hit_rate=90.0,
                    duration_ms=10.0,
                ),
            ]
        )
        row = PLOTTER.summarize_family(dataset, "count", panel_slot=7)
        self.assertEqual(row["launch_count"], 2)
        self.assertEqual(row["rendered_marker_count_per_figure"], 2)
        self.assertEqual(row["rendered_marker_count_across_both_figures"], 4)
        self.assertEqual(row["l1_sector_hit_rate_median_pct"], 30.0)
        self.assertEqual(row["l2_sector_hit_rate_median_pct"], 80.0)
        self.assertEqual(row["duration_median_ms"], 5.05)

    def test_axes_are_shared_across_levels_and_arms(self) -> None:
        baseline = make_dataset([make_point(duration_ms=0.00393519)])
        optimized = make_dataset(
            [make_point(duration_ms=84.95956481)],
            arm="synthesized_cta64",
        )
        shared = PLOTTER.shared_plot_parameters([baseline, optimized])
        self.assertEqual(shared.x_min_pct, -2.0)
        self.assertEqual(shared.x_max_pct, 102.0)
        self.assertEqual(shared.duration_min_ms, 0.001)
        self.assertEqual(shared.duration_max_ms, 100.0)

    def test_validation_rejects_non_target_count_symbol(self) -> None:
        dataset = make_dataset([make_point(kernel_name="void cuco::count<0>(T1)")])
        with self.assertRaisesRegex(ValueError, "non-target count"):
            PLOTTER.validate_dataset(dataset, expected_target_count=1)


if __name__ == "__main__":
    unittest.main()
