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
SCRIPT = VISUALIZE_DIR / "plot_distributed_tpch_l2_instruction_roofline_by_family.py"
SPEC = importlib.util.spec_from_file_location(
    "l2_instruction_roofline_by_family", SCRIPT
)
assert SPEC and SPEC.loader
PLOTTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOTTER
SPEC.loader.exec_module(PLOTTER)


def make_raw_point(
    *,
    launch_id: int = 1,
    instructions: float = 1_200.0,
    l2_total_sectors: float = 900.0,
    l2_read_hit_sectors: float = 300.0,
    sm_cycles_elapsed: float = 20.0,
    lts_to_sm_cycle_ratio: float = 0.97,
    duration_ns: float = 200_000.0,
    inferred_lts_instances: float = 190.0,
):
    return PLOTTER.RawL2Point(
        query=1,
        launch_id=launch_id,
        family="count",
        kernel_name="void open_addressing_ns::count<0>(T1)",
        instructions=instructions,
        l2_total_sectors=l2_total_sectors,
        l2_total_sectors_avg=l2_total_sectors / inferred_lts_instances,
        l2_read_hit_sectors=l2_read_hit_sectors,
        sm_cycles_elapsed=sm_cycles_elapsed,
        lts_cycles_elapsed=sm_cycles_elapsed * lts_to_sm_cycle_ratio,
        duration_ns=duration_ns,
        sm_count=152,
        inferred_lts_instances=inferred_lts_instances,
        source_csv="q01/profile.csv",
    )


class L2InstructionRooflineByFamilyTests(unittest.TestCase):
    def test_roof_uses_lts_count_architectural_rate_and_clock_ratio(self) -> None:
        ratios = [0.96, 0.97, 0.97, 0.98, 0.99]
        points = [
            make_raw_point(launch_id=index, lts_to_sm_cycle_ratio=ratio)
            for index, ratio in enumerate(ratios, start=1)
        ]
        roofs = PLOTTER.derive_l2_roofs(
            points, sectors_per_lts_cycle=2.0, minimum_duration_us=100.0
        )
        self.assertEqual(roofs.sm_count, 152)
        self.assertEqual(roofs.lts_instance_count, 190)
        self.assertEqual(roofs.calibration_launch_count, 5)
        self.assertAlmostEqual(roofs.lts_to_sm_cycle_ratio_median, 0.97)
        self.assertAlmostEqual(roofs.instruction_roof_inst_per_cycle, 608.0)
        self.assertAlmostEqual(roofs.l2_read_hit_roof_sectors_per_sm_cycle, 368.6)
        self.assertAlmostEqual(
            roofs.ridge_instruction_per_l2_read_hit_sector, 608.0 / 368.6
        )

    def test_point_metrics_and_compute_verdict(self) -> None:
        raw_points = [make_raw_point(launch_id=index) for index in range(1, 6)]
        roofs = PLOTTER.derive_l2_roofs(
            raw_points, sectors_per_lts_cycle=2.0, minimum_duration_us=100.0
        )
        point = PLOTTER.build_l2_point(raw_points[0], roofs)
        self.assertAlmostEqual(point.instruction_intensity, 4.0)
        self.assertAlmostEqual(point.instruction_throughput, 60.0)
        self.assertAlmostEqual(point.achieved_l2_read_hit_sectors_per_sm_cycle, 15.0)
        self.assertAlmostEqual(point.lts_to_sm_cycle_ratio, 0.97)
        self.assertEqual(point.bound_verdict, "compute")

    def test_point_below_ridge_is_l2_bound(self) -> None:
        raw_points = [make_raw_point(launch_id=index) for index in range(1, 6)]
        roofs = PLOTTER.derive_l2_roofs(
            raw_points, sectors_per_lts_cycle=2.0, minimum_duration_us=100.0
        )
        memory_point = make_raw_point(instructions=300.0, l2_read_hit_sectors=300.0)
        point = PLOTTER.build_l2_point(memory_point, roofs)
        self.assertEqual(point.bound_verdict, "memory")

    def test_roof_rejects_inconsistent_lts_instance_count(self) -> None:
        points = [make_raw_point(launch_id=index) for index in range(1, 6)]
        points[-1] = make_raw_point(launch_id=5, inferred_lts_instances=189.0)
        with self.assertRaisesRegex(ValueError, "inconsistent inferred LTS"):
            PLOTTER.derive_l2_roofs(
                points, sectors_per_lts_cycle=2.0, minimum_duration_us=100.0
            )

    def test_roof_rejects_unstable_cycle_conversion(self) -> None:
        ratios = [0.5, 0.7, 0.97, 1.3, 1.5]
        points = [
            make_raw_point(launch_id=index, lts_to_sm_cycle_ratio=ratio)
            for index, ratio in enumerate(ratios, start=1)
        ]
        with self.assertRaisesRegex(ValueError, "cycle conversion is unstable"):
            PLOTTER.derive_l2_roofs(
                points, sectors_per_lts_cycle=2.0, minimum_duration_us=100.0
            )


if __name__ == "__main__":
    unittest.main()
