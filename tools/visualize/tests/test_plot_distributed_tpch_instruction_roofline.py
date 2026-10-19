# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path


VISUALIZE_DIR = Path(__file__).resolve().parents[1]
SCRIPT = VISUALIZE_DIR / "plot_distributed_tpch_instruction_roofline.py"
SPEC = importlib.util.spec_from_file_location("instruction_roofline", SCRIPT)
assert SPEC and SPEC.loader
ROOFLINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ROOFLINE
SPEC.loader.exec_module(ROOFLINE)


class InstructionRooflineTests(unittest.TestCase):
    def test_select_attempts_rebases_stale_absolute_paths(self) -> None:
        arm = "legacy_cta64"
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory) / "archived_campaign"
            status_rows = ["timestamp_utc,arm,stage,query,status,exit_code,attempt"]
            for query in ROOFLINE.QUERIES:
                relative = (
                    Path("arms")
                    / arm
                    / "ncu_profiles"
                    / f"q{query:02d}"
                    / "attempt-0001"
                )
                attempt = campaign / relative
                attempt.mkdir(parents=True)
                (attempt / f"q{query:02d}_ncu_metrics.csv").write_text("x\n")
                (attempt / f"q{query:02d}_ncu_validation.json").write_text(
                    '{"status":"pass","missing_requested_metrics":[],'
                    '"missing_launch_metrics":[]}'
                )
                stale = Path("/old/location/campaign") / relative
                status_rows.append(f"now,{arm},ncu_profiles,Q{query},pass,0,{stale}")
            (campaign / "status.csv").write_text("\n".join(status_rows) + "\n")

            self.assertEqual(ROOFLINE.infer_arm(campaign), arm)
            selected = ROOFLINE.select_attempts(campaign, arm)

        self.assertEqual(len(selected), 22)
        self.assertEqual(selected[0].query, 1)
        self.assertIn("archived_campaign", str(selected[0].attempt))

    def test_count_matcher_is_strict(self) -> None:
        desired = "void open_addressing_ns::count<0, 4, 128>(T1)"
        family, reason = ROOFLINE.classify_kernel(desired)
        self.assertEqual(family, "count")
        self.assertIsNone(reason)

        family, reason = ROOFLINE.classify_kernel("void cuco::count<0, 4, 128>(T1)")
        self.assertIsNone(family)
        self.assertEqual(reason, "non_target_count_symbol")

        family, reason = ROOFLINE.classify_kernel(
            "void transform_kernel<thrust::counting_iterator<long>>(T1)"
        )
        self.assertIsNone(family)
        self.assertEqual(reason, "incidental_counting_iterator_match")

        family, reason = ROOFLINE.classify_kernel(
            "void open_addressing_ns::insert_if_n<"
            "thrust::counting_iterator<long>>(T1)"
        )
        self.assertEqual(family, "insert_if_n")
        self.assertIsNone(reason)

    def test_build_point_uses_warp_instructions_and_average_sm_cycles(
        self,
    ) -> None:
        launch = ROOFLINE.RawLaunch(
            query=2,
            launch_id=7,
            attempt=Path("/campaign/attempt-0001"),
            source_csv=Path("/campaign/attempt-0001/q02_ncu_metrics.csv"),
            kernel_name="void open_addressing_ns::count<0>(T1)",
            process_id="123",
            device="0",
            context="1",
            stream="16",
            block_size="(128, 1, 1)",
            grid_size="(10, 1, 1)",
            nvtx_context=(
                "<default domain>:step[4]:DistributedEqJoin(foo):none "
                "<default domain>:phase:compute:eq_join.probe:none"
            ),
            metrics={
                "smsp__inst_executed.sum": 1200.0,
                "dram__sectors.sum": 100.0,
                "dram__sectors_read.sum": 75.0,
                "dram__sectors_write.sum": 25.0,
                "sm__cycles_elapsed.avg": 20.0,
                "dram__cycles_elapsed.avg": 40.0,
                "dram__throughput.sum.pct_of_peak_sustained_elapsed": 25.0,
                "gpu__time_duration.sum": 1000.0,
                "launch__sm_count": 152.0,
            },
        )
        point, excluded = ROOFLINE.build_point(launch, Path("/campaign"))
        self.assertIsNone(excluded)
        assert point is not None
        self.assertEqual(point.instruction_intensity, 12.0)
        self.assertEqual(point.instruction_throughput, 60.0)
        self.assertEqual(point.achieved_dram_sectors_per_sm_cycle, 5.0)
        self.assertEqual(point.inferred_peak_dram_sectors_per_dram_cycle, 10.0)
        self.assertEqual(point.inferred_peak_dram_sectors_per_sm_cycle, 20.0)
        self.assertEqual(point.step_index, "4")
        self.assertEqual(point.phase_name, "eq_join.probe")

    def test_roofs_use_report_derived_memory_rate(self) -> None:
        points = []
        for index, peak_sm in enumerate((126.0, 127.0, 128.0, 129.0, 130.0)):
            points.append(
                ROOFLINE.RooflinePoint(
                    query=1,
                    launch_id=index,
                    kernel_family="count",
                    kernel_name="void open_addressing_ns::count<0>(T1)",
                    step_index="1",
                    step_label="join",
                    phase_category="compute",
                    phase_name="join",
                    process_id="1",
                    device="0",
                    context="1",
                    stream="1",
                    block_size="(128, 1, 1)",
                    grid_size="(1, 1, 1)",
                    instructions=1000.0,
                    dram_sectors=100.0,
                    dram_read_sectors=75.0,
                    dram_write_sectors=25.0,
                    sm_cycles_elapsed=100.0,
                    dram_cycles_elapsed=200.0,
                    dram_throughput_pct_of_peak=10.0,
                    duration_ns=200_000.0,
                    duration_ms=0.2,
                    sm_count=152,
                    instruction_intensity=10.0,
                    instruction_throughput=10.0,
                    achieved_dram_sectors_per_sm_cycle=1.0,
                    inferred_peak_dram_sectors_per_dram_cycle=64.0,
                    inferred_peak_dram_sectors_per_sm_cycle=peak_sm,
                    source_csv="q01.csv",
                )
            )
        roofs = ROOFLINE.derive_roofs(points, 100.0, 1.0)
        self.assertEqual(roofs.sm_count, 152)
        self.assertEqual(roofs.instruction_roof_inst_per_cycle, 608.0)
        self.assertEqual(roofs.memory_roof_sector_per_sm_cycle, 128.0)
        self.assertTrue(
            math.isclose(
                roofs.ridge_point_instruction_per_sector,
                608.0 / 128.0,
            )
        )

        bounds = ROOFLINE.bound_summary(points, ridge=10.0)
        self.assertEqual(bounds["memory_bound_launch_count"], 0)
        self.assertEqual(bounds["compute_bound_launch_count"], 5)
        self.assertEqual(bounds["memory_bound_percent"], 0.0)
        self.assertEqual(bounds["compute_bound_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
