# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


VISUALIZE_DIR = Path(__file__).resolve().parents[1]
if str(VISUALIZE_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZE_DIR))
SCRIPT = VISUALIZE_DIR / "export_distributed_tpch_ncu_kernel_fact_sheet.py"
SPEC = importlib.util.spec_from_file_location("ncu_kernel_fact_sheet", SCRIPT)
assert SPEC and SPEC.loader
EXPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORTER
SPEC.loader.exec_module(EXPORTER)


def make_launch(kernel_name: str = "void open_addressing_ns::count<0>(T1)"):
    metrics = {metric: 1.0 for metric in EXPORTER.REQUIRED_METRICS}
    metrics.update(
        {
            "launch__sm_count": 152.0,
            "smsp__inst_executed.sum": 1_200.0,
            "dram__sectors.sum": 100.0,
            "dram__sectors_read.sum": 75.0,
            "dram__sectors_write.sum": 25.0,
            "dram__bytes.sum": 3_200.0,
            "dram__bytes_read.sum": 2_400.0,
            "dram__bytes_write.sum": 800.0,
            "sm__cycles_elapsed.avg": 20.0,
            "gpu__time_duration.sum": 2_500_000.0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 63.0,
            "l1tex__throughput.avg.pct_of_peak_sustained_active": 51.0,
            "lts__throughput.avg.pct_of_peak_sustained_elapsed": 72.0,
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 35.0,
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": 18.0,
        }
    )
    return EXPORTER.FactLaunch(
        query=2,
        launch_id=7,
        attempt=Path("/campaign/arms/legacy/ncu_profiles/q02/attempt-0001"),
        source_csv=Path(
            "/campaign/arms/legacy/ncu_profiles/q02/attempt-0001/q02_ncu_metrics.csv"
        ),
        kernel_name=kernel_name,
        process_id="123",
        process_name="benchmark_tpch_ir_mpi",
        profile_host="host0",
        device="0",
        compute_capability="10.0",
        context="1",
        stream="16",
        block_size="(128, 1, 1)",
        grid_size="(10, 1, 1)",
        nvtx_context=(
            "<default domain>:step[4]:DistributedEqJoin(foo):none "
            "<default domain>:phase:compute:eq_join.probe:none"
        ),
        metrics=metrics,
    )


def metadata():
    return EXPORTER.CampaignMetadata(
        campaign=Path("/campaign"),
        arm="legacy_cta64",
        arm_label="Baseline",
        scale_factor=1000,
        topology="nodes2_gpn4_np8",
        node_count=2,
        ranks=8,
        gpus_per_node=4,
        gpu_count=8,
        hosts="host0,host1",
        gpu_model="NVIDIA GB200",
        ncu_version="2025.3.1.0",
        ncu_capture_host="host0",
    )


class NcuKernelFactSheetTests(unittest.TestCase):
    def test_build_row_preserves_components_and_derives_characteristics(self) -> None:
        roofs = EXPORTER.Roofs(152, 608.0, 128.0, 4.75, 10, 0.01)
        row = EXPORTER.build_row(make_launch(), metadata(), roofs)
        self.assertEqual(set(row), set(EXPORTER.FACT_SHEET_FIELDS))
        self.assertEqual(row["kernel_duration_ms"], 2.5)
        self.assertEqual(row["dram_sectors_total"], 100)
        self.assertEqual(row["dram_sectors_read"], 75)
        self.assertEqual(row["dram_sectors_write"], 25)
        self.assertEqual(row["instruction_intensity_inst_per_dram_sector"], 12.0)
        self.assertEqual(row["instruction_throughput_inst_per_sm_cycle"], 60.0)
        self.assertEqual(row["bound_verdict"], "compute")
        self.assertEqual(row["dominant_memory_level"], "l2")
        self.assertEqual(row["leading_stall_reason"], "long_scoreboard")
        self.assertTrue(row["target_kernel_match"])

    def test_non_target_count_is_retained_but_flagged(self) -> None:
        roofs = EXPORTER.Roofs(152, 608.0, 128.0, 4.75, 10, 0.01)
        row = EXPORTER.build_row(
            make_launch("void cuco::count<0>(T1)"), metadata(), roofs
        )
        self.assertFalse(row["target_kernel_match"])
        self.assertEqual(row["target_match_reason"], "non_target_count_symbol")
        self.assertEqual(row["kernel_family"], "")

    def test_derive_roofs_uses_target_launches(self) -> None:
        launches = []
        for peak in (126.0, 127.0, 128.0, 129.0, 130.0):
            launch = make_launch()
            launch.metrics["dram__sectors.sum"] = peak
            launch.metrics["dram__sectors_read.sum"] = peak - 25.0
            launch.metrics["sm__cycles_elapsed.avg"] = 10.0
            launch.metrics["dram__throughput.sum.pct_of_peak_sustained_elapsed"] = 10.0
            launches.append(launch)
        roofs = EXPORTER.derive_roofs(launches)
        self.assertEqual(roofs.instruction_roof, 608.0)
        self.assertTrue(math.isclose(roofs.memory_roof, 128.0))
        self.assertTrue(math.isclose(roofs.ridge, 608.0 / 128.0))


if __name__ == "__main__":
    unittest.main()
