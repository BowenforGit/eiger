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
SCRIPT = VISUALIZE_DIR / "plot_distributed_tpch_compute_memory_throughput.py"
SPEC = importlib.util.spec_from_file_location("compute_memory_throughput", SCRIPT)
assert SPEC and SPEC.loader
PLOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT
SPEC.loader.exec_module(PLOT)


def make_launch(kernel_name: str = "void open_addressing_ns::count<0>(T1)"):
    return PLOT.common.RawLaunch(
        query=2,
        launch_id=7,
        attempt=Path("/campaign/attempt-0001"),
        source_csv=Path("/campaign/attempt-0001/q02_ncu_metrics.csv"),
        kernel_name=kernel_name,
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
            PLOT.COMPUTE_METRIC: 63.0,
            PLOT.LEVEL_METRICS["dram"]: 35.0,
            PLOT.LEVEL_METRICS["l2"]: 72.0,
            PLOT.LEVEL_METRICS["l1"]: 51.0,
            PLOT.DURATION_METRIC: 2_500_000.0,
        },
    )


class ComputeMemoryThroughputTests(unittest.TestCase):
    def test_build_point_uses_max_memory_throughput_and_shape(self) -> None:
        point, excluded = PLOT.build_point(make_launch(), Path("/campaign"))
        self.assertIsNone(excluded)
        assert point is not None
        self.assertEqual(point.compute_throughput_pct, 63.0)
        self.assertEqual(point.memory_throughput_pct, 72.0)
        self.assertEqual(point.dominant_memory_level, "l2")
        self.assertEqual(point.dominant_memory_metric, PLOT.LEVEL_METRICS["l2"])
        self.assertEqual(point.duration_ms, 2.5)

    def test_dominant_level_ties_use_documented_priority(self) -> None:
        level, maximum, tie_count = PLOT.dominant_memory_level(
            {"dram": 80.0, "l2": 80.0, "l1": 80.0}
        )
        self.assertEqual(level, "dram")
        self.assertEqual(maximum, 80.0)
        self.assertEqual(tie_count, 3)

    def test_non_target_count_symbol_is_excluded(self) -> None:
        point, excluded = PLOT.build_point(
            make_launch("void cuco::count<0>(T1)"), Path("/campaign")
        )
        self.assertIsNone(point)
        assert excluded is not None
        self.assertEqual(excluded.reason, "non_target_count_symbol")


if __name__ == "__main__":
    unittest.main()
