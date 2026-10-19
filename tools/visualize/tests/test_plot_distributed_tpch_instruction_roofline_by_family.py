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
SCRIPT = VISUALIZE_DIR / "plot_distributed_tpch_instruction_roofline_by_family.py"
SPEC = importlib.util.spec_from_file_location("instruction_roofline_by_family", SCRIPT)
assert SPEC and SPEC.loader
PLOTTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOTTER
SPEC.loader.exec_module(PLOTTER)


def make_point(
    family: str = "count",
    intensity: float = 10.0,
    throughput: float = 100.0,
    duration_ms: float = 1.0,
    verdict: str = "compute",
    kernel_name: str = "void open_addressing_ns::count<0>(T1)",
    launch_id: int = 1,
):
    return PLOTTER.FamilyPoint(
        query=1,
        launch_id=launch_id,
        family=family,
        kernel_name=kernel_name,
        instruction_intensity=intensity,
        instruction_throughput=throughput,
        duration_ms=duration_ms,
        bound_verdict=verdict,
    )


def make_dataset(points, arm: str = "legacy_cta64"):
    return PLOTTER.ArmDataset(
        campaign=Path(f"/campaign/{arm}"),
        arm=arm,
        arm_label="Baseline" if arm == "legacy_cta64" else "Optimized Computation",
        points=tuple(points),
        instruction_roof=608.0,
        memory_roof=128.0,
        ridge=4.75,
        fact_sheet_row_count=len(points),
        excluded_capture_count=0,
        capture_caveat="isolated replay",
    )


class InstructionRooflineByFamilyTests(unittest.TestCase):
    def test_expression_implementations_share_one_panel_slot(self) -> None:
        baseline = make_dataset(
            [
                make_point(
                    family="evaluate_expr_kernel",
                    kernel_name="evaluate_expr_kernel<T>",
                )
            ]
        )
        optimized = make_dataset(
            [
                make_point(
                    family="kernel_binary_op_vec",
                    kernel_name="kernel_binary_op_vec<T>",
                )
            ],
            arm="synthesized_cta64",
        )
        self.assertEqual(
            PLOTTER.family_for_slot(baseline, PLOTTER.PANEL_SLOTS[0]),
            "evaluate_expr_kernel",
        )
        self.assertEqual(
            PLOTTER.family_for_slot(optimized, PLOTTER.PANEL_SLOTS[0]),
            "kernel_binary_op_vec",
        )

    def test_family_summary_counts_bound_verdicts(self) -> None:
        dataset = make_dataset(
            [
                make_point(intensity=1.0, verdict="memory", launch_id=1),
                make_point(intensity=10.0, verdict="compute", launch_id=2),
                make_point(intensity=20.0, verdict="compute", launch_id=3),
            ]
        )
        row = PLOTTER.summarize_family(dataset, "count", 7)
        self.assertEqual(row["launch_count"], 3)
        self.assertEqual(row["memory_bound_count"], 1)
        self.assertAlmostEqual(row["memory_bound_percent"], 100.0 / 3.0)
        self.assertAlmostEqual(row["compute_bound_percent"], 200.0 / 3.0)

    def test_axes_and_duration_normalization_are_shared(self) -> None:
        baseline = make_dataset(
            [
                make_point(
                    intensity=0.2,
                    throughput=2.0,
                    duration_ms=0.1,
                    verdict="memory",
                )
            ]
        )
        optimized = make_dataset(
            [
                make_point(
                    intensity=200.0,
                    throughput=600.0,
                    duration_ms=10.0,
                )
            ],
            arm="synthesized_cta64",
        )
        shared = PLOTTER.shared_plot_parameters([baseline, optimized])
        self.assertEqual(shared.x_min, 0.1)
        self.assertEqual(shared.x_max, 1000.0)
        self.assertEqual(shared.y_min, 1.0)
        self.assertEqual(shared.y_max, 1000.0)
        self.assertEqual(shared.duration_min_ms, 0.1)
        self.assertEqual(shared.duration_max_ms, 10.0)

    def test_validation_rejects_non_target_count_symbol(self) -> None:
        dataset = make_dataset([make_point(kernel_name="void cuco::count<0>(T1)")])
        with self.assertRaisesRegex(ValueError, "non-target count kernel"):
            PLOTTER.validate_dataset(dataset, expected_target_count=1)

    def test_validation_rejects_verdict_that_disagrees_with_ridge(self) -> None:
        dataset = make_dataset([make_point(intensity=1.0, verdict="compute")])
        with self.assertRaisesRegex(ValueError, "bound verdicts disagree"):
            PLOTTER.validate_dataset(dataset, expected_target_count=1)


if __name__ == "__main__":
    unittest.main()
