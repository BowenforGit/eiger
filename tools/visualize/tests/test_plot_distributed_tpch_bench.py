#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.


import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "plot_distributed_tpch_bench.py"
SPEC = importlib.util.spec_from_file_location("plot_distributed_tpch_bench", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PhaseAggregationTest(unittest.TestCase):
    def test_selects_critical_rank_then_median_iteration(self):
        selected = MODULE.select_critical_median_samples(
            {
                (1, "run-a", 0, 0): 10.0,
                (1, "run-a", 0, 1): 12.0,
                (1, "run-a", 1, 0): 30.0,
                (1, "run-a", 1, 1): 20.0,
                (1, "run-a", 2, 0): 15.0,
                (1, "run-a", 2, 1): 18.0,
            }
        )
        self.assertEqual(selected[1], ("run-a", 2, 1, 18.0))

    def test_critical_rank_tie_uses_lower_rank(self):
        selected = MODULE.select_critical_median_samples(
            {(7, "run-a", 3, 2): 9.0, (7, "run-a", 3, 1): 9.0}
        )
        self.assertEqual(selected[7], ("run-a", 3, 1, 9.0))

    def test_appended_runs_remain_distinct_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            steps_path = Path(directory) / "steps.csv"
            phases_path = Path(directory) / "phases.csv"
            step_fields = [
                "timestamp_utc",
                "query",
                "scale",
                "record_type",
                "rank",
                "iteration",
                "gpu_ms",
            ]
            phase_fields = step_fields + ["category", "communication_pattern"]
            with steps_path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=step_fields)
                writer.writeheader()
                for query in MODULE.QUERIES:
                    for timestamp, duration in (("run-a", 5.0), ("run-b", 9.0)):
                        writer.writerow(
                            {
                                "timestamp_utc": timestamp,
                                "query": f"Q{query}",
                                "scale": 100,
                                "record_type": "rank",
                                "rank": 0,
                                "iteration": 0,
                                "gpu_ms": duration,
                            }
                        )
            with phases_path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=phase_fields)
                writer.writeheader()
                for query in MODULE.QUERIES:
                    for timestamp, duration in (("run-a", 5.0), ("run-b", 9.0)):
                        writer.writerow(
                            {
                                "timestamp_utc": timestamp,
                                "query": f"Q{query}",
                                "scale": 100,
                                "record_type": "rank",
                                "rank": 0,
                                "iteration": 0,
                                "gpu_ms": duration,
                                "category": "compute",
                                "communication_pattern": "none",
                            }
                        )

            summaries = MODULE.summarize_phases(steps_path, phases_path, 100)
            self.assertEqual(summaries[1].timestamp_utc, "run-a")
            self.assertEqual(summaries[1].step_total_ms, 5.0)
            self.assertEqual(summaries[1].components_ms["Compute"], 5.0)

    def test_phase_component_keeps_pattern_separate_from_category(self):
        self.assertEqual(
            MODULE.phase_component(
                {"category": "communication", "communication_pattern": "broadcast"}
            ),
            "Broadcast",
        )
        self.assertEqual(
            MODULE.phase_component(
                {"category": "compute", "communication_pattern": "none"}
            ),
            "Compute",
        )

    def test_interval_attribution_does_not_double_count_overlap(self):
        rows = [
            {
                "start_ms": "0",
                "end_ms": "6",
                "category": "communication",
                "communication_pattern": "shuffle",
            },
            {
                "start_ms": "4",
                "end_ms": "9",
                "category": "compute",
                "communication_pattern": "none",
            },
        ]
        attributed = MODULE.attribute_phase_intervals(rows, 10.0)
        self.assertEqual(attributed["Shuffle"], 4.0)
        self.assertEqual(attributed["Overlap"], 2.0)
        self.assertEqual(attributed["Compute"], 3.0)
        self.assertEqual(sum(attributed.values()), 9.0)

    def test_interval_phase_summary_reconciles_overlapping_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            steps_path = Path(directory) / "steps.csv"
            phases_path = Path(directory) / "phases.csv"
            step_fields = [
                "timestamp_utc", "query", "scale", "record_type", "rank",
                "iteration", "step_index", "gpu_ms",
            ]
            phase_fields = step_fields + [
                "category", "communication_pattern", "start_ms", "end_ms",
                "quality",
            ]
            with steps_path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=step_fields)
                writer.writeheader()
                for query in MODULE.QUERIES:
                    writer.writerow(
                        {
                            "timestamp_utc": "run-a", "query": f"Q{query}",
                            "scale": 100, "record_type": "rank", "rank": 0,
                            "iteration": 0, "step_index": 3, "gpu_ms": 10.0,
                        }
                    )
            with phases_path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=phase_fields)
                writer.writeheader()
                for query in MODULE.QUERIES:
                    common = {
                        "timestamp_utc": "run-a", "query": f"Q{query}",
                        "scale": 100, "record_type": "rank", "rank": 0,
                        "iteration": 0, "step_index": 3,
                    }
                    writer.writerow(
                        common | {
                            "gpu_ms": 6.0, "category": "communication",
                            "communication_pattern": "shuffle", "start_ms": 0,
                            "end_ms": 6, "quality": "direct",
                        }
                    )
                    writer.writerow(
                        common | {
                            "gpu_ms": 5.0, "category": "compute",
                            "communication_pattern": "none", "start_ms": 4,
                            "end_ms": 9, "quality": "direct",
                        }
                    )
                    writer.writerow(
                        common | {
                            "gpu_ms": 1.0, "category": "overhead",
                            "communication_pattern": "none", "start_ms": -1,
                            "end_ms": -1, "quality": "derived",
                        }
                    )

            summaries = MODULE.summarize_phases(steps_path, phases_path, 100)
            values = summaries[1].components_ms
            self.assertEqual(values["Shuffle"], 4.0)
            self.assertEqual(values["Overlap"], 2.0)
            self.assertEqual(values["Compute"], 3.0)
            self.assertEqual(values["Overhead"], 1.0)
            self.assertEqual(sum(values.values()), 10.0)

    def test_e2e_caption_uses_observed_sample_counts(self):
        summary = MODULE.TimingSummary(1.0, 0.5, 1.5, 7)
        caption = MODULE.e2e_caption({100: {1: summary}, 300: {1: summary}})
        self.assertIn("across 7 runs per query", caption)


if __name__ == "__main__":
    unittest.main()
