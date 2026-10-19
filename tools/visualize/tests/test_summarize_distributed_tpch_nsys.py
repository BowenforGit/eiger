#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.


import importlib.util
import sqlite3
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "summarize_distributed_tpch_nsys.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_distributed_tpch_nsys", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class HardwareMetricSummaryTest(unittest.TestCase):
    def test_parse_integer_ranges(self):
        self.assertEqual(MODULE.parse_int_spec("1,3-5,3", 1, 22), [1, 3, 4, 5])
        with self.assertRaises(ValueError):
            MODULE.parse_int_spec("5-3", 1, 22)

    def test_percentile_interpolates(self):
        self.assertEqual(MODULE.percentile([0.0, 10.0], 0.5), 5.0)

    def test_derived_bandwidth_metrics_are_sample_aligned(self):
        metrics = {
            MODULE.DRAM_READ: [10.0, 20.0],
            MODULE.DRAM_WRITE: [1.0, 2.0],
            MODULE.PCIE_RX: [3.0, 4.0],
            MODULE.PCIE_TX: [5.0, 6.0],
        }
        for index, name in enumerate(MODULE.NVLINK_RX):
            metrics[name] = [float(index + 1), float(index + 2)]
        for index, name in enumerate(MODULE.NVLINK_TX):
            metrics[name] = [float(index + 5), float(index + 6)]

        MODULE.add_derived_metrics(metrics)

        self.assertEqual(metrics[MODULE.DERIVED_DRAM], [11.0, 22.0])
        self.assertEqual(metrics[MODULE.DERIVED_NVLINK_RX], [10.0, 14.0])
        self.assertEqual(metrics[MODULE.DERIVED_NVLINK_TX], [26.0, 30.0])
        self.assertEqual(metrics[MODULE.DERIVED_NVLINK_DUPLEX], [18.0, 22.0])
        self.assertEqual(metrics[MODULE.DERIVED_PCIE], [8.0, 10.0])

    def test_derived_metrics_reject_misaligned_samples(self):
        metrics = {
            MODULE.DRAM_READ: [1.0],
            MODULE.DRAM_WRITE: [1.0, 2.0],
        }
        with self.assertRaisesRegex(RuntimeError, "sample count mismatch"):
            MODULE.add_derived_metrics(metrics)

    def test_profile_window_uses_distributed_envelope(self):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        connection.execute(
            "CREATE TABLE NVTX_EVENTS "
            "(start INTEGER, end INTEGER, text TEXT, textId INTEGER)"
        )
        connection.execute(
            "INSERT INTO StringIds VALUES (1, ?)", (MODULE.PROFILE_RANGE,)
        )
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, NULL, 1)",
            [(100, 300), (125, 290), (120, 310), (121, 291)],
        )

        (start, end, durations, start_skew, end_skew,
         range_count) = MODULE.profile_window(connection)

        self.assertEqual((start, end), (100, 310))
        self.assertEqual(durations, [200, 190, 170, 165])
        self.assertEqual(start_skew, 25)
        self.assertEqual(end_skew, 20)
        self.assertEqual(range_count, 4)


if __name__ == "__main__":
    unittest.main()
