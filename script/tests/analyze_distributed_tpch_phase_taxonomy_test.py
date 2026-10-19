#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Compatibility tests for distributed TPC-H phase analysis."""

import importlib.util
import pathlib
import sqlite3
import tempfile
import unittest


REPOSITORY = pathlib.Path(__file__).resolve().parents[2]


def load_module(name, relative_path):
    specification = importlib.util.spec_from_file_location(
        name, REPOSITORY / relative_path
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


PLAN_ANALYZER = load_module(
    "distributed_tpch_plan_analyzer",
    "script/analyze_distributed_tpch_plans.py",
)
COLUMN_OVERLAP_ANALYZER = load_module(
    "distributed_tpch_column_overlap_analyzer",
    "exp_results/distributed_tpch_legacy_vs_column_overlap_20260728/analyze.py",
)


def phase_row(category, name, primitive="", **extra):
    row = {
        "category": category,
        "phase_name": name,
        "collective_primitive": primitive,
        "communication_pattern": "shuffle",
        "start_ms": "1.0",
        "end_ms": "4.0",
    }
    row.update(extra)
    return row


class PhaseTaxonomyCompatibilityTest(unittest.TestCase):
    def test_old_and_new_hash_partition_rows_are_compute(self):
        old = phase_row(
            "communication",
            "shuffle.left.hash_partition",
            "hash_partition",
        )
        new = phase_row(
            "compute",
            "shuffle.left.hash_partition",
            "hash_partition",
        )
        for analyzer in (PLAN_ANALYZER, COLUMN_OVERLAP_ANALYZER):
            self.assertEqual(analyzer.phase_component(old), "Compute")
            self.assertEqual(analyzer.phase_component(new), "Compute")
            self.assertEqual(
                analyzer.attribute_intervals([old], 5.0),
                analyzer.attribute_intervals([new], 5.0),
            )

    def test_exchange_remains_communication(self):
        exchange = phase_row(
            "communication",
            "join.column.key_exchange",
            "all_to_all",
        )
        for analyzer in (PLAN_ANALYZER, COLUMN_OVERLAP_ANALYZER):
            self.assertEqual(analyzer.phase_component(exchange), "Shuffle")

    def test_optional_column_identity_supports_both_schemas(self):
        old = phase_row(
            "communication",
            "join.column.key_exchange",
            "all_to_all",
        )
        new = phase_row(
            "communication",
            "join.column.key_exchange",
            "all_to_all",
            column_side="right",
            column_ordinal="3",
        )
        for analyzer in (PLAN_ANALYZER, COLUMN_OVERLAP_ANALYZER):
            self.assertEqual(analyzer.phase_column_identity(old), ("", -1))
            self.assertEqual(
                analyzer.phase_column_identity(new), ("right", 3)
            )

    def test_old_and_new_nvtx_hash_partition_labels_are_compute(self):
        old = (
            "phase:communication:shuffle:"
            "shuffle.right.hash_partition"
        )
        new = "phase:compute:shuffle:shuffle.right.hash_partition"
        self.assertEqual(
            PLAN_ANALYZER.normalized_nvtx_phase_scope(old),
            "phase_compute",
        )
        self.assertEqual(
            PLAN_ANALYZER.normalized_nvtx_phase_scope(new),
            "phase_compute",
        )
        self.assertEqual(
            PLAN_ANALYZER.normalized_nvtx_phase_scope(
                "phase:communication:shuffle:join.column.key_exchange"
            ),
            "phase_communication",
        )

    def test_alignment_mechanism_nvtx_scopes(self):
        self.assertEqual(
            PLAN_ANALYZER.mechanism_nvtx_phase_scope(
                "phase:communication:shuffle:shuffle.exchange"
            ),
            "phase_shuffle",
        )
        self.assertEqual(
            PLAN_ANALYZER.mechanism_nvtx_phase_scope(
                "phase:communication:broadcast:broadcast.left"
            ),
            "phase_replication",
        )
        self.assertEqual(
            PLAN_ANALYZER.mechanism_nvtx_phase_scope(
                "phase:communication:sampling:sampling.replicate"
            ),
            "phase_replication",
        )
        self.assertIsNone(
            PLAN_ANALYZER.mechanism_nvtx_phase_scope(
                "phase:communication:gather:gather.to_root"
            )
        )

    def test_column_overlap_direct_breakdown_normalizes_old_rows(self):
        old_hash = phase_row(
            "communication",
            "shuffle.right.hash_partition",
            "hash_partition",
        )
        new_hash = dict(old_hash, category="compute")
        exchange = phase_row(
            "communication",
            "join.column.key_exchange",
            "all_to_all",
            start_ms="3.0",
            end_ms="5.0",
        )
        expected = {
            "Compute": 2.0,
            "Overlap": 1.0,
            "Communication": 1.0,
        }
        for hash_row in (old_hash, new_hash):
            self.assertEqual(
                dict(
                    COLUMN_OVERLAP_ANALYZER.attribute_compute_communication(
                        [hash_row, exchange], 6.0
                    )
                ),
                expected,
            )

    def test_plan_analyzer_rejects_missing_breakdown_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError, "requires non-empty steps.csv and phases.csv"
            ):
                PLAN_ANALYZER.breakdown_summaries(directory)

    def test_nccl_kernel_intervals_use_cupti_timestamps_and_physical_gpu(self):
        database = sqlite3.connect(":memory:")
        database.executescript(
            """
            CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO(
                processId INTEGER, deviceId INTEGER
            );
            CREATE TABLE TARGET_INFO_CUDA_DEVICE(
                pid INTEGER, cudaId INTEGER, gpuId INTEGER
            );
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                start INTEGER, end INTEGER, globalPid INTEGER,
                demangledName INTEGER, shortName INTEGER
            );
            """
        )
        process_id = 37
        global_pid = process_id << 24
        database.execute(
            "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (?,?)",
            (process_id, 0),
        )
        database.execute(
            "INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (?,?,?)",
            (process_id, 0, 5),
        )
        database.executemany(
            "INSERT INTO StringIds VALUES (?,?)",
            (
                (1, "ncclDevKernel_SendRecv"),
                (2, "ordinary_compute_kernel"),
            ),
        )
        database.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)",
            (
                (100, 200, global_pid, 1, 1),
                (190, 250, global_pid, 1, 1),
                (150, 300, global_pid, 2, 2),
            ),
        )
        self.assertEqual(
            PLAN_ANALYZER.nccl_kernel_intervals(database),
            {5: [(100, 250)]},
        )
        database.close()


if __name__ == "__main__":
    unittest.main()
