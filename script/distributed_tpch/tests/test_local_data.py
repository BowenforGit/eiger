# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

import local_data
import run
import topology


class PartitionAssignmentTests(unittest.TestCase):
    def test_round_robin_assignment_matches_reader(self) -> None:
        self.assertEqual(
            local_data.assigned_part_indices(10, 2, 4, "round_robin"),
            (2, 6),
        )

    def test_contiguous_assignment_matches_reader(self) -> None:
        assignments = [
            local_data.assigned_part_indices(10, rank, 4, "contiguous_range")
            for rank in range(4)
        ]
        self.assertEqual(assignments, [(0, 1), (2, 3, 4), (5, 6), (7, 8, 9)])

    def test_host_ranks_are_global_and_contiguous(self) -> None:
        cluster = topology.Topology(
            (topology.Host("a", 4), topology.Host("b", 4)), False, 2222
        )
        self.assertEqual(
            run.host_global_ranks(cluster),
            {"a": (0, 1, 2, 3), "b": (4, 5, 6, 7)},
        )

    def test_cache_path_encodes_scale_partition_table_and_part(self) -> None:
        tasks = local_data.required_tasks(
            Path("/tmp/cache"), 3000, 64, (3,), 16, "round_robin"
        )
        lineitem = [task for task in tasks if task.table == "lineitem"]
        self.assertEqual([task.part for task in lineitem], [4, 20, 36, 52])
        self.assertEqual(
            lineitem[0].output,
            Path("/tmp/cache/parts64/sf3000/lineitem/lineitem.4.parquet"),
        )

    def test_one_partition_uses_single_files(self) -> None:
        tasks = local_data.required_tasks(
            Path("/tmp/cache"), 300, 1, (0,), 8, "round_robin"
        )
        self.assertEqual(len(tasks), 8)
        self.assertTrue(all(task.part is None for task in tasks))


class SyntheticManifestTests(unittest.TestCase):
    @staticmethod
    def records(partitions: int) -> list[dict]:
        records = []
        for table in local_data.ALL_TABLES:
            parts = (
                (None,)
                if partitions == 1 or table in local_data.REPLICATED_TABLES
                else tuple(range(1, partitions + 1))
            )
            for part in parts:
                path = (
                    f"{table}.parquet"
                    if part is None
                    else f"{table}/{table}.{part}.parquet"
                )
                records.append(
                    {
                        "table": table,
                        "part": part,
                        "path": path,
                        "bytes": 100 + (part or 0),
                        "rows": 10 + (part or 0),
                        "row_groups": 1,
                        "schema": [
                            {
                                "name": f"{table}_key",
                                "nullable": True,
                                "type": "int64",
                            }
                        ],
                    }
                )
        return records

    def test_manifest_declares_all_global_parts(self) -> None:
        manifest = local_data.build_synthetic_manifest(
            self.records(4), scale=1000, partitions=4, generator="tpchgen-cli"
        )
        lineitem = manifest["tables"]["lineitem"]
        self.assertEqual(lineitem["layout"], "partitioned")
        self.assertEqual(lineitem["part_count"], 4)
        self.assertEqual([part["id"] for part in lineitem["parts"]], [1, 2, 3, 4])
        self.assertEqual(manifest["tables"]["nation"]["layout"], "single")

    def test_manifest_bundle_success_hash_matches_bytes(self) -> None:
        manifest = local_data.build_synthetic_manifest(
            self.records(1), scale=300, partitions=1, generator="tpchgen-cli"
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, success_path = local_data.write_manifest_bundle(
                Path(temporary), manifest
            )
            expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            self.assertEqual(success_path.read_text(), f"manifest_sha256={expected}\n")
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)


class LauncherConfigurationTests(unittest.TestCase):
    @staticmethod
    def generated_args(generator: Path, partitions: int, synthesis_arg: list[str]):
        return Namespace(
            generate_local_data=True,
            generated_data_partitions=partitions,
            tpchgen=generator,
            local_data_cache_root=Path("/tmp/cache"),
            synthesis_arg=synthesis_arg,
        )

    def test_generated_mode_requires_partition_count(self) -> None:
        cluster = topology.Topology((topology.Host("a", 8),), False, 2222)
        args = Namespace(
            generate_local_data=True,
            generated_data_partitions=None,
            tpchgen=Path("/missing"),
            local_data_cache_root=Path("/tmp/cache"),
        )
        with self.assertRaisesRegex(run.ConfigurationError, "is required"):
            run.validate_generated_data_configuration(args, cluster)

    def test_partitioned_mode_rejects_empty_rank_assignments(self) -> None:
        cluster = topology.Topology((topology.Host("a", 8),), False, 2222)
        args = Namespace(
            generate_local_data=True,
            generated_data_partitions=4,
            tpchgen=Path("/missing"),
            local_data_cache_root=Path("/tmp/cache"),
        )
        with self.assertRaisesRegex(run.ConfigurationError, "at least the MPI rank"):
            run.validate_generated_data_configuration(args, cluster)

    def test_generated_commands_skip_only_dataset_preflight(self) -> None:
        args = Namespace(generate_local_data=True, parquet_decode_threads=4)
        self.assertIn("--skip-dataset-preflight", run.common_benchmark_args(args))

    def test_generator_is_staged_as_a_campaign_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source-tpchgen"
            source.write_bytes(b"generator-v1")
            destination = run.campaign_tpchgen_path(root / "campaign")
            run.stage_campaign_tpchgen(source, destination)
            self.assertEqual(destination.read_bytes(), b"generator-v1")
            self.assertTrue(destination.stat().st_mode & 0o111)

            source.write_bytes(b"generator-v2")
            run.stage_campaign_tpchgen(source, destination)
            self.assertEqual(destination.read_bytes(), b"generator-v2")

    def test_partitioned_generation_accepts_round_robin_synthesis_override(
        self,
    ) -> None:
        cluster = topology.Topology((topology.Host("a", 4),), False, 2222)
        with tempfile.TemporaryDirectory() as temporary:
            generator = Path(temporary) / "tpchgen-cli"
            generator.touch(mode=0o755)
            args = self.generated_args(
                generator,
                8,
                ["--file-shard-strategy", "round-robin"],
            )
            run.validate_generated_data_configuration(args, cluster)

    def test_partitioned_generation_rejects_contiguous_synthesis_override(
        self,
    ) -> None:
        cluster = topology.Topology((topology.Host("a", 4),), False, 2222)
        with tempfile.TemporaryDirectory() as temporary:
            generator = Path(temporary) / "tpchgen-cli"
            generator.touch(mode=0o755)
            args = self.generated_args(
                generator,
                8,
                ["--file-shard-strategy=contiguous_range"],
            )
            with self.assertRaisesRegex(
                run.ConfigurationError, "requires synthesis file sharding"
            ):
                run.validate_generated_data_configuration(args, cluster)


if __name__ == "__main__":
    unittest.main()
