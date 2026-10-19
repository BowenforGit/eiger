#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.


import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from tpch_dataset_layout import (  # noqa: E402
    DatasetLayoutError,
    resolve_table_files,
)


class DatasetLayoutTest(unittest.TestCase):
    def schema_fingerprint(self, schema: list[dict]) -> str:
        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def write_completed_manifest(self, root: Path, manifest: dict) -> None:
        text = json.dumps(manifest, indent=2) + "\n"
        manifest_path = root / "manifest.json"
        manifest_path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (root / "_SUCCESS").write_text(
            f"manifest_sha256={digest}\n", encoding="utf-8")

    def test_legacy_single_file_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "region.parquet"
            path.write_bytes(b"parquet")
            self.assertEqual(
                resolve_table_files(root, "region", 1), [str(path)])

    def test_manifest_order_is_numeric_not_lexical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "lineitem"
            directory.mkdir()
            paths = []
            for part in (1, 2, 10):
                path = directory / f"lineitem.{part}.parquet"
                path.write_bytes(bytes([part]))
                paths.append(path)
            manifest = {
                "format_version": 1,
                "scale_factor": 1000,
                "tables": {
                    "lineitem": {
                        "layout": "partitioned",
                        "read_policy": "sharded",
                        "schema": [],
                        "schema_fingerprint": self.schema_fingerprint([]),
                        "rows": 3,
                        "bytes": 3,
                        "part_count": 3,
                        "parts": [
                            {
                                "id": index,
                                "path": str(path.relative_to(root)),
                                "rows": 1,
                                "bytes": 1,
                                "row_groups": 1,
                                "sha256": "0" * 64,
                            }
                            for index, path in enumerate(paths, 1)
                        ],
                    },
                },
            }
            self.write_completed_manifest(root, manifest)
            self.assertEqual(
                resolve_table_files(root, "lineitem", 1000),
                [str(path.resolve()) for path in paths])

    def test_unexpected_part_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "orders"
            directory.mkdir()
            declared = directory / "orders.1.parquet"
            declared.write_bytes(b"x")
            manifest = {
                "format_version": 1,
                "scale_factor": 1,
                "tables": {
                    "orders": {
                        "layout": "partitioned",
                        "read_policy": "sharded",
                        "schema": [],
                        "schema_fingerprint": self.schema_fingerprint([]),
                        "rows": 1,
                        "bytes": 1,
                        "part_count": 1,
                        "parts": [{
                            "id": 1,
                            "path": "orders/orders.1.parquet",
                            "rows": 1,
                            "bytes": 1,
                            "row_groups": 1,
                            "sha256": "0" * 64,
                        }],
                    },
                },
            }
            self.write_completed_manifest(root, manifest)
            (directory / "orders.2.parquet").write_bytes(b"stale")
            with self.assertRaisesRegex(
                    DatasetLayoutError, "does not exactly match manifest"):
                resolve_table_files(root, "orders", 1)

    def test_completion_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(
                '{"format_version":1,"scale_factor":1,"tables":{}}',
                encoding="utf-8")
            (root / "_SUCCESS").write_text(
                "manifest_sha256=bad\n", encoding="utf-8")
            with self.assertRaisesRegex(
                    DatasetLayoutError, "does not match"):
                resolve_table_files(root, "region", 1)

    def test_schema_fingerprint_must_match_schema_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "region.parquet"
            path.write_bytes(b"x")
            manifest = {
                "format_version": 1,
                "scale_factor": 1,
                "tables": {
                    "region": {
                        "layout": "single",
                        "read_policy": "replicated",
                        "schema": [{
                            "name": "r_regionkey",
                            "nullable": False,
                            "type": "int64",
                        }],
                        "schema_fingerprint": "0" * 64,
                        "rows": 1,
                        "bytes": 1,
                        "file": {
                            "path": "region.parquet",
                            "rows": 1,
                            "bytes": 1,
                            "row_groups": 1,
                            "sha256": "0" * 64,
                        },
                    },
                },
            }
            self.write_completed_manifest(root, manifest)
            with self.assertRaisesRegex(
                    DatasetLayoutError,
                    "schema_fingerprint does not match schema"):
                resolve_table_files(root, "region", 1)


if __name__ == "__main__":
    unittest.main()
