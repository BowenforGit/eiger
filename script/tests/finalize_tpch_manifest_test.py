#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.


import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
FINALIZER = SCRIPT_DIR / "finalize_tpch_manifest.py"


class FinalizeTpchManifestTest(unittest.TestCase):
    def make_incomplete_staging(self, root: Path) -> tuple[Path, Path]:
        staging = root / "sf1.staging"
        staging.mkdir()
        (staging / ".generation_state.json").write_text(
            json.dumps({}), encoding="utf-8")
        metadata_tool = root / "fake-metadata-tool"
        metadata_tool.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        metadata_tool.chmod(0o755)
        return staging, metadata_tool

    def run_finalizer(
        self,
        staging: Path,
        metadata_tool: Path,
        publish: Path,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(FINALIZER),
                "--staging-dir", str(staging),
                "--metadata-tool", str(metadata_tool),
                "--publish-dir", str(publish),
            ],
            capture_output=True,
            text=True,
        )

    def assert_staging_not_completed(self, staging: Path) -> None:
        self.assertFalse((staging / "manifest.json").exists())
        self.assertFalse((staging / "_SUCCESS").exists())

    def test_existing_publish_target_fails_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging, metadata_tool = self.make_incomplete_staging(root)
            publish = root / "sf1"
            publish.mkdir()

            result = self.run_finalizer(staging, metadata_tool, publish)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite publish path", result.stderr)
            self.assert_staging_not_completed(staging)

    def test_missing_publish_parent_fails_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging, metadata_tool = self.make_incomplete_staging(root)
            publish = root / "missing" / "sf1"

            result = self.run_finalizer(staging, metadata_tool, publish)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "publish parent is not a directory", result.stderr)
            self.assert_staging_not_completed(staging)

    def test_cross_filesystem_publish_fails_before_completion(self) -> None:
        shared_memory = Path("/dev/shm")
        if not shared_memory.is_dir():
            self.skipTest("/dev/shm is unavailable")
        with tempfile.TemporaryDirectory() as temporary, \
                tempfile.TemporaryDirectory(dir=shared_memory) as publish_parent:
            root = Path(temporary)
            staging, metadata_tool = self.make_incomplete_staging(root)
            parent = Path(publish_parent)
            if staging.stat().st_dev == parent.stat().st_dev:
                self.skipTest("test paths are on the same filesystem")

            result = self.run_finalizer(
                staging, metadata_tool, parent / "sf1")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different filesystems", result.stderr)
            self.assert_staging_not_completed(staging)


if __name__ == "__main__":
    unittest.main()
