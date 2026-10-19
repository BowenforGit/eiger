#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.


import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
DRIVER = SCRIPT_DIR / "generate_tpch_multifile.py"


class SequentialGenerationDriverTest(unittest.TestCase):
    def make_fake_generator(self, root: Path) -> Path:
        generator = root / "fake-tpchgen"
        generator.write_text(textwrap.dedent("""\
            #!/usr/bin/env python3
            import pathlib
            import sys
            if "--version" in sys.argv:
                print("fake-tpchgen 1.0")
                raise SystemExit
            def value(name):
                return sys.argv[sys.argv.index(name) + 1]
            output = pathlib.Path(value("--output-dir"))
            table = value("--tables")
            if "--part" in sys.argv:
                part = value("--part")
                path = output / table / f"{table}.{part}.parquet"
            else:
                path = output / f"{table}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{table}:{path.name}".encode())
            with (output / "invocations.log").open("a") as log:
                log.write(" ".join(sys.argv[1:]) + "\\n")
            """), encoding="utf-8")
        generator.chmod(0o755)
        return generator

    def run_driver(
        self,
        generator: Path,
        staging: Path,
        *extra: str,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable, str(DRIVER),
                "--scale", "1",
                "--parts", "3",
                "--tables", "lineitem",
                "--generator", str(generator),
                "--staging-dir", str(staging),
                *extra,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_stop_and_resume_never_regenerates_completed_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = self.make_fake_generator(root)
            staging = root / "sf1.staging"
            self.run_driver(generator, staging, "--stop-after", "2")
            state = json.loads(
                (staging / ".generation_state.json").read_text())
            self.assertEqual(len(state["completed"]), 2)

            self.run_driver(generator, staging)
            invocations = (
                staging / "invocations.log").read_text().splitlines()
            self.assertEqual(len(invocations), 3)
            self.assertTrue(all("--part" in line for line in invocations))
            self.assertEqual(
                sorted((staging / "lineitem").glob("*.parquet")),
                [
                    staging / "lineitem" / "lineitem.1.parquet",
                    staging / "lineitem" / "lineitem.2.parquet",
                    staging / "lineitem" / "lineitem.3.parquet",
                ])

    def test_refuses_to_overwrite_untracked_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = self.make_fake_generator(root)
            staging = root / "sf1.staging"
            path = staging / "lineitem" / "lineitem.1.parquet"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"existing")
            result = subprocess.run(
                [
                    sys.executable, str(DRIVER),
                    "--scale", "1",
                    "--parts", "3",
                    "--tables", "lineitem",
                    "--generator", str(generator),
                    "--staging-dir", str(staging),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual(path.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
