# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

import ncu_profile


class NativeSectionTests(unittest.TestCase):
    def test_parses_literal_kernel_substrings_and_builds_escaped_regex(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "kernels.txt"
            path.write_text("# selected kernels\nfoo.bar\n\na+b\n")
            kernels = ncu_profile.parse_kernel_substrings(path)
        self.assertEqual(kernels, ("foo.bar", "a+b"))
        self.assertEqual(
            ncu_profile.kernel_name_expression(kernels),
            r"regex:^(?!.*[Nn][Cc][Cc][Ll]).*(foo\.bar|a\+b).*$",
        )

    def test_section_arguments_pass_one_native_file_to_ncu(self) -> None:
        self.assertEqual(
            ncu_profile.section_arguments(Path("/repo/ncu_metrics.section")),
            [
                "--section-folder",
                "/repo",
                "--section",
                "AlpsTpchMetrics",
            ],
        )

    def test_queries_counter_selectors_and_launch_metrics_from_ncu(self) -> None:
        completed = {
            "argv": ["ncu"],
            "exit_code": 0,
            "timed_out": False,
            "elapsed_seconds": 0.1,
            "output": "\n".join(
                (
                    "regex:^dram__bytes[.](sum|min|max|avg)$",
                    " -regex:^dram__bytes[.](sum|min|max|avg)$",
                    "launch__block_size",
                    " -launch__block_size",
                )
            ),
        }
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(
            ncu_profile, "run_bounded", return_value=completed
        ):
            requested, launch = ncu_profile.query_section_metrics(
                Path("/repo/ncu_metrics.section"), "ncu", 10
            )
        self.assertEqual(
            requested, ("regex:^dram__bytes[.](sum|min|max|avg)$",)
        )
        self.assertEqual(launch, ("launch__block_size",))

    def test_selector_matches_expanded_metric(self) -> None:
        self.assertTrue(
            ncu_profile.selector_matches(
                "regex:^dram__bytes[.](sum|min|max|avg)$", "dram__bytes.sum"
            )
        )
        self.assertFalse(
            ncu_profile.selector_matches(
                "regex:^dram__bytes[.](sum|min|max|avg)$",
                "dram__bytes.sum.per_second",
            )
        )


class ReportValidationTests(unittest.TestCase):
    def test_validates_requested_and_launch_metrics_and_rejects_nccl(self) -> None:
        csv_output = "\n".join(
            (
                "==PROF== Connected",
                '"Kernel Name","Metric Name","Metric Value"',
                '"local_kernel","dram__bytes.sum","4"',
                '"local_kernel","launch__block_size","256"',
            )
        )
        completed = {
            "argv": ["ncu"],
            "exit_code": 0,
            "timed_out": False,
            "elapsed_seconds": 0.1,
            "output": csv_output,
        }
        with mock.patch.object(ncu_profile, "run_bounded", return_value=completed):
            result, output = ncu_profile.inspect_report(
                Path("capture.ncu-rep"),
                ("regex:^dram__bytes[.](sum|min|max|avg)$",),
                ("launch__block_size",),
                10,
            )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(output, csv_output)

        with mock.patch.object(ncu_profile, "run_bounded", return_value=completed):
            result, _ = ncu_profile.inspect_report(
                Path("capture.ncu-rep"),
                ("regex:^dram__bytes[.](sum|min|max|avg)$",),
                ("launch__block_size",),
                10,
                kernel_substrings=("wanted_kernel",),
            )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(
            result["reason"], "report contains kernels outside the selected subset"
        )
        self.assertEqual(result["unexpected_kernels"], ["local_kernel"])

        nccl_output = csv_output.replace("local_kernel", "ncclDevKernel")
        completed["output"] = nccl_output
        with mock.patch.object(ncu_profile, "run_bounded", return_value=completed):
            result, _ = ncu_profile.inspect_report(
                Path("capture.ncu-rep"),
                ("regex:^dram__bytes[.](sum|min|max|avg)$",),
                ("launch__block_size",),
                10,
            )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["reason"], "report contains excluded NCCL kernels")

    def test_cli_failure_is_json_serializable(self) -> None:
        with mock.patch.object(
            ncu_profile,
            "run_preflight",
            side_effect=ncu_profile.NcuProfileError("blocked"),
        ), mock.patch(
            "sys.argv",
            [
                "ncu_profile.py",
                "--probe",
                "/probe",
                "--section-file",
                "/repo/ncu_metrics.section",
            ],
        ), mock.patch("builtins.print") as printed:
            self.assertEqual(ncu_profile.main(), 1)
        json.loads(printed.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
