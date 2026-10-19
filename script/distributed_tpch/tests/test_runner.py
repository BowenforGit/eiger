# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from subprocess import CompletedProcess
from unittest import mock

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

import artifacts
import definitions
import nccl_config_preflight
import nsys_preflight
import run
import topology


class ParsingTests(unittest.TestCase):
    def test_parquet_decode_threads_must_be_positive(self) -> None:
        parser = run.build_parser()
        required = [
            "new",
            "/campaign",
            "--scale",
            "1",
            "--arms",
            "legacy",
            "--stages",
            "memory",
            "--parquet-decode-threads",
        ]
        self.assertEqual(parser.parse_args([*required, "4"]).parquet_decode_threads, 4)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([*required, "0"])

    def test_query_ranges_are_sorted_and_deduplicated(self) -> None:
        self.assertEqual(run.parse_queries("9,1-3,2"), (1, 2, 3, 9))

    def test_query_range_rejects_out_of_range_id(self) -> None:
        with self.assertRaises(run.ConfigurationError):
            run.parse_queries("1,23")

    def test_query_memory_reserve_overrides_accept_q_prefix(self) -> None:
        self.assertEqual(
            run.parse_query_memory_reserve_bytes(
                ["Q1=12000000000", "9=58000000000"], (1, 8, 9)
            ),
            {1: 12_000_000_000, 9: 58_000_000_000},
        )

    def test_query_memory_reserve_override_rejects_unselected_query(self) -> None:
        with self.assertRaisesRegex(run.ConfigurationError, "not in --queries"):
            run.parse_query_memory_reserve_bytes(["Q9=58000000000"], (1, 8))

    def test_query_memory_reserve_override_rejects_duplicate_query(self) -> None:
        with self.assertRaisesRegex(run.ConfigurationError, "duplicate"):
            run.parse_query_memory_reserve_bytes(
                ["Q9=58000000000", "9=59000000000"], (9,)
            )

    def test_explicit_host_slots_are_uniform(self) -> None:
        resolved = topology.resolve_topology(
            "node-a:4,node-b:4", None, 2222, Path("unused")
        )
        self.assertEqual(resolved.ranks, 8)
        self.assertEqual(resolved.host_spec, "node-a:4,node-b:4")
        self.assertFalse(resolved.local)

    def test_local_topology_uses_visible_gpu_probe(self) -> None:
        with mock.patch.object(topology, "visible_gpu_count", return_value=8):
            resolved = topology.resolve_topology(None, None, 2222, Path("probe"))
        self.assertTrue(resolved.local)
        self.assertEqual(resolved.ranks, 8)


class InputResolutionTests(unittest.TestCase):
    def test_data_root_accepts_single_and_multifile_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scale = root / "sf7"
            scale.mkdir()
            for index, table in enumerate(run.TPC_H_TABLES):
                if index % 2:
                    (scale / table).mkdir()
                else:
                    (scale / f"{table}.parquet").touch()
            golden = scale / "golden"
            golden.mkdir()
            (golden / "q01.parquet").touch()
            self.assertEqual(run.resolve_data_root(root, 7, (1,)), root.resolve())

    def test_unique_build_auto_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            build = repository / "build-release"
            for relative in run.BUILD_BINARIES.values():
                binary = build / relative
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.touch(mode=0o755)
            with mock.patch.object(run, "REPOSITORY", repository), mock.patch.dict(
                os.environ, {}, clear=True
            ):
                self.assertEqual(
                    run.resolve_build_dir(None, ("synthesis", "timing")),
                    build.resolve(),
                )

    def test_auto_detected_build_must_have_executable_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            build = repository / "build-release"
            for relative in run.BUILD_BINARIES.values():
                binary = build / relative
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.touch(mode=0o755)
            (build / run.BUILD_BINARIES["benchmark"]).chmod(0o644)
            with mock.patch.object(run, "REPOSITORY", repository), mock.patch.dict(
                os.environ, {}, clear=True
            ):
                with self.assertRaises(run.ConfigurationError):
                    run.resolve_build_dir(None, ("timing",))

    def test_synthesized_arm_requires_plans_or_synthesis(self) -> None:
        with self.assertRaises(run.ConfigurationError):
            run.validate_plan_configuration(("synthesized",), ("timing",), None, (1,))

    def test_supplied_plans_conflict_with_synthesis_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(run.ConfigurationError):
                run.validate_plan_configuration(
                    ("synthesized",),
                    ("synthesis", "timing"),
                    Path(temporary),
                    (1,),
                )

    def test_synthesized_cta_variant_uses_shared_synthesis_stage(self) -> None:
        run.validate_plan_configuration(
            ("synthesized_cta32",), ("synthesis", "timing"), None, (1,)
        )

    def test_messages_stage_requires_only_legacy_arm(self) -> None:
        run.validate_plan_configuration(
            ("legacy",), ("messages",), None, (1,)
        )
        with self.assertRaisesRegex(
            run.ConfigurationError, "requires --arms legacy"
        ):
            run.validate_plan_configuration(
                ("legacy", "synthesized"),
                ("messages", "synthesis"),
                None,
                (1,),
            )


class ArmDefinitionTests(unittest.TestCase):
    def test_configured_nccl_bases_have_all_fixed_cta_variants(self) -> None:
        self.assertNotIn("all_co_global_ce_pull", definitions.ARMS)
        for base in definitions.NCCL_CTA_VARIANT_BASES:
            for ctas in definitions.NCCL_CTA_CHOICES:
                name = f"{base}_cta{ctas}"
                arm = definitions.ARMS[name]
                self.assertEqual(arm.variant_of, base)
                self.assertEqual(
                    arm.benchmark_args, definitions.ARMS[base].benchmark_args
                )
                self.assertEqual(
                    dict(arm.environment_overrides),
                    {"NCCL_MIN_CTAS": str(ctas), "NCCL_MAX_CTAS": str(ctas)},
                )
        self.assertFalse(
            any(name.startswith("all_co_ce_pull_cta") for name in definitions.ARMS)
        )
        self.assertFalse(
            any(name.startswith("legacy_ce_pull_cta") for name in definitions.ARMS)
        )

    def test_operator_isolated_arms_have_no_cta_variants(self) -> None:
        for name in (
            "group_co_nccl",
            "group_co_ce_pull",
            "semi_co_nccl",
            "semi_co_ce_pull",
        ):
            self.assertEqual(
                definitions.ARMS[name].uses_ce_pull, name.endswith("ce_pull")
            )
            self.assertFalse(
                any(
                    candidate.startswith(f"{name}_cta")
                    for candidate in definitions.ARMS
                )
            )


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = Namespace(
            scale=300,
            synthesis_attempt_timeout=100,
            synthesis_max_restarts=2,
            synthesis_seed=7,
            synthesis_arg=[],
            synthesis_continue_on_verify_failure=False,
            memory_reserve_bytes=32_000_000_000,
            parquet_decode_threads=4,
            nccl_socket_ifname="eth",
            mpirun="mpirun",
            profile_gpu_index=5,
            profile_metrics_rank=0,
            profile_gpu_metrics_frequency=10000,
            ncu_metrics=["dram__bytes.sum"],
            ncu_launch_metrics=["launch__block_size"],
            ncu_metrics_file=Path("/repo/ncu_metrics.section"),
            ncu_kernels=["selected.kernel"],
            ncu_kernel_expression=(
                r"regex:^(?!.*[Nn][Cc][Cc][Ll]).*(selected\.kernel).*$"
            ),
            ncu_capture_timeout=600,
        )
        self.topology = topology.Topology((topology.Host("local", 8),), True, 2222)

    def test_synthesized_profile_passes_per_query_config_to_binary(self) -> None:
        command = run.benchmark_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            "synthesized",
            "profiles",
            9,
            Path("/out"),
            Path("/plans"),
        )
        separator = command.index("--")
        self.assertNotIn("--physical-plan-dir", command[:separator])
        self.assertEqual(
            command[command.index("--physical-plan-config") + 1],
            "/plans/q09/plan.json",
        )
        self.assertEqual(command[command.index("--gpu-metrics-devices") + 1], "5")

    def test_synthesis_uses_per_query_directory_layout(self) -> None:
        command = run.synthesis_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            9,
            Path("/campaign/arms/synthesized/synthesis"),
        )
        self.assertIn("--query-dir-layout", command)
        separator = command.index("--")
        self.assertEqual(command[separator + 1 :], ["--parquet-decode-threads", "4"])

    def test_synthesis_continue_on_verify_failure_is_forwarded(self) -> None:
        self.args.synthesis_continue_on_verify_failure = True
        command = run.synthesis_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            3,
            Path("/campaign/arms/synthesized/synthesis"),
        )
        separator = command.index("--")
        self.assertIn(
            "--continue-on-verification-failure", command[separator + 1 :]
        )

    def test_benchmark_uses_campaign_parquet_decode_threads(self) -> None:
        command = run.benchmark_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            "legacy",
            "memory",
            1,
            Path("/attempt"),
            None,
        )
        self.assertEqual(command[command.index("--parquet-decode-threads") + 1], "4")

    def test_messages_stage_uses_attempt_local_csv(self) -> None:
        command = run.benchmark_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            "legacy",
            "messages",
            1,
            Path("/attempt"),
            None,
        )
        self.assertIn("--messages-only", command)
        self.assertEqual(command[command.index("--message-iterations") + 1], "1")
        self.assertEqual(
            command[command.index("--messages-csv") + 1],
            "/attempt/messages.csv",
        )
        self.assertNotIn("--timing-only", command)
        self.assertNotIn("--steps-only", command)

    def test_retry_synthesis_forces_existing_plan_rerun(self) -> None:
        self.args.command = "retry"
        command = run.synthesis_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            21,
            Path("/campaign/synthesis"),
        )
        self.assertIn("--force", command)

    def test_ncu_profile_is_rank0_scoped_and_metric_limited(self) -> None:
        command = run.benchmark_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            "legacy",
            "ncu_profiles",
            3,
            Path("/out"),
            None,
        )
        self.assertEqual(command[:4], ["timeout", "--signal=TERM", "--kill-after=30", "600"])
        self.assertIn("profile_distributed_tpch_ncu.sh", command[4])
        self.assertEqual(
            command[command.index("--section-file") + 1],
            "/repo/ncu_metrics.section",
        )
        self.assertEqual(
            command[command.index("--kernel-name") + 1],
            self.args.ncu_kernel_expression,
        )
        self.assertNotIn("--metrics", command)
        benchmark_args = command[command.index("--") + 1 :]
        self.assertIn("--warmup-min", benchmark_args)
        self.assertNotIn("--timing-only", benchmark_args)

    def test_combined_timing_breakdown_uses_both_clean_sample_sets(self) -> None:
        command = run.combined_timing_breakdown_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            "legacy",
            9,
            Path("/campaign/arms/legacy/timing/q09/attempt-0001"),
            Path("/campaign/arms/legacy/breakdown/q09/attempt-0001"),
            None,
        )
        self.assertNotIn("--timing-only", command)
        self.assertNotIn("--steps-only", command)
        self.assertEqual(command[command.index("--iterations") + 1], "10")
        self.assertEqual(command[command.index("--step-iterations") + 1], "5")
        self.assertEqual(
            command[command.index("--steps-csv") + 1],
            "/campaign/arms/legacy/breakdown/q09/attempt-0001/steps.csv",
        )

    def test_collect_on_verify_failure_is_forwarded_with_attempt_sidecar(self) -> None:
        self.args.collect_on_verify_failure = True
        command = run.benchmark_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            "legacy",
            "memory",
            3,
            Path("/campaign/arms/legacy/memory/q03/attempt-0001"),
            None,
        )
        self.assertIn("--collect-on-verify-failure", command)
        self.assertEqual(
            command[command.index("--verification-json") + 1],
            "/campaign/arms/legacy/memory/q03/attempt-0001/verification.json",
        )

    def test_ce_pull_memory_uses_attempt_local_csvs(self) -> None:
        command = run.benchmark_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            "all_co_ce_pull",
            "memory",
            1,
            Path("/attempt"),
            None,
        )
        self.assertIn("/attempt/memory.csv", command)
        self.assertIn("/attempt/ce_pull.csv", command)

    def test_group_only_ce_arm_keeps_other_operators_legacy(self) -> None:
        command = run.benchmark_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            "group_co_ce_pull",
            "timing",
            1,
            Path("/attempt"),
            None,
        )
        self.assertEqual(command[command.index("--eq-join") + 1], "legacy")
        self.assertEqual(command[command.index("--group-by") + 1], "column-overlap")
        self.assertEqual(command[command.index("--semi-join") + 1], "legacy")
        self.assertIn("/attempt/ce_pull.csv", command)

    def test_legacy_ce_arm_selects_the_global_backend(self) -> None:
        command = run.benchmark_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            "legacy_ce_pull",
            "timing",
            1,
            Path("/attempt"),
            None,
        )
        self.assertEqual(
            command[command.index("--communication-backend") + 1], "ce-pull"
        )
        self.assertNotIn("--column-exchange-backend", command)
        self.assertIn("/attempt/ce_pull.csv", command)

    def test_global_ce_column_overlap_arm_inherits_the_backend(self) -> None:
        command = run.benchmark_command(
            self.args,
            Path("/build"),
            Path("/data"),
            self.topology,
            "all_co_ce_pull",
            "timing",
            1,
            Path("/attempt"),
            None,
        )
        self.assertEqual(
            command[command.index("--communication-backend") + 1], "ce-pull"
        )
        self.assertNotIn("--column-exchange-backend", command)
        self.assertIn("/attempt/ce_pull.csv", command)

    def test_multinode_profile_selects_local_rank_metrics(self) -> None:
        multi = topology.Topology(
            (topology.Host("a", 4), topology.Host("b", 4)), False, 2222
        )
        command = run.benchmark_command(
            self.args,
            Path("/build"),
            Path("/data"),
            multi,
            "legacy",
            "profiles",
            1,
            Path("/attempt"),
            None,
        )
        self.assertNotIn("--single-report", command)
        self.assertEqual(command[command.index("--gpu-metrics-devices") + 1], "5")
        self.assertEqual(command[command.index("--gpu-metrics-rank") + 1], "0")

    def test_cuda_visible_devices_is_recorded(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,1"}, clear=False):
            _, selected = run.command_environment(self.args, Path("/data"))
        self.assertEqual(selected["CUDA_VISIBLE_DEVICES"], "3,1")

    def test_query_environment_overrides_only_selected_query(self) -> None:
        self.args.query_memory_reserve_bytes = {9: 58_000_000_000}
        inherited = {"KEEP": "value", "GAMAX_MEM_RESERVE_BYTES": "32000000000"}
        q8_environment, _ = run.query_environment(inherited, inherited, self.args, 8)
        q9_environment, q9_selected = run.query_environment(
            inherited, inherited, self.args, 9
        )
        self.assertEqual(q8_environment["GAMAX_MEM_RESERVE_BYTES"], "32000000000")
        self.assertEqual(q9_environment["GAMAX_MEM_RESERVE_BYTES"], "58000000000")
        self.assertEqual(q9_selected["GAMAX_MEM_RESERVE_BYTES"], "58000000000")
        self.assertEqual(q9_environment["KEEP"], "value")
        self.assertEqual(inherited["GAMAX_MEM_RESERVE_BYTES"], "32000000000")

    def test_baseline_arm_removes_all_inherited_cta_controls(self) -> None:
        inherited = {key: "17" for key in run.NCCL_CTA_ENVIRONMENT_KEYS}
        environment, selected = run.arm_environment(inherited, inherited, "legacy")
        for key in run.NCCL_CTA_ENVIRONMENT_KEYS:
            self.assertNotIn(key, environment)
            self.assertEqual(selected[key], "<unset>")

    def test_fixed_cta_arm_sets_equal_bounds_and_removes_aliases(self) -> None:
        inherited = {key: "17" for key in run.NCCL_CTA_ENVIRONMENT_KEYS}
        environment, selected = run.arm_environment(
            inherited, inherited, "all_co_nccl_cta64"
        )
        self.assertEqual(environment["NCCL_MIN_CTAS"], "64")
        self.assertEqual(environment["NCCL_MAX_CTAS"], "64")
        self.assertEqual(selected["NCCL_MIN_CTAS"], "64")
        self.assertEqual(selected["NCCL_MAX_CTAS"], "64")
        for key in run.NCCL_CTA_ENVIRONMENT_KEYS[2:]:
            self.assertNotIn(key, environment)
            self.assertEqual(selected[key], "<unset>")

    def test_parser_finds_only_controlled_configuration_entries(self) -> None:
        entries = nccl_config_preflight.parse_controlled_entries(
            "# comment\nNCCL_SOCKET_IFNAME=eth\nNCCL_MIN_CTAS=8\n"
            "NCCL_MAX_NCHANNELS = 32\n"
        )
        self.assertEqual(
            entries,
            {"NCCL_MIN_CTAS": "8", "NCCL_MAX_NCHANNELS": "32"},
        )

    def test_configuration_file_cta_control_is_a_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "nccl.conf"
            config.write_text("NCCL_MAX_CTAS=32\n")
            report = nccl_config_preflight.inspect_configuration(
                {"NCCL_CONF_FILE": str(config), "NCCL_ENV_PLUGIN": "none"},
                root,
            )
        self.assertTrue(report["conflicts"])
        file_records = report["configuration_files"]
        self.assertEqual(
            file_records[-1]["controlled_entries"], {"NCCL_MAX_CTAS": "32"}
        )

    def test_inherited_cta_is_recorded_but_not_a_configuration_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = nccl_config_preflight.inspect_configuration(
                {"NCCL_MIN_CTAS": "16", "NCCL_ENV_PLUGIN": "none"},
                Path(temporary),
            )
        self.assertEqual(report["observed_environment"], {"NCCL_MIN_CTAS": "16"})
        self.assertEqual(report["conflicts"], [])

    def test_explicit_environment_plugin_is_a_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = nccl_config_preflight.inspect_configuration(
                {"NCCL_ENV_PLUGIN": "custom"}, Path(temporary)
            )
        self.assertTrue(
            any("NCCL_ENV_PLUGIN" in conflict for conflict in report["conflicts"])
        )

    def test_plugin_in_rank_library_directory_is_a_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library_dir = root / "rank-libs"
            library_dir.mkdir()
            plugin = library_dir / "libnccl-env.so"
            plugin.touch()
            report = nccl_config_preflight.inspect_configuration(
                {}, root, library_directories=(library_dir,)
            )
        self.assertEqual(report["environment_plugin"]["resolved"], str(plugin))
        self.assertTrue(report["conflicts"])

    def test_remote_configuration_does_not_inspect_unused_coordinator(self) -> None:
        remote = topology.Topology((topology.Host("node-a", 4),), False, 2222)
        report = {
            "host": "node-a",
            "observed_environment": {},
            "configuration_files": [],
            "environment_plugin": {"selection": "<unset>", "resolved": None},
            "conflicts": [],
        }
        completed = CompletedProcess([], 0, json.dumps(report) + "\n", "")
        with mock.patch.object(
            run,
            "inspect_configuration",
            side_effect=AssertionError("coordinator must not be inspected"),
        ), mock.patch.object(run, "run_remote", return_value=completed) as remote_call:
            sources, observed = run.capture_nccl_configuration(remote, Path("/build"))
        self.assertEqual(tuple(sources), ("node-a",))
        self.assertEqual(observed, {"node-a": {}})
        helper_args = remote_call.call_args.args[2]
        self.assertIn("--library-dir", helper_args)
        self.assertIn("/build/src/cuda/tpch", helper_args)

    def test_shell_helper_forwards_fixed_ctas_and_unsets_aliases(self) -> None:
        script = """
source script/distributed_tpch_common.sh
mpi=(mpirun)
prefix=()
distributed_tpch_configure_nccl_cta_environment mpi prefix
printf '%s\\n' "${mpi[*]}" "${prefix[*]}"
"""
        result = subprocess.run(
            ["bash", "-c", script],
            cwd=run.REPOSITORY,
            env={
                "PATH": os.environ["PATH"],
                "NCCL_MIN_CTAS": "32",
                "NCCL_MAX_CTAS": "32",
                "NCCL_MIN_NCHANNELS": "64",
            },
            check=True,
            capture_output=True,
            text=True,
        )
        mpi, prefix = result.stdout.splitlines()
        self.assertEqual(mpi, "mpirun -x NCCL_MIN_CTAS -x NCCL_MAX_CTAS")
        self.assertNotIn("-u NCCL_MIN_CTAS", prefix)
        self.assertIn("-u NCCL_MIN_NCHANNELS", prefix)

    def test_shell_helper_unsets_ctas_for_automatic_baseline(self) -> None:
        script = """
source script/distributed_tpch_common.sh
mpi=(mpirun)
prefix=()
distributed_tpch_configure_nccl_cta_environment mpi prefix
printf '%s\\n' "${mpi[*]}" "${prefix[*]}"
"""
        result = subprocess.run(
            ["bash", "-c", script],
            cwd=run.REPOSITORY,
            env={"PATH": os.environ["PATH"]},
            check=True,
            capture_output=True,
            text=True,
        )
        mpi, prefix = result.stdout.splitlines()
        self.assertEqual(mpi, "mpirun")
        for key in run.NCCL_CTA_ENVIRONMENT_KEYS:
            self.assertIn(f"-u {key}", prefix)

    def test_remote_topology_does_not_check_coordinator_gpus(self) -> None:
        remote = topology.Topology((topology.Host("node-a", 4),), False, 2222)
        with mock.patch.object(
            run, "active_gpu_processes", side_effect=AssertionError("must not run")
        ):
            run.ensure_selected_gpus_idle(remote)

    def test_profiles_require_remote_rank_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = {}
            for name in ("device_probe", "benchmark"):
                path = root / name
                path.touch(mode=0o755)
                binaries[name] = path
            launcher = root / "profile_rank.sh"
            launcher.touch(mode=0o755)
            required = run.required_remote_executables(
                binaries,
                ("device_probe", "benchmark"),
                ("profiles",),
                {"ALPS_TPCH_PROFILE_RANK_LAUNCHER": str(launcher)},
            )
            self.assertIn(launcher, required)
            self.assertEqual(required[-1].name, "nsys_preflight.py")

    def test_ce_pull_non_profile_requires_telemetry(self) -> None:
        expected = run.expected_artifacts(
            "all_co_ce_pull", "timing", 1, Path("/attempt"), self.topology
        )
        self.assertEqual(expected, ["e2e.csv", "ce_pull.csv"])
        isolated = run.expected_artifacts(
            "semi_co_ce_pull", "timing", 20, Path("/attempt"), self.topology
        )
        self.assertEqual(isolated, ["e2e.csv", "ce_pull.csv"])

    def test_messages_stage_requires_messages_csv(self) -> None:
        self.assertEqual(
            run.expected_artifacts(
                "legacy", "messages", 1, Path("/attempt"), self.topology
            ),
            ["messages.csv"],
        )

    def test_ncu_stage_requires_one_rank0_report_and_derived_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            missing = run.expected_artifacts(
                "legacy", "ncu_profiles", 3, attempt, self.topology
            )
            self.assertEqual(missing, ["__missing_rank0_ncu_profile__"])
            (attempt / "q03_node-a_rank0.ncu-rep").write_text("report")
            self.assertEqual(
                run.expected_artifacts(
                    "legacy", "ncu_profiles", 3, attempt, self.topology
                ),
                [
                    "q03_node-a_rank0.ncu-rep",
                    "q03_ncu_metrics.csv",
                    "q03_ncu_validation.json",
                ],
            )


class RemotePreflightTests(unittest.TestCase):
    def test_remote_preflight_checks_every_table_layout(self) -> None:
        remote_topology = topology.Topology((topology.Host("node-a", 4),), False, 2222)
        calls: list[list[str]] = []

        def fake_remote(host: str, port: int, argv: list[str]):
            calls.append(argv)
            if "--print-visible-device-count" in argv:
                output = "4\n"
            elif argv and argv[0] == "sha256sum":
                output = "hash  /build/benchmark\n"
            elif "--query-compute-apps=gpu_uuid,pid,process_name" in argv:
                output = ""
            elif argv and argv[0] == "uname":
                output = "Linux node-a"
            else:
                output = "gpu inventory"
            return CompletedProcess(argv, 0, output, "")

        with mock.patch.object(topology, "run_remote", side_effect=fake_remote):
            topology.preflight_remote_hosts(
                remote_topology,
                Path("/build/probe"),
                (Path("/build/benchmark"),),
                Path("/data"),
                300,
                (1,),
                run.TPC_H_TABLES,
                Path("/output"),
                False,
            )
        predicate_command = calls[0][2]
        for table in run.TPC_H_TABLES:
            self.assertIn(f"/data/sf300/{table}.parquet", predicate_command)
            self.assertIn(f"/data/sf300/{table}", predicate_command)


class NsysPreflightTests(unittest.TestCase):
    def test_remote_preflight_resolves_candidates_on_remote_host(self) -> None:
        remote = topology.Topology((topology.Host("node-a", 4),), False, 2222)
        args = Namespace(
            profile_preflight_timeout=20,
            profile_preflight_seconds=2,
            profile_gpu_metrics_frequency=10000,
        )
        payload = json.dumps(
            {
                "status": "pass",
                "gpu_index": 2,
                "attempts": [
                    {
                        "gpu_index": 2,
                        "status": "pass",
                        "metric_samples": 10,
                    }
                ],
            }
        )
        completed = CompletedProcess([], 0, payload, "")
        with mock.patch.object(
            run, "physical_gpu_candidates", side_effect=AssertionError("local probe")
        ), mock.patch.object(run, "run_remote", return_value=completed) as remote_call:
            result = run.run_profile_gpu_preflight(
                args, remote, {"CUDA_VISIBLE_DEVICES": "<unset>"}
            )
        self.assertEqual(result["gpu_index"], 2)
        command = remote_call.call_args.args[2]
        self.assertIn("--expected-gpus", command)
        self.assertNotIn("--candidates", command)

    def test_metric_tables_must_have_actual_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "metrics.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE GPU_METRICS(value INTEGER)")
                connection.execute("CREATE TABLE TARGET_INFO_GPU_METRICS(name TEXT)")
            self.assertEqual(nsys_preflight.metric_row_counts(database), (0, 0))
            with sqlite3.connect(database) as connection:
                connection.execute("INSERT INTO GPU_METRICS VALUES (1)")
                connection.execute(
                    "INSERT INTO TARGET_INFO_GPU_METRICS VALUES ('SM Active')"
                )
            self.assertEqual(nsys_preflight.metric_row_counts(database), (1, 1))

    def test_selection_skips_failed_or_timed_out_gpu(self) -> None:
        def fake_probe(gpu_index, *unused):
            return {
                "gpu_index": gpu_index,
                "status": "pass" if gpu_index == 3 else "fail",
            }

        with mock.patch.object(nsys_preflight, "probe_gpu", side_effect=fake_probe):
            selected = nsys_preflight.select_gpu([1, 3, 5], 20, 2, 10000)
        self.assertEqual(selected["gpu_index"], 3)
        self.assertEqual([item["gpu_index"] for item in selected["attempts"]], [1, 3])

    def test_bounded_command_terminates_a_hang(self) -> None:
        result = nsys_preflight.run_bounded(["/bin/sleep", "30"], 1)
        self.assertTrue(result["timed_out"])
        self.assertLess(result["elapsed_seconds"], 5)


class CampaignStatusTests(unittest.TestCase):
    def create_campaign(
        self,
        root: Path,
        *,
        arms: tuple[str, ...] = ("legacy",),
        stages: tuple[str, ...] = ("timing",),
        queries: tuple[int, ...] = (1, 2),
    ) -> Path:
        campaign = root / "campaign"
        artifacts.create_campaign(
            campaign,
            {
                "schema_version": 4,
                "arms": list(arms),
                "stages": list(stages),
                "queries": list(queries),
            },
            False,
        )
        return campaign

    def run_status(self, campaign: Path) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = run.main(["status", str(campaign)])
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_complete_success_uses_latest_attempt_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.create_campaign(Path(temporary))
            artifacts.append_status(
                campaign, "legacy", "timing", 1, "fail", 1, Path("attempt-0001")
            )
            artifacts.append_status(
                campaign, "legacy", "timing", 1, "pass", 0, Path("attempt-0002")
            )
            artifacts.append_status(
                campaign,
                "legacy",
                "timing",
                2,
                "skip_pass",
                0,
                Path("attempt-0001"),
            )
            exit_code, stdout, stderr = self.run_status(campaign)
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Completion: COMPLETE", stdout)
        self.assertIn("Result: SUCCESS", stdout)
        self.assertIn("Passed: 2", stdout)
        self.assertIn("Failed: 0", stdout)
        self.assertNotIn("Failed cases:", stdout)

    def test_complete_failure_lists_latest_failed_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.create_campaign(Path(temporary), queries=(9,))
            attempt = campaign / "arms" / "legacy" / "timing" / "q09" / "attempt-0002"
            artifacts.append_status(
                campaign, "legacy", "timing", 9, "fail", 137, attempt
            )
            exit_code, stdout, stderr = self.run_status(campaign)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr, "")
        self.assertIn("Completion: COMPLETE", stdout)
        self.assertIn("Result: FAILURE", stdout)
        self.assertIn("legacy/timing/Q09 exit=137", stdout)
        self.assertIn(f"attempt={attempt}", stdout)

    def test_complete_blocked_campaign_lists_dependency_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.create_campaign(
                Path(temporary), arms=("synthesized",), queries=(21,)
            )
            artifacts.append_status(
                campaign, "synthesized", "timing", 21, "blocked", 66, None
            )
            exit_code, stdout, stderr = self.run_status(campaign)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr, "")
        self.assertIn("Completion: COMPLETE", stdout)
        self.assertIn("Result: FAILURE", stdout)
        self.assertIn("Failed: 0", stdout)
        self.assertIn("Blocked: 1", stdout)
        self.assertIn("synthesized/timing/Q21 missing synthesized plan", stdout)

    def test_incomplete_campaign_lists_missing_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.create_campaign(Path(temporary))
            artifacts.append_status(
                campaign, "legacy", "timing", 1, "pass", 0, Path("attempt-0001")
            )
            exit_code, stdout, stderr = self.run_status(campaign)
        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr, "")
        self.assertIn("Completion: INCOMPLETE", stdout)
        self.assertIn("Result: PENDING", stdout)
        self.assertIn("Missing: 1", stdout)
        self.assertIn("legacy/timing/Q02", stdout)

    def test_expected_matrix_counts_synthesis_only_once(self) -> None:
        configuration = {
            "schema_version": 4,
            "arms": ["legacy", "synthesized_cta32"],
            "stages": ["synthesis", "timing", "breakdown"],
            "queries": [1, 2],
        }
        cells = run.expected_campaign_cells(configuration)
        self.assertEqual(len(cells), 10)
        self.assertEqual(
            cells[:2],
            (("synthesized", "synthesis", 1), ("synthesized", "synthesis", 2)),
        )

    def test_invalid_campaign_returns_usage_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "missing"
            exit_code, stdout, stderr = self.run_status(campaign)
        self.assertEqual(exit_code, 64)
        self.assertEqual(stdout, "")
        self.assertIn("campaign directory does not exist", stderr)

    def test_malformed_status_journal_returns_usage_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.create_campaign(Path(temporary), queries=(1,))
            (campaign / "status.csv").write_text("wrong\n")
            exit_code, stdout, stderr = self.run_status(campaign)
        self.assertEqual(exit_code, 64)
        self.assertEqual(stdout, "")
        self.assertIn("status journal is missing columns", stderr)


class CampaignRetryTests(unittest.TestCase):
    def create_campaign(self, root: Path) -> Path:
        campaign = root / "campaign"
        artifacts.create_campaign(
            campaign,
            {
                "schema_version": 4,
                "arms": ["synthesized"],
                "stages": ["synthesis", "timing", "breakdown", "profiles"],
                "queries": [21],
                "policy": {"combine_timing_breakdown": True},
            },
            False,
        )
        return campaign

    def append(self, campaign: Path, stage: str, status: str) -> None:
        artifacts.append_status(
            campaign,
            "synthesized",
            stage,
            21,
            status,
            0 if status in run.SUCCESSFUL_CELL_STATUSES else 1,
            Path("attempt"),
        )

    def test_query_selector_retries_only_unsuccessful_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.create_campaign(Path(temporary))
            for stage in ("synthesis", "timing", "breakdown", "profiles"):
                self.append(campaign, stage, "fail")
            cells, reasons = run.select_retry_cells(
                campaign,
                artifacts.load_campaign(campaign)["configuration"],
                None,
                None,
                "21",
            )
        self.assertEqual(
            cells,
            (
                ("synthesized", "synthesis", 21),
                ("synthesized", "timing", 21),
                ("synthesized", "breakdown", 21),
                ("synthesized", "profiles", 21),
            ),
        )
        self.assertTrue(all(reasons[cell] == "fail" for cell in cells))

    def test_missing_plan_adds_synthesis_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.create_campaign(Path(temporary))
            self.append(campaign, "synthesis", "pass")
            self.append(campaign, "timing", "pass")
            self.append(campaign, "breakdown", "pass")
            self.append(campaign, "profiles", "fail")
            cells, reasons = run.select_retry_cells(
                campaign,
                artifacts.load_campaign(campaign)["configuration"],
                None,
                "profiles",
                "21",
            )
        self.assertEqual(
            cells,
            (
                ("synthesized", "synthesis", 21),
                ("synthesized", "profiles", 21),
            ),
        )
        self.assertEqual(
            reasons[("synthesized", "synthesis", 21)],
            "dependency: missing synthesized plan",
        )

    def test_existing_plan_does_not_reselect_passed_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.create_campaign(Path(temporary))
            plan = campaign / "arms" / "synthesized" / "synthesis" / "q21" / "plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text("{}\n")
            self.append(campaign, "synthesis", "pass")
            self.append(campaign, "timing", "pass")
            self.append(campaign, "breakdown", "pass")
            self.append(campaign, "profiles", "fail")
            cells, _ = run.select_retry_cells(
                campaign,
                artifacts.load_campaign(campaign)["configuration"],
                None,
                "profiles",
                "21",
            )
        self.assertEqual(cells, (("synthesized", "profiles", 21),))

    def test_combined_timing_breakdown_is_an_atomic_retry_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.create_campaign(Path(temporary))
            self.append(campaign, "synthesis", "pass")
            self.append(campaign, "timing", "fail")
            self.append(campaign, "breakdown", "pass")
            self.append(campaign, "profiles", "pass")
            plan = campaign / "arms" / "synthesized" / "synthesis" / "q21" / "plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text("{}\n")
            cells, reasons = run.select_retry_cells(
                campaign,
                artifacts.load_campaign(campaign)["configuration"],
                None,
                "timing",
                "21",
            )
        self.assertEqual(
            cells,
            (
                ("synthesized", "timing", 21),
                ("synthesized", "breakdown", 21),
            ),
        )
        self.assertEqual(
            reasons[("synthesized", "breakdown", 21)], "atomic peer of timing"
        )

    def test_retry_selector_rejects_values_outside_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.create_campaign(Path(temporary))
            with self.assertRaisesRegex(
                run.ConfigurationError, "queries are not part of the campaign"
            ):
                run.select_retry_cells(
                    campaign,
                    artifacts.load_campaign(campaign)["configuration"],
                    None,
                    None,
                    "20",
                )

    def test_code_drift_relaxation_is_scoped(self) -> None:
        comparisons = {
            "workflow hashes": ({"run.py": "old"}, {"run.py": "new"}),
            "binary hashes": ({"synthesis": "old"}, {"synthesis": "new"}),
            "dataset manifest": ({"lineitem": "same"}, {"lineitem": "same"}),
        }
        accepted = run.validate_campaign_comparisons(
            comparisons, retry=True, allow_code_drift=True
        )
        self.assertEqual(set(accepted), {"workflow hashes", "binary hashes"})
        with self.assertRaisesRegex(artifacts.ArtifactError, "dataset manifest"):
            run.validate_campaign_comparisons(
                {"dataset manifest": ("old", "new")},
                retry=True,
                allow_code_drift=True,
            )
        with self.assertRaisesRegex(artifacts.ArtifactError, "allow-code-drift"):
            run.validate_campaign_comparisons(
                {"workflow hashes": ("old", "new")},
                retry=True,
                allow_code_drift=False,
            )


class SynthesisRestartTests(unittest.TestCase):
    def run_wrapper(
        self,
        mode: str,
        *,
        existing_plan: bool = False,
        existing_checkpoint: bool = False,
        force: bool = False,
    ) -> tuple[CompletedProcess[str], int, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        build = root / "build"
        binary = build / "src" / "cuda" / "tpch" / "synthesize_tpch_ir_mpi"
        binary.parent.mkdir(parents=True)
        counter = root / "count"
        binary.write_text(
            """#!/usr/bin/env bash
set -u
count=0
if [[ -s $FAKE_COUNT ]]; then read -r count < "$FAKE_COUNT"; fi
count=$((count + 1))
printf '%s\n' "$count" > "$FAKE_COUNT"
output=''
checkpoint=''
resumed=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-config) output=$2; shift 2 ;;
        --checkpoint) checkpoint=$2; shift 2 ;;
        --resume) resumed=1; shift ;;
        *) shift ;;
    esac
done
case $FAKE_MODE in
    terminal) exit 64 ;;
    no_progress) exit 1 ;;
    progress_then_success)
        if (( count == 1 )); then
            printf '{}\n' > "$checkpoint"
            exit 1
        fi
        printf '{}\n' > "$output"
        exit 0
        ;;
    replace_existing)
        if (( resumed )); then exit 98; fi
        printf '{"fresh": true}\n' > "${output}.tmp"
        mv "${output}.tmp" "$output"
        exit 0
        ;;
esac
exit 99
"""
        )
        binary.chmod(0o755)
        probe = root / "probe"
        probe.write_text("#!/usr/bin/env bash\nprintf '1\\n'\n")
        probe.chmod(0o755)
        mpirun = root / "mpirun"
        mpirun.write_text(
            """#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do
    case $1 in
        -np|-x) shift 2 ;;
        *) break ;;
    esac
done
exec "$@"
"""
        )
        mpirun.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "ALPS_TPCH_SYNTH_BINARY": str(binary),
                "ALPS_CUDA_VISIBLE_DEVICE_COUNT_BINARY": str(probe),
                "MPIRUN": str(mpirun),
                "FAKE_COUNT": str(counter),
                "FAKE_MODE": mode,
            }
        )
        output = root / "output"
        query_dir = output / "plans"
        if existing_plan:
            query_dir.mkdir(parents=True)
            (query_dir / "q01.json").write_text('{"stale": true}\n')
        if existing_checkpoint:
            checkpoint_dir = output / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "q01.json").write_text('{"completed": true}\n')
        command = [
            str(run.REPOSITORY / "script" / "run_distributed_tpch_synthesis.sh"),
            "--scale",
            "1",
            "--queries",
            "1",
            "--output-dir",
            str(output),
            "--build-dir",
            str(build),
            "--ranks",
            "1",
            "--max-restarts",
            "8",
        ]
        if force:
            command.append("--force")
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        attempts = int(counter.read_text()) if counter.is_file() else 0
        return completed, attempts, query_dir / "q01.json"

    def test_no_checkpoint_progress_stops_after_one_attempt(self) -> None:
        completed, attempts, _ = self.run_wrapper("no_progress")
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(attempts, 1)
        self.assertIn("made no checkpoint progress; not restarting", completed.stdout)

    def test_terminal_exit_stops_after_one_attempt(self) -> None:
        completed, attempts, _ = self.run_wrapper("terminal")
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(attempts, 1)
        self.assertIn("terminal synthesis failure; not restarting", completed.stdout)

    def test_checkpoint_progress_allows_resume(self) -> None:
        completed, attempts, _ = self.run_wrapper("progress_then_success")
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(attempts, 2)
        self.assertIn("checkpoint advanced; restarting", completed.stdout)

    def test_force_replaces_existing_plan_without_resuming_stale_checkpoint(
        self,
    ) -> None:
        completed, attempts, plan = self.run_wrapper(
            "replace_existing",
            existing_plan=True,
            existing_checkpoint=True,
            force=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(attempts, 1)
        self.assertEqual(json.loads(plan.read_text()), {"fresh": True})


class ArtifactTests(unittest.TestCase):
    def test_new_rejects_even_an_empty_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "campaign"
            output.mkdir()
            with self.assertRaises(artifacts.ArtifactError):
                artifacts.create_campaign(output, {"scale": 1}, False)

    def test_campaign_can_be_loaded_by_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "campaign"
            fingerprint = artifacts.create_campaign(
                output,
                {"scale": 1},
                False,
                provenance={"git_worktree_status": "clean"},
            )
            manifest = artifacts.load_campaign(output)
            self.assertEqual(manifest["fingerprint"], fingerprint)
            self.assertEqual(manifest["provenance"]["git_worktree_status"], "clean")

    def test_tampered_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "campaign"
            artifacts.create_campaign(output, {"scale": 1}, False)
            manifest_path = output / "manifest" / "campaign.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["configuration"]["scale"] = 10
            artifacts.write_json(manifest_path, manifest)
            with self.assertRaises(artifacts.ArtifactError):
                artifacts.load_campaign(output)

    def test_parser_resume_accepts_only_campaign_directory(self) -> None:
        parser = run.build_parser()
        parsed = parser.parse_args(["resume", "/campaign"])
        self.assertEqual(parsed.campaign_dir, Path("/campaign"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["resume", "/campaign", "--scale", "1"])

    def test_resume_namespace_defaults_old_campaign_to_no_query_overrides(self) -> None:
        namespace = run.resume_namespace(
            {
                "arguments": {
                    "queries": [1, 9],
                    "memory_reserve_bytes": 40_000_000_000,
                }
            }
        )
        self.assertFalse(namespace.synthesis_continue_on_verify_failure)
        self.assertEqual(namespace.query_memory_reserve_bytes, {})
        self.assertEqual(namespace.memory_reserve_bytes, 40_000_000_000)
        self.assertEqual(
            namespace.parquet_decode_threads,
            definitions.DEFAULT_PARQUET_DECODE_THREADS,
        )

    def test_resume_namespace_restores_parquet_decode_threads(self) -> None:
        namespace = run.resume_namespace(
            {
                "arguments": {
                    "queries": [1],
                    "memory_reserve_bytes": 40_000_000_000,
                    "parquet_decode_threads": 4,
                }
            }
        )
        self.assertEqual(namespace.parquet_decode_threads, 4)

    def test_external_plans_are_copied_into_campaign_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "campaign" / "arms" / "synthesized" / "synthesis"
            source.mkdir()
            (source / "q01.json").write_text("{}\n")
            run.copy_external_plans(source, destination, (1,))
            self.assertEqual((destination / "q01" / "plan.json").read_text(), "{}\n")

    def test_new_then_directory_only_resume_uses_frozen_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / "build"
            for name in ("device_probe", "benchmark"):
                binary = build / run.BUILD_BINARIES[name]
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_text("#!/bin/sh\nexit 0\n")
                binary.chmod(0o755)
            data = root / "data"
            scale = data / "sf1"
            scale.mkdir(parents=True)
            for table in run.TPC_H_TABLES:
                (scale / f"{table}.parquet").write_text("table\n")
            golden = scale / "golden"
            golden.mkdir()
            (golden / "q01.parquet").write_text("golden\n")
            campaign = root / "campaign"
            local = topology.Topology((topology.Host("local", 1),), True, 2222)
            argv = [
                "new",
                str(campaign),
                "--scale",
                "1",
                "--arms",
                "legacy",
                "--stages",
                "timing",
                "--queries",
                "1",
                "--query-memory-reserve-bytes",
                "Q1=12000000000",
                "--parquet-decode-threads",
                "4",
                "--build-dir",
                str(build),
                "--tpch-data",
                str(data),
                "--mpirun",
                "true",
            ]
            with mock.patch.object(
                run, "resolve_topology", return_value=local
            ), mock.patch.object(run, "ensure_selected_gpus_idle"), mock.patch.object(
                run, "capture_local_platform", return_value={}
            ), mock.patch.object(
                run, "run_evaluation_cell", return_value=True
            ) as evaluate:
                self.assertEqual(run.main(argv), 0)
                first_root = evaluate.call_args.args[7]
                first_environment = evaluate.call_args.args[10]
                first_selected_environment = evaluate.call_args.args[11]
                evaluate.reset_mock()
                self.assertEqual(run.main(["resume", str(campaign)]), 0)
                resumed_root = evaluate.call_args.args[7]
                resumed_environment = evaluate.call_args.args[10]
            expected = campaign / "arms" / "legacy" / "timing" / "q01"
            self.assertEqual(first_root, expected)
            self.assertEqual(resumed_root, expected)
            self.assertEqual(
                first_environment["GAMAX_MEM_RESERVE_BYTES"], "12000000000"
            )
            self.assertEqual(
                first_selected_environment["GAMAX_MEM_RESERVE_BYTES"],
                "12000000000",
            )
            self.assertEqual(
                resumed_environment["GAMAX_MEM_RESERVE_BYTES"], "12000000000"
            )
            manifest = artifacts.load_campaign(campaign)
            self.assertEqual(manifest["configuration"]["schema_version"], 4)
            self.assertEqual(manifest["configuration"]["scale"], 1)
            self.assertEqual(
                manifest["configuration"]["arguments"]["query_memory_reserve_bytes"],
                {"1": 12_000_000_000},
            )
            self.assertEqual(
                manifest["configuration"]["arguments"]["parquet_decode_threads"],
                4,
            )
            common_args = manifest["configuration"]["policy"]["common_benchmark_args"]
            self.assertEqual(
                common_args[common_args.index("--parquet-decode-threads") + 1],
                "4",
            )
            self.assertEqual(
                manifest["configuration"]["policy"]["arms"]["legacy"]["environment"][
                    "NCCL_MIN_CTAS"
                ],
                "<unset>",
            )
            self.assertIn("nccl_configuration_sources", manifest["configuration"])
            self.assertEqual(len(list((campaign / "invocations").glob("*.json"))), 2)

    def test_only_complete_pass_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            query_root = Path(temporary)
            attempt = artifacts.next_attempt_dir(query_root)
            artifacts.write_json(
                attempt / "result.json",
                {"status": "pass", "expected_artifacts": ["e2e.csv"]},
            )
            self.assertIsNone(artifacts.successful_attempt(query_root))
            (attempt / "e2e.csv").write_text("header\nrow\n")
            self.assertEqual(artifacts.successful_attempt(query_root), attempt)

    def test_retry_does_not_skip_selected_synthesis_with_existing_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthesis_dir = root / "arms" / "synthesized" / "synthesis"
            plan = synthesis_dir / "q21" / "plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text('{"stale": true}\n')
            args = Namespace(
                resume=True,
                command="retry",
                dry_run=True,
                keep_going=True,
            )
            with mock.patch.object(
                run, "synthesis_command", return_value=["synthesize", "--force"]
            ) as build_command:
                self.assertTrue(
                    run.run_synthesis_stage(
                        args,
                        Path("/build"),
                        Path("/data"),
                        topology.Topology((topology.Host("local", 1),), True, 2222),
                        (21,),
                        synthesis_dir,
                        root,
                        {},
                        {},
                    )
                )
            build_command.assert_called_once()
            self.assertFalse((root / "status.csv").exists())

    def test_synthesis_receives_query_specific_memory_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthesis_dir = root / "arms" / "synthesized" / "synthesis"
            args = Namespace(
                resume=False,
                command="new",
                dry_run=False,
                keep_going=False,
                memory_reserve_bytes=40_000_000_000,
                query_memory_reserve_bytes={9: 58_000_000_000},
            )

            def fake_run(command, environment, log_path):
                self.assertEqual(environment["GAMAX_MEM_RESERVE_BYTES"], "58000000000")
                log_path.write_text("synthesis\n")
                plan = synthesis_dir / "q09" / "plan.json"
                plan.write_text("{}\n")
                return 0, 1.0

            with mock.patch.object(
                run, "synthesis_command", return_value=["synthesize"]
            ), mock.patch.object(run, "run_command", side_effect=fake_run):
                self.assertTrue(
                    run.run_synthesis_stage(
                        args,
                        Path("/build"),
                        Path("/data"),
                        topology.Topology((topology.Host("local", 1),), True, 2222),
                        (9,),
                        synthesis_dir,
                        root,
                        {"GAMAX_MEM_RESERVE_BYTES": "40000000000"},
                        {"GAMAX_MEM_RESERVE_BYTES": "40000000000"},
                    )
                )

    def test_combined_cell_materializes_and_resumes_both_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm_name = "legacy_cta32"
            timing_root = root / "arms" / arm_name / "timing" / "q01"
            breakdown_root = root / "arms" / arm_name / "breakdown" / "q01"
            args = Namespace(
                scale=1,
                resume=False,
                dry_run=False,
                keep_going=False,
                synthesis_attempt_timeout=100,
                synthesis_max_restarts=2,
                synthesis_seed=7,
                synthesis_arg=[],
            )
            local = topology.Topology((topology.Host("local", 1),), True, 2222)
            environment, selected_environment = run.arm_environment({}, {}, arm_name)

            def fake_run(command, environment, log_path):
                self.assertEqual(environment["NCCL_MIN_CTAS"], "32")
                self.assertEqual(environment["NCCL_MAX_CTAS"], "32")
                log_path.write_text("combined log\n")
                timing_attempt = log_path.parent
                breakdown_attempt = Path(
                    command[command.index("--steps-csv") + 1]
                ).parent
                (timing_attempt / "e2e.csv").write_text("timing\n")
                (breakdown_attempt / "steps.csv").write_text("steps\n")
                (breakdown_attempt / "phases.csv").write_text("phases\n")
                return 0, 1.5

            with mock.patch.object(run, "run_command", side_effect=fake_run):
                self.assertTrue(
                    run.run_combined_timing_breakdown_cell(
                        args,
                        Path("/build"),
                        Path("/data"),
                        local,
                        arm_name,
                        1,
                        timing_root,
                        breakdown_root,
                        None,
                        root,
                        environment,
                        selected_environment,
                    )
                )
            self.assertIsNotNone(artifacts.successful_attempt(timing_root))
            self.assertIsNotNone(artifacts.successful_attempt(breakdown_root))
            command_record = json.loads(
                (timing_root / "attempt-0001" / "command.json").read_text()
            )
            self.assertEqual(command_record["environment"]["NCCL_MIN_CTAS"], "32")
            self.assertEqual(command_record["environment"]["NCCL_MAX_CTAS"], "32")
            args.resume = True
            with mock.patch.object(
                run, "run_command", side_effect=AssertionError("must skip")
            ):
                self.assertTrue(
                    run.run_combined_timing_breakdown_cell(
                        args,
                        Path("/build"),
                        Path("/data"),
                        local,
                        arm_name,
                        1,
                        timing_root,
                        breakdown_root,
                        None,
                        root,
                        environment,
                        selected_environment,
                    )
                )

    def test_verification_failure_retains_complete_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                scale=1,
                resume=False,
                dry_run=False,
                collect_on_verify_failure=True,
                memory_reserve_bytes=40_000_000_000,
                query_memory_reserve_bytes={3: 18_000_000_000},
            )
            local = topology.Topology((topology.Host("local", 1),), True, 2222)
            environment, selected_environment = run.query_environment({}, {}, args, 3)

            def fake_run(command, environment, log_path):
                self.assertEqual(environment["GAMAX_MEM_RESERVE_BYTES"], "18000000000")
                attempt = log_path.parent
                log_path.write_text("VERIFY: MISMATCH\n")
                (attempt / "memory.csv").write_text("memory\n")
                (attempt / "verification.json").write_text(
                    json.dumps({"status": "mismatch", "diagnostic": "tied rows"})
                )
                return 1, 2.5

            query_root = root / "arms" / "legacy" / "memory" / "q03"
            with mock.patch.object(run, "run_command", side_effect=fake_run):
                self.assertFalse(
                    run.run_evaluation_cell(
                        args,
                        Path("/build"),
                        Path("/data"),
                        local,
                        "legacy",
                        "memory",
                        3,
                        query_root,
                        None,
                        root,
                        environment,
                        selected_environment,
                    )
                )

            attempt = query_root / "attempt-0001"
            self.assertTrue((attempt / "memory.csv").is_file())
            result = json.loads((attempt / "result.json").read_text())
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["failure_kind"], "verification")
            self.assertEqual(result["verification_status"], "mismatch")
            self.assertTrue(result["measurements_complete"])
            self.assertEqual(result["memory_reserve_bytes"], 18_000_000_000)
            command_record = json.loads((attempt / "command.json").read_text())
            self.assertEqual(
                command_record["environment"]["GAMAX_MEM_RESERVE_BYTES"],
                "18000000000",
            )

    def test_missing_synthesized_plan_blocks_evaluation_without_launching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(resume=False, dry_run=False)
            local = topology.Topology((topology.Host("local", 1),), True, 2222)
            with mock.patch.object(
                run, "run_command", side_effect=AssertionError("must not launch")
            ):
                self.assertFalse(
                    run.run_evaluation_cell(
                        args,
                        Path("/build"),
                        Path("/data"),
                        local,
                        "synthesized",
                        "memory",
                        21,
                        root / "arms" / "synthesized" / "memory" / "q21",
                        root / "arms" / "synthesized" / "synthesis",
                        root,
                        {},
                        {},
                    )
                )
            with (root / "status.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "blocked")
            self.assertEqual(rows[0]["exit_code"], "66")
            self.assertEqual(rows[0]["attempt"], "")

    def test_missing_plan_blocks_combined_timing_and_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(resume=False, dry_run=False)
            local = topology.Topology((topology.Host("local", 1),), True, 2222)
            with mock.patch.object(
                run, "run_command", side_effect=AssertionError("must not launch")
            ):
                self.assertFalse(
                    run.run_combined_timing_breakdown_cell(
                        args,
                        Path("/build"),
                        Path("/data"),
                        local,
                        "synthesized",
                        21,
                        root / "arms" / "synthesized" / "timing" / "q21",
                        root / "arms" / "synthesized" / "breakdown" / "q21",
                        root / "arms" / "synthesized" / "synthesis",
                        root,
                        {},
                        {},
                    )
                )
            with (root / "status.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [(row["stage"], row["status"]) for row in rows],
                [("timing", "blocked"), ("breakdown", "blocked")],
            )

    def test_resume_keeps_prior_pass_when_synthesized_plan_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_root = root / "arms" / "synthesized" / "memory" / "q21"
            attempt = artifacts.next_attempt_dir(query_root)
            artifacts.write_json(
                attempt / "result.json",
                {"status": "pass", "expected_artifacts": ["memory.csv"]},
            )
            (attempt / "memory.csv").write_text("memory\n")
            args = Namespace(resume=True, dry_run=False)
            local = topology.Topology((topology.Host("local", 1),), True, 2222)
            with mock.patch.object(
                run, "run_command", side_effect=AssertionError("must not launch")
            ):
                self.assertTrue(
                    run.run_evaluation_cell(
                        args,
                        Path("/build"),
                        Path("/data"),
                        local,
                        "synthesized",
                        "memory",
                        21,
                        query_root,
                        root / "arms" / "synthesized" / "synthesis",
                        root,
                        {},
                        {},
                    )
                )
            with (root / "status.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[-1]["status"], "skip_pass")


if __name__ == "__main__":
    unittest.main()
