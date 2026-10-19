#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Run reproducible distributed TPC-H experiment arms inside a container."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from artifacts import (  # noqa: E402
    ArtifactError,
    append_status,
    create_campaign,
    load_campaign,
    next_attempt_dir,
    next_attempt_path,
    record_invocation,
    successful_attempt,
    utc_now,
    write_json,
)
from definitions import (  # noqa: E402
    ARMS,
    COMBINED_TIMING_BREAKDOWN_ARGS,
    COMMON_BENCHMARK_ARGS,
    DEFAULT_PARQUET_DECODE_THREADS,
    STAGES,
    ExecutorKind,
    PlanSource,
)
from topology import (  # noqa: E402
    Topology,
    TopologyError,
    preflight_remote_hosts,
    resolve_topology,
    run_remote,
)
from local_data import (  # noqa: E402
    build_synthetic_manifest,
    cache_data_root,
    write_manifest_bundle,
)
from nsys_preflight import (  # noqa: E402
    PreflightError,
    physical_gpu_candidates,
    select_gpu,
)
from ncu_profile import (  # noqa: E402
    NcuProfileError,
    inspect_report as inspect_ncu_report,
    kernel_name_expression,
    parse_kernel_substrings,
    query_section_metrics,
    run_preflight as run_ncu_counter_preflight,
)
from nccl_config_preflight import CONTROLLED_KEYS, inspect_configuration  # noqa: E402


BUILD_BINARIES = {
    "benchmark": Path("src/cuda/tpch/benchmark_tpch_ir_mpi"),
    "synthesis": Path("src/cuda/tpch/synthesize_tpch_ir_mpi"),
    "device_probe": Path("src/cuda/tpch/cuda_visible_device_count"),
    "ncu_probe": Path("src/cuda/tpch/ncu_counter_probe"),
}
TPCH_METADATA_BINARY = Path("src/cuda/tpch/tpch_parquet_metadata")
LOCAL_DATA_HELPER = SCRIPT_DIR / "local_data.py"
TPC_H_TABLES = (
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
)
NCCL_CTA_ENVIRONMENT_KEYS = CONTROLLED_KEYS
SUCCESSFUL_CELL_STATUSES = frozenset(("pass", "skip_pass"))
TERMINAL_CELL_STATUSES = SUCCESSFUL_CELL_STATUSES | {"fail", "blocked"}
STATUS_COLUMNS = frozenset(
    ("timestamp_utc", "arm", "stage", "query", "status", "exit_code", "attempt")
)
MISSING_CASE_DISPLAY_LIMIT = 20
DEFAULT_MEMORY_RESERVE_BYTES = 32_000_000_000
DEFAULT_NCU_METRICS_FILE = REPOSITORY / "ncu_metrics.section"
DEFAULT_NCU_KERNELS_FILE = REPOSITORY / "ncu_kernels.txt"
DEFAULT_NCU_PREFLIGHT_TIMEOUT = 120
DEFAULT_NCU_CAPTURE_TIMEOUT = 7200
RELAXABLE_CODE_COMPARISONS = frozenset(
    ("binary hashes", "remote executable hashes", "workflow hashes")
)

StatusCell = tuple[str, str, int]


class ConfigurationError(ValueError):
    pass


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        meaningful_default = action.default not in (
            None,
            False,
            [],
            argparse.SUPPRESS,
        )
        if meaningful_default and "%(default)" not in help_text:
            help_text += " (default: %(default)s)"
        return help_text


def build_parser() -> argparse.ArgumentParser:
    arm_help = "\n".join(f"  {name:30} {arm.description}" for name, arm in ARMS.items())
    stage_help = "\n".join(
        f"  {name:12} {stage.description}" for name, stage in STAGES.items()
    )
    parser = argparse.ArgumentParser(
        description=(
            "Create, resume, retry, or inspect a distributed TPC-H campaign. "
            "Execution is sequential and uses one MPI rank per selected GPU.\n\n"
            "Use 'run.py new --help' for all campaign-creation options. "
            "Resume replays the frozen matrix; retry can select unsuccessful cells."
        ),
        formatter_class=HelpFormatter,
        epilog=f"Available arms:\n{arm_help}\n\nAvailable stages:\n{stage_help}",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser(
        "new",
        help="create a campaign in a directory that does not exist",
        description=(
            "Create and run a new campaign. CAMPAIGN_DIR must not exist, even "
            "if an existing directory is empty."
        ),
        formatter_class=HelpFormatter,
        epilog=f"Available arms:\n{arm_help}\n\nAvailable stages:\n{stage_help}",
    )
    new.add_argument(
        "campaign_dir",
        type=Path,
        help="new directory that will contain the manifest and all results",
    )
    required = new.add_argument_group("required campaign inputs")
    required.add_argument(
        "--scale", type=positive_int, required=True, help="TPC-H scale factor"
    )
    required.add_argument("--arms", required=True, help="comma-separated arm names")
    required.add_argument("--stages", required=True, help="comma-separated stage names")

    inputs = new.add_argument_group("repository and data inputs")
    inputs.add_argument(
        "--build-dir",
        type=Path,
        help="CMake build tree (fallback: EIGER_BUILD_DIR, then unique build*)",
    )
    inputs.add_argument(
        "--tpch-data",
        type=Path,
        help="root containing sf<N> directories (fallback: TPCH_DATA_PATH)",
    )
    inputs.add_argument(
        "--physical-plan-dir",
        type=Path,
        help="existing guarded qNN.json plans for the synthesized arm",
    )
    inputs.add_argument(
        "--queries", default="1-22", help="comma-separated query IDs and ranges"
    )
    inputs.add_argument(
        "--generate-local-data",
        action="store_true",
        help="generate assigned TPC-H files into host-local storage before execution",
    )
    inputs.add_argument(
        "--generated-data-partitions",
        type=positive_int,
        help="global Parquet partition count for --generate-local-data",
    )
    inputs.add_argument(
        "--local-data-cache-root",
        type=Path,
        default=Path("/tmp/alps-tpch"),
        help="host-local generated-data cache root",
    )
    inputs.add_argument(
        "--tpchgen",
        type=Path,
        default=Path("/home/scada/.local/bin/tpchgen-cli"),
        help="tpchgen-cli executable used for host-local generation",
    )

    topology = new.add_argument_group("direct-container topology")
    topology.add_argument(
        "--hosts",
        help="exact comma-separated hosts, optionally with :slots; omit for local",
    )
    topology.add_argument(
        "--gpus-per-node",
        type=positive_int,
        help=(
            "GPUs/ranks per selected host; inferred from local visibility or "
            "explicit host slot counts when omitted"
        ),
    )
    topology.add_argument(
        "--ssh-port",
        type=positive_int,
        default=2222,
        help="SSH port used by direct multi-node launches",
    )
    topology.add_argument(
        "--mpirun",
        default=os.environ.get("MPIRUN", "/usr/local/mpi/bin/mpirun"),
        help="MPI launcher path",
    )

    synthesis = new.add_argument_group("plan synthesis")
    synthesis.add_argument(
        "--synthesis-attempt-timeout",
        type=positive_int,
        default=21600,
        help="hard timeout in seconds for one synthesis attempt",
    )
    synthesis.add_argument(
        "--synthesis-max-restarts",
        type=nonnegative_int,
        default=8,
        help="maximum checkpoint-based restarts after the initial attempt",
    )
    synthesis.add_argument(
        "--synthesis-seed",
        type=nonnegative_int,
        default=20260728,
        help="random seed used by plan synthesis",
    )
    synthesis.add_argument(
        "--synthesis-arg",
        action="append",
        default=[],
        help="one extra synthesis-binary argument; repeat for multiple tokens",
    )
    synthesis.add_argument(
        "--synthesis-continue-on-verify-failure",
        action="store_true",
        help=(
            "continue plan synthesis after golden-result mismatches; execution "
            "failures remain fatal"
        ),
    )

    runtime = new.add_argument_group("runtime policy")
    runtime.add_argument(
        "--memory-reserve-bytes",
        type=positive_int,
        default=DEFAULT_MEMORY_RESERVE_BYTES,
        help="GPU memory reserved from the RMM arena",
    )
    runtime.add_argument(
        "--query-memory-reserve-bytes",
        action="append",
        default=[],
        metavar="QUERY=BYTES",
        help=(
            "override --memory-reserve-bytes for one selected query; QUERY may "
            "be written as 9 or Q9; repeat for multiple queries"
        ),
    )
    runtime.add_argument(
        "--parquet-decode-threads",
        type=positive_int,
        default=DEFAULT_PARQUET_DECODE_THREADS,
        help="Arrow CPU threads used to decode Parquet data per MPI rank",
    )
    runtime.add_argument(
        "--nccl-socket-ifname",
        default=os.environ.get("NCCL_SOCKET_IFNAME", "eth"),
        help="network interface passed through NCCL_SOCKET_IFNAME",
    )
    runtime.add_argument(
        "--keep-going",
        action="store_true",
        help="continue independent cells after a failure",
    )
    runtime.add_argument(
        "--collect-on-verify-failure",
        action="store_true",
        help=(
            "finish writing the current cell's measurement artifacts after a "
            "verification mismatch; the cell still fails"
        ),
    )
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved commands without creating the campaign or running work",
    )
    profiling = new.add_argument_group("profiling preflight")
    profiling.add_argument(
        "--profile-preflight-timeout",
        type=positive_int,
        default=20,
        help="hard timeout in seconds for each Nsight GPU candidate",
    )
    profiling.add_argument(
        "--profile-preflight-seconds",
        type=positive_int,
        default=2,
        help="duration of each GPU-metrics validation capture",
    )
    profiling.add_argument(
        "--profile-gpu-metrics-frequency",
        type=positive_int,
        default=10000,
        help="GPU hardware-metrics sampling frequency in Hz",
    )
    profiling.add_argument(
        "--ncu-metrics-file",
        type=Path,
        default=DEFAULT_NCU_METRICS_FILE,
        help="native NCU section file for the ncu_profiles stage",
    )
    profiling.add_argument(
        "--ncu-kernels-file",
        type=Path,
        default=DEFAULT_NCU_KERNELS_FILE,
        help="literal demangled kernel-name substrings for the ncu_profiles stage",
    )
    profiling.add_argument(
        "--ncu-preflight-timeout",
        type=positive_int,
        default=DEFAULT_NCU_PREFLIGHT_TIMEOUT,
        help="hard timeout in seconds for the standalone rank-0 NCU counter probe",
    )
    profiling.add_argument(
        "--ncu-capture-timeout",
        type=positive_int,
        default=DEFAULT_NCU_CAPTURE_TIMEOUT,
        help="hard timeout in seconds for one distributed NCU query capture",
    )

    resume = commands.add_parser(
        "resume",
        help="resume a campaign using only its existing directory",
        description=(
            "Resume the frozen matrix in an existing campaign. All scale, arm, "
            "stage, query, topology, and runtime settings come from its manifest."
        ),
        formatter_class=HelpFormatter,
    )
    resume.add_argument(
        "campaign_dir",
        type=Path,
        help="existing campaign directory containing manifest/campaign.json",
    )
    retry = commands.add_parser(
        "retry",
        help="retry selected unsuccessful cells in an existing campaign",
        description=(
            "Retry failed, blocked, or missing cells selected from an immutable "
            "campaign. Passed cells remain untouched, except that timing and "
            "breakdown are promoted together when the campaign combines them."
        ),
        formatter_class=HelpFormatter,
    )
    retry.add_argument(
        "campaign_dir",
        type=Path,
        help="existing campaign directory containing manifest/campaign.json",
    )
    retry.add_argument(
        "--queries",
        help="optional comma-separated query IDs/ranges within the campaign",
    )
    retry.add_argument(
        "--arms",
        help="optional comma-separated arm names within the campaign",
    )
    retry.add_argument(
        "--stages",
        help="optional comma-separated stage names within the campaign",
    )
    retry.add_argument(
        "--allow-code-drift",
        action="store_true",
        help=(
            "allow saved workflow and executable hashes to differ while keeping "
            "manifest, data, topology, runtime, and live local/remote agreement strict"
        ),
    )
    retry.add_argument(
        "--dry-run",
        action="store_true",
        help="print selected cells and commands without changing the campaign",
    )
    status = commands.add_parser(
        "status",
        help="report campaign completion, success, and failed cases",
        description=(
            "Inspect an existing campaign without modifying it. Completion is "
            "derived from the frozen manifest matrix and the latest status.csv "
            "record for each cell."
        ),
        formatter_class=HelpFormatter,
    )
    status.add_argument(
        "campaign_dir",
        type=Path,
        help="existing campaign directory containing manifest/campaign.json",
    )
    return parser


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def parse_named_list(
    value: str, available: dict[str, object], kind: str
) -> tuple[str, ...]:
    result: list[str] = []
    for token in value.split(","):
        name = token.strip()
        if not name:
            raise ConfigurationError(f"empty {kind} name")
        if name not in available:
            raise ConfigurationError(
                f"unknown {kind} {name!r}; choose from {', '.join(available)}"
            )
        if name in result:
            raise ConfigurationError(f"duplicate {kind}: {name}")
        result.append(name)
    return tuple(result)


def parse_queries(specification: str) -> tuple[int, ...]:
    selected: set[int] = set()
    for chunk in specification.split(","):
        chunk = chunk.strip()
        if not chunk:
            raise ConfigurationError("empty query specification")
        if "-" in chunk:
            bounds = chunk.split("-")
            if len(bounds) != 2 or not all(bound.isdigit() for bound in bounds):
                raise ConfigurationError(f"invalid query range: {chunk}")
            first, last = map(int, bounds)
            if first > last:
                raise ConfigurationError(f"descending query range: {chunk}")
            selected.update(range(first, last + 1))
        elif chunk.isdigit():
            selected.add(int(chunk))
        else:
            raise ConfigurationError(f"invalid query ID: {chunk}")
    if not selected or min(selected) < 1 or max(selected) > 22:
        raise ConfigurationError("query IDs must be in [1, 22]")
    return tuple(sorted(selected))


def parse_query_memory_reserve_bytes(
    specifications: Iterable[str], selected_queries: Iterable[int]
) -> dict[int, int]:
    selected = set(selected_queries)
    overrides: dict[int, int] = {}
    for specification in specifications:
        query_token, separator, bytes_token = specification.partition("=")
        query_token = query_token.strip()
        bytes_token = bytes_token.strip()
        if not separator or not query_token or not bytes_token:
            raise ConfigurationError(
                "query memory reserve overrides must use QUERY=BYTES"
            )
        if query_token[:1].lower() == "q":
            query_token = query_token[1:]
        if not query_token.isdigit():
            raise ConfigurationError(
                f"invalid query memory reserve query: {specification!r}"
            )
        query = int(query_token)
        if query < 1 or query > 22:
            raise ConfigurationError("query memory reserve IDs must be in [1, 22]")
        if query not in selected:
            raise ConfigurationError(
                f"query memory reserve override Q{query} is not in --queries"
            )
        if query in overrides:
            raise ConfigurationError(
                f"duplicate query memory reserve override: Q{query}"
            )
        try:
            reserve_bytes = int(bytes_token)
        except ValueError as error:
            raise ConfigurationError(
                f"invalid query memory reserve bytes: {bytes_token!r}"
            ) from error
        if reserve_bytes < 1:
            raise ConfigurationError("query memory reserve bytes must be positive")
        overrides[query] = reserve_bytes
    return overrides


def saved_query_memory_reserve_bytes(
    value: object, selected_queries: Iterable[int]
) -> dict[int, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ArtifactError("campaign query memory reserve overrides are invalid")
    specifications = [f"{query}={reserve}" for query, reserve in value.items()]
    try:
        return parse_query_memory_reserve_bytes(specifications, selected_queries)
    except ConfigurationError as error:
        raise ArtifactError(
            f"campaign query memory reserve overrides are invalid: {error}"
        ) from error


def resolve_path(value: Path) -> Path:
    return value.expanduser().resolve()


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def expected_campaign_cells(configuration: dict[str, object]) -> tuple[StatusCell, ...]:
    """Return the immutable status matrix encoded in a campaign manifest."""

    if configuration.get("schema_version") != 4:
        raise ArtifactError("this runner supports only schema-version 4 campaigns")

    arms_value = configuration.get("arms")
    stages_value = configuration.get("stages")
    queries_value = configuration.get("queries")
    if (
        not isinstance(arms_value, list)
        or not arms_value
        or not all(isinstance(arm, str) and arm in ARMS for arm in arms_value)
    ):
        raise ArtifactError("campaign manifest has invalid arms")
    if (
        not isinstance(stages_value, list)
        or not stages_value
        or not all(isinstance(stage, str) and stage in STAGES for stage in stages_value)
    ):
        raise ArtifactError("campaign manifest has invalid stages")
    if (
        not isinstance(queries_value, list)
        or not queries_value
        or not all(type(query) is int and 1 <= query <= 22 for query in queries_value)
    ):
        raise ArtifactError("campaign manifest has invalid queries")
    if len(set(arms_value)) != len(arms_value):
        raise ArtifactError("campaign manifest has duplicate arms")
    if len(set(stages_value)) != len(stages_value):
        raise ArtifactError("campaign manifest has duplicate stages")
    if len(set(queries_value)) != len(queries_value):
        raise ArtifactError("campaign manifest has duplicate queries")

    arms = tuple(arms_value)
    stages = tuple(stages_value)
    queries = tuple(queries_value)
    cells: list[StatusCell] = []
    if "synthesis" in stages:
        cells.extend(("synthesized", "synthesis", query) for query in queries)
    cells.extend(
        (arm, stage, query)
        for arm in arms
        for stage in stages
        if stage != "synthesis"
        for query in queries
    )
    return tuple(cells)


def latest_campaign_statuses(
    campaign_dir: Path, expected_cells: tuple[StatusCell, ...]
) -> dict[StatusCell, dict[str, object]]:
    """Read the append-only status journal, retaining the latest row per cell."""

    status_path = campaign_dir / "status.csv"
    if not status_path.exists():
        return {}
    if not status_path.is_file():
        raise ArtifactError(f"campaign status journal is not a file: {status_path}")

    expected = set(expected_cells)
    latest: dict[StatusCell, dict[str, object]] = {}
    try:
        with status_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            if not STATUS_COLUMNS.issubset(columns):
                missing = ", ".join(sorted(STATUS_COLUMNS - columns))
                raise ArtifactError(
                    f"campaign status journal is missing columns: {missing}"
                )
            for line_number, row in enumerate(reader, start=2):
                arm = row.get("arm")
                stage = row.get("stage")
                query_text = row.get("query")
                status = row.get("status")
                exit_code_text = row.get("exit_code")
                attempt = row.get("attempt")
                try:
                    if not all(
                        isinstance(value, str)
                        for value in (
                            arm,
                            stage,
                            query_text,
                            status,
                            exit_code_text,
                            attempt,
                        )
                    ):
                        raise ValueError
                    if not query_text.startswith("Q"):
                        raise ValueError
                    query = int(query_text[1:])
                    exit_code = int(exit_code_text)
                except (TypeError, ValueError) as error:
                    raise ArtifactError(
                        f"invalid campaign status row at line {line_number}"
                    ) from error
                if status not in TERMINAL_CELL_STATUSES:
                    raise ArtifactError(
                        f"invalid campaign status {status!r} at line {line_number}"
                    )
                cell = (arm, stage, query)
                if cell not in expected:
                    raise ArtifactError(
                        "unexpected campaign status cell at line "
                        f"{line_number}: {format_status_cell(cell)}"
                    )
                latest[cell] = {
                    "status": status,
                    "exit_code": exit_code,
                    "attempt": attempt,
                }
    except (OSError, csv.Error) as error:
        raise ArtifactError(f"cannot read campaign status journal: {error}") from error
    return latest


def inspect_campaign_status(campaign_dir: Path) -> dict[str, object]:
    """Summarize completion from the manifest matrix and latest journal rows."""

    manifest = load_campaign(campaign_dir)
    configuration = manifest["configuration"]
    expected = expected_campaign_cells(configuration)
    latest = latest_campaign_statuses(campaign_dir, expected)
    passed = tuple(
        cell
        for cell in expected
        if cell in latest and latest[cell]["status"] in SUCCESSFUL_CELL_STATUSES
    )
    failed = tuple(
        (cell, latest[cell])
        for cell in expected
        if cell in latest and latest[cell]["status"] == "fail"
    )
    blocked = tuple(
        (cell, latest[cell])
        for cell in expected
        if cell in latest and latest[cell]["status"] == "blocked"
    )
    missing = tuple(cell for cell in expected if cell not in latest)
    complete = not missing
    result = (
        "success"
        if complete and not failed and not blocked
        else "failure" if complete else "pending"
    )
    return {
        "campaign_dir": campaign_dir,
        "complete": complete,
        "result": result,
        "expected": expected,
        "passed": passed,
        "failed": failed,
        "blocked": blocked,
        "missing": missing,
    }


def format_status_cell(cell: StatusCell) -> str:
    arm, stage, query = cell
    return f"{arm}/{stage}/Q{query:02d}"


def print_campaign_status(report: dict[str, object]) -> None:
    failed = report["failed"]
    blocked = report["blocked"]
    missing = report["missing"]
    print(f"Campaign: {report['campaign_dir']}")
    print(f"Completion: {'COMPLETE' if report['complete'] else 'INCOMPLETE'}")
    print(f"Result: {str(report['result']).upper()}")
    print(f"Expected: {len(report['expected'])}")
    print(f"Passed: {len(report['passed'])}")
    print(f"Failed: {len(failed)}")
    print(f"Blocked: {len(blocked)}")
    print(f"Missing: {len(missing)}")
    if failed:
        print("\nFailed cases:")
        for cell, record in failed:
            attempt = record["attempt"] or "<none>"
            print(
                f"  {format_status_cell(cell)} exit={record['exit_code']} "
                f"attempt={attempt}"
            )
    if blocked:
        print("\nBlocked cases:")
        for cell, _ in blocked:
            print(f"  {format_status_cell(cell)} missing synthesized plan")
    if missing:
        print("\nMissing cases:")
        for cell in missing[:MISSING_CASE_DISPLAY_LIMIT]:
            print(f"  {format_status_cell(cell)}")
        remaining = len(missing) - MISSING_CASE_DISPLAY_LIMIT
        if remaining > 0:
            print(f"  ... {remaining} more")


def campaign_status_exit_code(report: dict[str, object]) -> int:
    if not report["complete"]:
        return 2
    return 0 if report["result"] == "success" else 1


def select_retry_cells(
    campaign_dir: Path,
    configuration: dict[str, object],
    arm_spec: str | None,
    stage_spec: str | None,
    query_spec: str | None,
) -> tuple[tuple[StatusCell, ...], dict[StatusCell, str]]:
    """Select unsuccessful cells and close synthesized-plan dependencies."""

    expected = expected_campaign_cells(configuration)
    expected_set = set(expected)
    latest = latest_campaign_statuses(campaign_dir, expected)
    campaign_arms = tuple(configuration["arms"])
    campaign_stages = tuple(configuration["stages"])
    campaign_queries = tuple(configuration["queries"])

    selected_arms = (
        parse_named_list(arm_spec, ARMS, "arm")
        if arm_spec is not None
        else campaign_arms
    )
    selected_stages = (
        parse_named_list(stage_spec, STAGES, "stage")
        if stage_spec is not None
        else campaign_stages
    )
    selected_queries = (
        parse_queries(query_spec) if query_spec is not None else campaign_queries
    )
    for name, selected, available in (
        ("arms", selected_arms, campaign_arms),
        ("stages", selected_stages, campaign_stages),
        ("queries", selected_queries, campaign_queries),
    ):
        outside = [value for value in selected if value not in available]
        if outside:
            rendered = ",".join(map(str, outside))
            raise ConfigurationError(
                f"retry {name} are not part of the campaign: {rendered}"
            )

    selected_arm_set = set(selected_arms)
    selected_stage_set = set(selected_stages)
    selected_query_set = set(selected_queries)
    includes_synthesized_arm = any(
        ARMS[arm].plan_source == PlanSource.SYNTHESIZED for arm in selected_arms
    )
    reasons: dict[StatusCell, str] = {}
    for cell in expected:
        arm, stage, query = cell
        if query not in selected_query_set or stage not in selected_stage_set:
            continue
        if stage == "synthesis":
            if not includes_synthesized_arm:
                continue
        elif arm not in selected_arm_set:
            continue
        record = latest.get(cell)
        if record is not None and record["status"] in SUCCESSFUL_CELL_STATUSES:
            continue
        reasons[cell] = "missing" if record is None else str(record["status"])

    # A synthesized evaluation cannot run without its per-query plan. Add the
    # synthesis cell even when --stages omitted synthesis explicitly.
    for arm, stage, query in tuple(reasons):
        if stage == "synthesis" or ARMS[arm].plan_source != PlanSource.SYNTHESIZED:
            continue
        plan = (
            campaign_dir
            / "arms"
            / "synthesized"
            / "synthesis"
            / f"q{query:02d}"
            / "plan.json"
        )
        if is_nonempty_file(plan):
            continue
        dependency = ("synthesized", "synthesis", query)
        if dependency not in expected_set:
            raise ConfigurationError(
                f"{format_status_cell((arm, stage, query))} needs a synthesized "
                "plan, but synthesis is not part of this campaign"
            )
        reasons.setdefault(dependency, "dependency: missing synthesized plan")

    policy = configuration.get("policy")
    combines_timing_breakdown = (
        isinstance(policy, dict) and policy.get("combine_timing_breakdown") is True
    )
    if combines_timing_breakdown:
        for arm, stage, query in tuple(reasons):
            if stage not in ("timing", "breakdown"):
                continue
            peer_stage = "breakdown" if stage == "timing" else "timing"
            peer = (arm, peer_stage, query)
            if peer in expected_set:
                reasons.setdefault(peer, f"atomic peer of {stage}")

    return tuple(cell for cell in expected if cell in reasons), reasons


def print_retry_selection(
    cells: tuple[StatusCell, ...], reasons: dict[StatusCell, str]
) -> None:
    print(f"Retry cells: {len(cells)}")
    for cell in cells[:MISSING_CASE_DISPLAY_LIMIT]:
        print(f"  {format_status_cell(cell)} [{reasons[cell]}]")
    remaining = len(cells) - MISSING_CASE_DISPLAY_LIMIT
    if remaining > 0:
        print(f"  ... {remaining} more")


def retry_execution_axes(
    configuration: dict[str, object], cells: tuple[StatusCell, ...]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    cell_set = set(cells)
    arms = tuple(
        arm
        for arm in configuration["arms"]
        if any(cell[0] == arm and cell[1] != "synthesis" for cell in cell_set)
    )
    stages = tuple(
        stage
        for stage in configuration["stages"]
        if any(cell[1] == stage for cell in cell_set)
    )
    queries = tuple(
        query
        for query in configuration["queries"]
        if any(cell[2] == query for cell in cell_set)
    )
    return arms, stages, queries


def validate_campaign_comparisons(
    comparisons: Mapping[str, tuple[object, object]],
    *,
    retry: bool,
    allow_code_drift: bool,
) -> dict[str, object]:
    accepted: dict[str, object] = {}
    for name, (saved, current) in comparisons.items():
        if saved == current:
            continue
        if retry and allow_code_drift and name in RELAXABLE_CODE_COMPARISONS:
            accepted[name] = {"saved": saved, "current": current}
            continue
        hint = (
            "; retry with --allow-code-drift to accept only "
            "workflow/executable changes"
            if retry and name in RELAXABLE_CODE_COMPARISONS
            else ""
        )
        raise ArtifactError(f"campaign {name} no longer match{hint}")
    return accepted


def required_binary_names(stages: tuple[str, ...]) -> tuple[str, ...]:
    names = ["device_probe"]
    if any(STAGES[stage].executor != ExecutorKind.SYNTHESIS for stage in stages):
        names.append("benchmark")
    if "synthesis" in stages:
        names.append("synthesis")
    if "ncu_profiles" in stages:
        names.append("ncu_probe")
    return tuple(names)


def resolve_build_dir(requested: Path | None, stages: tuple[str, ...]) -> Path:
    required = required_binary_names(stages)
    configured = requested
    if configured is None and os.environ.get("EIGER_BUILD_DIR"):
        configured = Path(os.environ["EIGER_BUILD_DIR"])
    if configured is not None:
        candidate = resolve_path(configured)
        validate_build_dir(candidate, required)
        return candidate

    candidates = []
    for candidate in sorted(REPOSITORY.glob("build*")):
        if candidate.is_dir() and all(
            (candidate / BUILD_BINARIES[name]).is_file() for name in required
        ):
            candidates.append(candidate.resolve())
    if len(candidates) == 1:
        validate_build_dir(candidates[0], required)
        return candidates[0]
    rendered = ", ".join(str(path) for path in candidates) or "none"
    raise ConfigurationError(
        "cannot choose a build directory unambiguously; pass --build-dir "
        f"(valid auto-detected candidates: {rendered})"
    )


def validate_build_dir(build_dir: Path, required: Iterable[str]) -> None:
    if not build_dir.is_dir():
        raise ConfigurationError(f"build directory does not exist: {build_dir}")
    for name in required:
        binary = build_dir / BUILD_BINARIES[name]
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ConfigurationError(f"required binary is not executable: {binary}")


def resolve_data_root(
    requested: Path | None,
    scale: int,
    queries: tuple[int, ...],
    *,
    require_tables: bool = True,
) -> Path:
    configured = requested
    if configured is None and os.environ.get("TPCH_DATA_PATH"):
        configured = Path(os.environ["TPCH_DATA_PATH"])
    if configured is None:
        raise ConfigurationError("pass --tpch-data or set TPCH_DATA_PATH")
    root = resolve_path(configured)
    scale_dir = root / f"sf{scale}"
    if not scale_dir.is_dir():
        raise ConfigurationError(f"TPC-H scale directory does not exist: {scale_dir}")
    if require_tables:
        missing_tables = [
            table
            for table in TPC_H_TABLES
            if not (scale_dir / f"{table}.parquet").is_file()
            and not (scale_dir / table).is_dir()
        ]
        if missing_tables:
            raise ConfigurationError(
                f"TPC-H SF{scale} is missing table data for: {', '.join(missing_tables)}"
            )
    missing_goldens = [
        query
        for query in queries
        if not (scale_dir / "golden" / f"q{query:02d}.parquet").is_file()
    ]
    if missing_goldens:
        rendered = ",".join(f"Q{query}" for query in missing_goldens)
        raise ConfigurationError(f"missing verification goldens: {rendered}")
    return root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_ncu_metrics(args: argparse.Namespace, stages: tuple[str, ...]) -> None:
    if "ncu_profiles" not in stages:
        args.ncu_metrics = []
        args.ncu_launch_metrics = []
        args.ncu_metrics_sha256 = None
        args.ncu_kernels = []
        args.ncu_kernel_expression = None
        args.ncu_kernels_sha256 = None
        return
    path = resolve_path(Path(args.ncu_metrics_file))
    kernels_path = resolve_path(Path(args.ncu_kernels_file))
    try:
        metrics, launch_metrics = query_section_metrics(
            path, "ncu", args.ncu_preflight_timeout
        )
        kernels = parse_kernel_substrings(kernels_path)
    except (OSError, NcuProfileError) as error:
        raise ConfigurationError(str(error)) from error
    args.ncu_metrics_file = path
    args.ncu_metrics = list(metrics)
    args.ncu_launch_metrics = list(launch_metrics)
    args.ncu_metrics_sha256 = sha256(path)
    args.ncu_kernels_file = kernels_path
    args.ncu_kernels = list(kernels)
    args.ncu_kernel_expression = kernel_name_expression(kernels)
    args.ncu_kernels_sha256 = sha256(kernels_path)


def dataset_manifest(data_root: Path, scale: int) -> list[dict[str, object]]:
    scale_dir = data_root / f"sf{scale}"
    return [
        {"path": str(path.relative_to(data_root)), "size": path.stat().st_size}
        for path in sorted(scale_dir.rglob("*"))
        if path.is_file()
    ]


def validate_generated_data_configuration(
    args: argparse.Namespace, topology: Topology
) -> None:
    enabled = bool(getattr(args, "generate_local_data", False))
    partitions = getattr(args, "generated_data_partitions", None)
    if enabled and partitions is None:
        raise ConfigurationError(
            "--generated-data-partitions is required with --generate-local-data"
        )
    if not enabled and partitions is not None:
        raise ConfigurationError(
            "--generated-data-partitions requires --generate-local-data"
        )
    if not enabled:
        return
    assert isinstance(partitions, int)
    if partitions != 1 and partitions < topology.ranks:
        raise ConfigurationError(
            "generated partition count must be 1 or at least the MPI rank count "
            f"({partitions} partitions for {topology.ranks} ranks)"
        )
    synthesis_strategy = "round_robin"
    synthesis_args = list(getattr(args, "synthesis_arg", ()) or ())
    for index, token in enumerate(synthesis_args):
        if token == "--file-shard-strategy":
            if index + 1 == len(synthesis_args):
                raise ConfigurationError(
                    "--synthesis-arg=--file-shard-strategy requires a following "
                    "--synthesis-arg=VALUE"
                )
            synthesis_strategy = synthesis_args[index + 1]
        elif token.startswith("--file-shard-strategy="):
            synthesis_strategy = token.partition("=")[2]
    if partitions != 1 and synthesis_strategy not in (
        "round_robin",
        "round-robin",
    ):
        raise ConfigurationError(
            "partitioned host-local data requires synthesis file sharding to use "
            "round_robin; remove the --synthesis-arg file-shard-strategy override"
        )
    generator = resolve_path(Path(args.tpchgen))
    if not generator.is_file() or not os.access(generator, os.X_OK):
        raise ConfigurationError(f"tpchgen-cli is not executable: {generator}")
    args.tpchgen = generator
    args.local_data_cache_root = resolve_path(Path(args.local_data_cache_root))


def host_global_ranks(topology: Topology) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    first = 0
    for host in topology.hosts:
        result[host.name] = tuple(range(first, first + host.slots))
        first += host.slots
    return result


def campaign_tpchgen_path(campaign_dir: Path) -> Path:
    return campaign_dir / "manifest" / "generated_data" / "tpchgen-cli"


def stage_campaign_tpchgen(source: Path, destination: Path) -> None:
    source_digest = sha256(source)
    if destination.is_file() and sha256(destination) == source_digest:
        destination.chmod(destination.stat().st_mode | 0o111)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        temporary.chmod(temporary.stat().st_mode | 0o111)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _run_local_data_helper(
    topology: Topology, host: str, command: list[str]
) -> subprocess.CompletedProcess[str]:
    if topology.local:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    return run_remote(host, topology.ssh_port, command)


def prepare_generated_local_data(
    args: argparse.Namespace,
    topology: Topology,
    build_dir: Path,
    source_data_root: Path,
    campaign_dir: Path,
    tpchgen: Path,
) -> Path:
    partitions = int(args.generated_data_partitions)
    cache_root = Path(args.local_data_cache_root)
    metadata_tool = build_dir / TPCH_METADATA_BINARY
    rank_map = host_global_ranks(topology)
    strategy = "round_robin"

    def prepare(host: str) -> list[dict]:
        command = [
            sys.executable,
            str(LOCAL_DATA_HELPER),
            "prepare",
            "--cache-root",
            str(cache_root),
            "--scale",
            str(args.scale),
            "--partitions",
            str(partitions),
            "--local-ranks",
            ",".join(map(str, rank_map[host])),
            "--world-size",
            str(topology.ranks),
            "--strategy",
            strategy,
            "--generator",
            str(tpchgen),
            "--metadata-tool",
            str(metadata_tool),
        ]
        try:
            result = _run_local_data_helper(topology, host, command)
            records = json.loads(result.stdout)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
            diagnostic = getattr(error, "stderr", "") or str(error)
            raise ConfigurationError(
                f"local TPC-H preparation failed on {host}: {diagnostic.strip()}"
            ) from error
        if not isinstance(records, list):
            raise ConfigurationError(
                f"local TPC-H preparation returned invalid metadata on {host}"
            )
        return records

    all_records: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(topology.hosts)
    ) as executor:
        futures = {
            executor.submit(prepare, host.name): host.name for host in topology.hosts
        }
        for future in concurrent.futures.as_completed(futures):
            all_records.extend(future.result())

    try:
        manifest = build_synthetic_manifest(
            all_records,
            scale=args.scale,
            partitions=partitions,
            generator=str(tpchgen),
        )
        durable = campaign_dir / "manifest" / "generated_data"
        manifest_path, success_path = write_manifest_bundle(durable, manifest)
    except (OSError, RuntimeError) as error:
        raise ConfigurationError(
            f"cannot publish generated TPC-H manifest: {error}"
        ) from error
    golden_source = source_data_root / f"sf{args.scale}" / "golden"

    def install(host: str) -> None:
        command = [
            sys.executable,
            str(LOCAL_DATA_HELPER),
            "install-manifest",
            "--cache-root",
            str(cache_root),
            "--scale",
            str(args.scale),
            "--partitions",
            str(partitions),
            "--manifest",
            str(manifest_path),
            "--success",
            str(success_path),
            "--golden-source",
            str(golden_source),
        ]
        try:
            _run_local_data_helper(topology, host, command)
        except (OSError, subprocess.CalledProcessError) as error:
            diagnostic = getattr(error, "stderr", "") or str(error)
            raise ConfigurationError(
                f"cannot install generated TPC-H manifest on {host}: "
                f"{diagnostic.strip()}"
            ) from error

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(topology.hosts)
    ) as executor:
        futures = [executor.submit(install, host.name) for host in topology.hosts]
        for future in futures:
            future.result()
    return cache_data_root(cache_root, partitions)


def active_gpu_processes() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConfigurationError(
            f"could not inspect active GPU processes: {error}"
        ) from error
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def ensure_selected_gpus_idle(topology: Topology) -> None:
    if not topology.local:
        return
    active = active_gpu_processes()
    if active:
        raise ConfigurationError(
            "selected GPUs already have active compute processes:\n  "
            + "\n  ".join(active)
        )


def capture_local_platform(
    mpirun: str, require_nsys: bool, require_ncu: bool = False
) -> dict[str, str]:
    commands: dict[str, list[str]] = {
        "uname": ["uname", "-a"],
        "gpu_inventory": [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        "gpu_topology": ["nvidia-smi", "topo", "-m"],
        "mpi_version": [mpirun, "--version"],
    }
    if require_nsys:
        commands["nsys_version"] = ["nsys", "--version"]
    if require_ncu:
        commands["ncu_version"] = ["ncu", "--version"]
    captured: dict[str, str] = {}
    for name, command in commands.items():
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            captured[name] = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            captured[name] = f"unavailable: {error}"
    return captured


def rank_library_directories(build_dir: Path) -> tuple[Path, ...]:
    directories = [
        build_dir / "src/cuda/tpch",
        build_dir / "src/cuda/alps",
        build_dir / "src/cuda",
        build_dir / "_deps/rmm-build",
        build_dir / "_deps/rapids_logger-build",
        Path("/storage/conda/x86_64/envs/eiger-b200-build/lib"),
        Path("/usr/local/cuda/targets/sbsa-linux/lib"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/opt/hpcx/ompi/lib"),
    ]
    directories.extend(
        Path(directory)
        for directory in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if directory
    )
    return tuple(dict.fromkeys(directories))


def capture_nccl_configuration(
    topology: Topology,
    build_dir: Path,
    inspect_remote: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    helper = SCRIPT_DIR / "nccl_config_preflight.py"
    library_directories = rank_library_directories(build_dir)
    reports: dict[str, dict[str, object]] = {}
    if topology.local:
        reports[topology.hosts[0].name] = inspect_configuration(
            library_directories=library_directories
        )
    elif inspect_remote:
        helper_args = [
            str(helper),
            "--json",
            *(
                argument
                for directory in library_directories
                for argument in ("--library-dir", str(directory))
            ),
        ]
        for host in topology.hosts:
            try:
                result = run_remote(host.name, topology.ssh_port, helper_args)
                report = json.loads(result.stdout.strip().splitlines()[-1])
            except (
                OSError,
                subprocess.CalledProcessError,
                json.JSONDecodeError,
                IndexError,
            ) as error:
                raise ConfigurationError(
                    f"cannot inspect NCCL configuration on {host.name}: {error}"
                ) from error
            if not isinstance(report, dict):
                raise ConfigurationError(
                    f"invalid NCCL configuration report from {host.name}"
                )
            reports[host.name] = report

    conflicts = [
        f"{host}: {conflict}"
        for host, report in reports.items()
        for conflict in report.get("conflicts", [])
    ]
    if conflicts:
        raise ConfigurationError(
            "NCCL CTA policy is controlled by external configuration:\n  "
            + "\n  ".join(conflicts)
        )

    sources: dict[str, object] = {}
    observed: dict[str, object] = {}
    for host, report in reports.items():
        sources[host] = {
            "configuration_files": report.get("configuration_files", []),
            "environment_plugin": report.get("environment_plugin", {}),
        }
        host_observed = report.get("observed_environment", {})
        observed[host] = host_observed
        if not topology.local and host_observed:
            rendered = ", ".join(
                f"{key}={value!r}" for key, value in sorted(host_observed.items())
            )
            print(
                "warning: inherited NCCL CTA controls on "
                f"{host} will be ignored by rank launchers: {rendered}",
                file=sys.stderr,
            )
    return sources, observed


def run_profile_gpu_preflight(
    args: argparse.Namespace,
    topology: Topology,
    selected_environment: dict[str, str],
    saved_selection: dict[str, object] | None = None,
) -> dict[str, object]:
    candidates: list[int] | None = None
    if topology.local:
        try:
            all_candidates = physical_gpu_candidates(
                selected_environment.get("CUDA_VISIBLE_DEVICES", "<unset>"),
                topology.gpus_per_node,
            )
        except (OSError, subprocess.CalledProcessError, PreflightError) as error:
            raise ConfigurationError(
                f"cannot resolve Nsight GPU candidates: {error}"
            ) from error
        candidates = all_candidates

    if saved_selection is not None:
        saved_gpu = saved_selection.get("gpu_index")
        if not isinstance(saved_gpu, int):
            raise ConfigurationError(f"invalid saved Nsight GPU: {saved_gpu!r}")
        if candidates is not None and saved_gpu not in candidates:
            raise ConfigurationError(
                f"saved Nsight GPU {saved_gpu!r} is not currently selected"
            )
        candidates = [saved_gpu]

    target_host = topology.hosts[0].name
    helper_args = [
        "--timeout-seconds",
        str(args.profile_preflight_timeout),
        "--sample-seconds",
        str(args.profile_preflight_seconds),
        "--frequency",
        str(args.profile_gpu_metrics_frequency),
    ]
    if candidates is None:
        helper_args.extend(
            [
                "--expected-gpus",
                str(topology.gpus_per_node),
                "--cuda-visible-devices",
                selected_environment.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            ]
        )
    else:
        helper_args.extend(["--candidates", ",".join(map(str, candidates))])
    if topology.local:
        assert candidates is not None
        result = select_gpu(
            candidates,
            args.profile_preflight_timeout,
            args.profile_preflight_seconds,
            args.profile_gpu_metrics_frequency,
        )
    else:
        command = [str(SCRIPT_DIR / "nsys_preflight.py"), *helper_args]
        try:
            completed = run_remote(target_host, topology.ssh_port, command)
            output = completed.stdout
        except subprocess.CalledProcessError as error:
            output = error.stdout or ""
            if not output.strip():
                raise ConfigurationError(
                    f"Nsight GPU preflight failed on {target_host}: {error.stderr or error}"
                ) from error
        try:
            result = json.loads(output.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise ConfigurationError(
                f"Nsight GPU preflight returned invalid output on {target_host}: {output!r}"
            ) from error

    if result.get("status") != "pass" or not isinstance(result.get("gpu_index"), int):
        summaries = []
        for attempt in result.get("attempts", []):
            capture = attempt.get("capture", {})
            summaries.append(
                f"GPU {attempt.get('gpu_index')}: {attempt.get('reason')} "
                f"(timeout={capture.get('timed_out')}, exit={capture.get('exit_code')})"
            )
        raise ConfigurationError(
            "no selected GPU produced nonempty Nsight hardware metrics within "
            f"the timeout on {target_host}: " + "; ".join(summaries)
        )
    return {
        "timestamp_utc": utc_now(),
        "host": target_host,
        "gpu_index": result["gpu_index"],
        "metrics_rank": 0,
        "frequency": args.profile_gpu_metrics_frequency,
        "attempts": result["attempts"],
    }


def run_ncu_preflight(
    args: argparse.Namespace,
    topology: Topology,
    binaries: dict[str, Path],
) -> dict[str, object]:
    target_host = topology.hosts[0].name
    section_file = Path(args.ncu_metrics_file)
    if topology.local:
        try:
            result = run_ncu_counter_preflight(
                binaries["ncu_probe"],
                section_file,
                args.ncu_preflight_timeout,
                kernel_substrings=args.ncu_kernels,
            )
        except (OSError, NcuProfileError) as error:
            raise ConfigurationError(f"NCU preflight failed: {error}") from error
    else:
        command = [
            str(SCRIPT_DIR / "ncu_profile.py"),
            "--probe",
            str(binaries["ncu_probe"]),
            "--section-file",
            str(section_file),
            "--kernels-file",
            str(args.ncu_kernels_file),
            "--timeout-seconds",
            str(args.ncu_preflight_timeout),
        ]
        try:
            completed = run_remote(target_host, topology.ssh_port, command)
            output = completed.stdout
        except subprocess.CalledProcessError as error:
            output = error.stdout or ""
            if not output.strip():
                raise ConfigurationError(
                    f"NCU preflight failed on {target_host}: {error.stderr or error}"
                ) from error
        try:
            result = json.loads(output.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise ConfigurationError(
                f"NCU preflight returned invalid output on {target_host}: {output!r}"
            ) from error

    if result.get("status") != "pass":
        capture = result.get("capture", {})
        reason = result.get("reason", "unknown failure")
        raise ConfigurationError(
            f"rank-0 NCU preflight failed on {target_host}: {reason} "
            f"(timeout={capture.get('timed_out')}, exit={capture.get('exit_code')})"
        )
    if list(result.get("metrics", [])) != list(args.ncu_metrics):
        raise ConfigurationError(
            f"rank-0 NCU metric selectors differ on {target_host}: "
            f"campaign={args.ncu_metrics!r}, "
            f"rank0={result.get('metrics', [])!r}"
        )
    if list(result.get("launch_metrics", [])) != list(args.ncu_launch_metrics):
        raise ConfigurationError(
            f"rank-0 NCU LaunchStats metrics differ on {target_host}: "
            f"campaign={args.ncu_launch_metrics!r}, "
            f"rank0={result.get('launch_metrics', [])!r}"
        )
    if list(result.get("kernels", [])) != list(args.ncu_kernels):
        raise ConfigurationError(
            f"rank-0 NCU kernel subset differs on {target_host}: "
            f"campaign={args.ncu_kernels!r}, rank0={result.get('kernels', [])!r}"
        )
    if result.get("kernel_expression") != args.ncu_kernel_expression:
        raise ConfigurationError(
            f"rank-0 NCU kernel expression differs on {target_host}: "
            f"campaign={args.ncu_kernel_expression!r}, "
            f"rank0={result.get('kernel_expression')!r}"
        )
    return {
        "timestamp_utc": utc_now(),
        "host": target_host,
        "rank": 0,
        "logical_gpu": 0,
        "metrics": result["metrics"],
        "launch_metrics": result["launch_metrics"],
        "kernels": result["kernels"],
        "kernel_expression": result["kernel_expression"],
        "version": result.get("version", "unknown"),
        "validation": result.get("validation", {}),
    }


def git_metadata() -> dict[str, object]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {"head": git("rev-parse", "HEAD"), "status": git("status", "--short")}
    except (OSError, subprocess.CalledProcessError):
        return {"head": "unknown", "status": "unavailable"}


def validate_plan_configuration(
    arms: tuple[str, ...],
    stages: tuple[str, ...],
    plan_dir: Path | None,
    queries: tuple[int, ...],
) -> None:
    if "messages" in stages and arms != ("legacy",):
        raise ConfigurationError("the messages stage currently requires --arms legacy")
    wants_synthesized = any(
        ARMS[arm].plan_source == PlanSource.SYNTHESIZED for arm in arms
    )
    wants_synthesis = "synthesis" in stages
    if wants_synthesis and not wants_synthesized:
        raise ConfigurationError("the synthesis stage requires the synthesized arm")
    if wants_synthesis and plan_dir is not None:
        raise ConfigurationError(
            "--physical-plan-dir cannot be combined with the synthesis stage"
        )
    if wants_synthesized and not wants_synthesis and plan_dir is None:
        raise ConfigurationError(
            "the synthesized arm requires the synthesis stage or --physical-plan-dir"
        )
    if plan_dir is not None:
        if not plan_dir.is_dir():
            raise ConfigurationError(
                f"physical-plan directory does not exist: {plan_dir}"
            )
        missing = [q for q in queries if not (plan_dir / f"q{q:02d}.json").is_file()]
        if missing:
            raise ConfigurationError(
                "missing guarded physical plans: "
                + ",".join(f"q{query:02d}.json" for query in missing)
            )


def command_environment(
    args: argparse.Namespace, data_root: Path
) -> tuple[dict[str, str], dict[str, str]]:
    profile_rank_launcher = resolve_path(
        Path(
            os.environ.get(
                "ALPS_TPCH_PROFILE_RANK_LAUNCHER",
                REPOSITORY / "script" / "profile_distributed_tpch_rank.sh",
            )
        )
    )
    ncu_rank_launcher = resolve_path(
        Path(
            os.environ.get(
                "ALPS_TPCH_NCU_RANK_LAUNCHER",
                REPOSITORY / "script" / "profile_distributed_tpch_ncu_rank.sh",
            )
        )
    )
    selected = {
        "TPCH_DATA_PATH": str(data_root),
        "GAMAX_RMM_RESOURCE": "arena",
        "GAMAX_MEM_RESERVE_BYTES": str(args.memory_reserve_bytes),
        "NCCL_SOCKET_IFNAME": args.nccl_socket_ifname,
        "NCCL_MNNVL_ENABLE": os.environ.get("NCCL_MNNVL_ENABLE", "1"),
        "MPIRUN": (
            str(resolve_path(Path(args.mpirun))) if "/" in args.mpirun else args.mpirun
        ),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "ALPS_TPCH_PROFILE_RANK_LAUNCHER": str(profile_rank_launcher),
        "ALPS_TPCH_NCU_RANK_LAUNCHER": str(ncu_rank_launcher),
    }
    environment = os.environ.copy()
    environment.update(
        {key: value for key, value in selected.items() if value != "<unset>"}
    )
    for key in NCCL_CTA_ENVIRONMENT_KEYS:
        environment.pop(key, None)
        selected[key] = "<unset>"
    return environment, selected


def inherited_nccl_cta_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values = os.environ if source is None else source
    return {key: values[key] for key in NCCL_CTA_ENVIRONMENT_KEYS if key in values}


def arm_environment(
    base_environment: dict[str, str],
    base_selected: dict[str, str],
    arm_name: str,
) -> tuple[dict[str, str], dict[str, str]]:
    environment = base_environment.copy()
    selected = base_selected.copy()
    for key in NCCL_CTA_ENVIRONMENT_KEYS:
        environment.pop(key, None)
        selected[key] = "<unset>"
    for key, value in ARMS[arm_name].environment_overrides:
        environment[key] = value
        selected[key] = value
    return environment, selected


def effective_memory_reserve_bytes(args: argparse.Namespace, query: int) -> int:
    overrides = getattr(args, "query_memory_reserve_bytes", {})
    if isinstance(overrides, Mapping):
        value = overrides.get(query, overrides.get(str(query)))
        if value is not None:
            return int(value)
    return int(getattr(args, "memory_reserve_bytes", DEFAULT_MEMORY_RESERVE_BYTES))


def query_environment(
    base_environment: dict[str, str],
    base_selected: dict[str, str],
    args: argparse.Namespace,
    query: int,
) -> tuple[dict[str, str], dict[str, str]]:
    environment = base_environment.copy()
    selected = base_selected.copy()
    reserve_bytes = str(effective_memory_reserve_bytes(args, query))
    environment["GAMAX_MEM_RESERVE_BYTES"] = reserve_bytes
    selected["GAMAX_MEM_RESERVE_BYTES"] = reserve_bytes
    return environment, selected


def recorded_memory_reserve_bytes(
    selected_environment: Mapping[str, str],
    args: argparse.Namespace,
    query: int,
) -> int:
    return int(
        selected_environment.get(
            "GAMAX_MEM_RESERVE_BYTES",
            str(effective_memory_reserve_bytes(args, query)),
        )
    )


def arm_policy(name: str) -> dict[str, object]:
    arm = ARMS[name]
    _, effective = arm_environment({}, {}, name)
    return {
        "variant_of": arm.variant_of,
        "plan_source": arm.plan_source.value,
        "benchmark_args": list(arm.benchmark_args),
        "environment": {key: effective[key] for key in NCCL_CTA_ENVIRONMENT_KEYS},
    }


def render_runtime_policy(selected: Mapping[str, str]) -> str:
    keys = (
        "GAMAX_MEM_RESERVE_BYTES",
        "NCCL_MIN_CTAS",
        "NCCL_MAX_CTAS",
    )
    return " ".join(f"{key}={selected.get(key, '<unset>')}" for key in keys)


def required_remote_executables(
    binaries: dict[str, Path],
    required_names: tuple[str, ...],
    stages: tuple[str, ...],
    selected_environment: dict[str, str],
) -> tuple[Path, ...]:
    paths = [
        *(binaries[name] for name in required_names),
        SCRIPT_DIR / "nccl_config_preflight.py",
    ]
    if "profiles" in stages:
        rank_launcher = Path(selected_environment["ALPS_TPCH_PROFILE_RANK_LAUNCHER"])
        if not rank_launcher.is_file() or not os.access(rank_launcher, os.X_OK):
            raise ConfigurationError(
                f"profile rank launcher is not executable: {rank_launcher}"
            )
        paths.append(rank_launcher)
        preflight_helper = SCRIPT_DIR / "nsys_preflight.py"
        if not preflight_helper.is_file() or not os.access(preflight_helper, os.X_OK):
            raise ConfigurationError(
                f"Nsight preflight helper is not executable: {preflight_helper}"
            )
        paths.append(preflight_helper)
    if "ncu_profiles" in stages:
        rank_launcher = Path(selected_environment["ALPS_TPCH_NCU_RANK_LAUNCHER"])
        if not rank_launcher.is_file() or not os.access(rank_launcher, os.X_OK):
            raise ConfigurationError(
                f"NCU rank launcher is not executable: {rank_launcher}"
            )
        paths.extend((rank_launcher, SCRIPT_DIR / "ncu_profile.py"))
    return tuple(paths)


def base_launcher_args(
    build_dir: Path, data_root: Path, topology: Topology
) -> list[str]:
    return [
        "--build-dir",
        str(build_dir),
        "--tpch-data",
        str(data_root),
        *topology.launcher_args(),
    ]


def common_benchmark_args(args: argparse.Namespace) -> list[str]:
    result = [
        *COMMON_BENCHMARK_ARGS,
        "--parquet-decode-threads",
        str(
            getattr(
                args,
                "parquet_decode_threads",
                DEFAULT_PARQUET_DECODE_THREADS,
            )
        ),
    ]
    if getattr(args, "generate_local_data", False):
        result.append("--skip-dataset-preflight")
    return result


def synthesis_command(
    args: argparse.Namespace,
    build_dir: Path,
    data_root: Path,
    topology: Topology,
    query: int,
    synthesis_dir: Path,
) -> list[str]:
    command = [
        str(REPOSITORY / "script" / "run_distributed_tpch_synthesis.sh"),
        "--scale",
        str(args.scale),
        "--queries",
        str(query),
        "--output-dir",
        str(synthesis_dir),
        "--query-dir-layout",
        "--attempt-timeout",
        str(args.synthesis_attempt_timeout),
        "--max-restarts",
        str(args.synthesis_max_restarts),
        "--seed",
        str(args.synthesis_seed),
        *base_launcher_args(build_dir, data_root, topology),
    ]
    if getattr(args, "command", None) == "retry":
        command.append("--force")
    command.extend(
        [
            "--",
            *args.synthesis_arg,
            *(
                ["--continue-on-verification-failure"]
                if getattr(
                    args, "synthesis_continue_on_verify_failure", False
                )
                else []
            ),
            "--parquet-decode-threads",
            str(
                getattr(
                    args,
                    "parquet_decode_threads",
                    DEFAULT_PARQUET_DECODE_THREADS,
                )
            ),
            *(
                ["--skip-dataset-preflight"]
                if getattr(args, "generate_local_data", False)
                else []
            ),
        ]
    )
    return command


def benchmark_command(
    args: argparse.Namespace,
    build_dir: Path,
    data_root: Path,
    topology: Topology,
    arm_name: str,
    stage_name: str,
    query: int,
    attempt: Path,
    plan_dir: Path | None,
) -> list[str]:
    arm = ARMS[arm_name]
    stage = STAGES[stage_name]
    if stage.executor == ExecutorKind.NSYS:
        command = [
            str(REPOSITORY / "script" / "profile_distributed_tpch_nsys.sh"),
            "--query",
            str(query),
            "--scale",
            str(args.scale),
            "--output-dir",
            str(attempt),
            "--gpu-metrics-frequency",
            str(args.profile_gpu_metrics_frequency),
            *base_launcher_args(build_dir, data_root, topology),
        ]
        if topology.local:
            command.extend(
                [
                    "--single-report",
                    "--gpu-metrics-devices",
                    str(args.profile_gpu_index),
                ]
            )
        else:
            command.extend(
                [
                    "--profile-ranks",
                    "all",
                    "--gpu-metrics-devices",
                    str(args.profile_gpu_index),
                    "--gpu-metrics-rank",
                    str(args.profile_metrics_rank),
                ]
            )
    elif stage.executor == ExecutorKind.NCU:
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=30",
            str(args.ncu_capture_timeout),
            str(REPOSITORY / "script" / "profile_distributed_tpch_ncu.sh"),
            "--query",
            str(query),
            "--scale",
            str(args.scale),
            "--output-dir",
            str(attempt),
            "--section-file",
            str(args.ncu_metrics_file),
            "--kernel-name",
            args.ncu_kernel_expression,
            *base_launcher_args(build_dir, data_root, topology),
        ]
    else:
        command = [
            str(REPOSITORY / "script" / "run_distributed_tpch_bench.sh"),
            "--scale",
            str(args.scale),
            "--queries",
            str(query),
            "--output-dir",
            str(attempt),
            "--fail-fast",
            *base_launcher_args(build_dir, data_root, topology),
        ]
    benchmark_args = [
        *common_benchmark_args(args),
        *stage.benchmark_args,
        *arm.benchmark_args,
    ]
    if getattr(args, "collect_on_verify_failure", False):
        benchmark_args.extend(
            [
                "--collect-on-verify-failure",
                "--verification-json",
                str(attempt / "verification.json"),
            ]
        )
    if arm.plan_source == PlanSource.SYNTHESIZED:
        assert plan_dir is not None
        benchmark_args.extend(
            ["--physical-plan-config", str(plan_dir / f"q{query:02d}" / "plan.json")]
        )
    if stage_name == "memory":
        benchmark_args.extend(["--memory-csv", str(attempt / "memory.csv")])
    if stage_name == "messages":
        benchmark_args.extend(["--messages-csv", str(attempt / "messages.csv")])
    if arm.uses_ce_pull:
        benchmark_args.extend(["--ce-pull-csv", str(attempt / "ce_pull.csv")])
    command.extend(["--", *benchmark_args])
    return command


def combined_timing_breakdown_command(
    args: argparse.Namespace,
    build_dir: Path,
    data_root: Path,
    topology: Topology,
    arm_name: str,
    query: int,
    timing_attempt: Path,
    breakdown_attempt: Path,
    plan_dir: Path | None,
) -> list[str]:
    arm = ARMS[arm_name]
    command = [
        str(REPOSITORY / "script" / "run_distributed_tpch_bench.sh"),
        "--scale",
        str(args.scale),
        "--queries",
        str(query),
        "--output-dir",
        str(timing_attempt),
        "--fail-fast",
        *base_launcher_args(build_dir, data_root, topology),
    ]
    benchmark_args = [
        *common_benchmark_args(args),
        *COMBINED_TIMING_BREAKDOWN_ARGS,
        *arm.benchmark_args,
        "--steps-csv",
        str(breakdown_attempt / "steps.csv"),
        "--phases-csv",
        str(breakdown_attempt / "phases.csv"),
    ]
    if getattr(args, "collect_on_verify_failure", False):
        benchmark_args.extend(
            [
                "--collect-on-verify-failure",
                "--verification-json",
                str(timing_attempt / "verification.json"),
            ]
        )
    if arm.plan_source == PlanSource.SYNTHESIZED:
        assert plan_dir is not None
        benchmark_args.extend(
            ["--physical-plan-config", str(plan_dir / f"q{query:02d}" / "plan.json")]
        )
    if arm.uses_ce_pull:
        benchmark_args.extend(["--ce-pull-csv", str(timing_attempt / "ce_pull.csv")])
    command.extend(["--", *benchmark_args])
    return command


def expected_artifacts(
    arm: str, stage: str, query: int, attempt: Path, topology: Topology
) -> list[str]:
    expected: list[str]
    if stage == "timing":
        expected = ["e2e.csv"]
    elif stage == "breakdown":
        expected = ["steps.csv", "phases.csv"]
    elif stage == "memory":
        expected = ["memory.csv"]
    elif stage == "messages":
        expected = ["messages.csv"]
    elif stage == "profiles":
        if topology.local:
            return [f"q{query:02d}.nsys-rep"]
        reports = sorted(attempt.glob(f"q{query:02d}_*_rank*.nsys-rep"))
        if len(reports) != topology.ranks:
            return [f"__missing_{topology.ranks}_rank_profile_set__"]
        return [str(path.relative_to(attempt)) for path in reports]
    elif stage == "ncu_profiles":
        reports = sorted(attempt.glob(f"q{query:02d}_*_rank0.ncu-rep"))
        if len(reports) != 1:
            return ["__missing_rank0_ncu_profile__"]
        return [
            str(reports[0].relative_to(attempt)),
            f"q{query:02d}_ncu_metrics.csv",
            f"q{query:02d}_ncu_validation.json",
        ]
    else:
        raise AssertionError(stage)
    if ARMS[arm].uses_ce_pull:
        expected.append("ce_pull.csv")
    return expected


def verification_result(attempt: Path) -> dict[str, object] | None:
    path = attempt / "verification.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("status") not in {
        "pass",
        "mismatch",
    }:
        return None
    return record


def add_measurement_verdict(
    result: dict[str, object], attempt: Path, measurements_complete: bool
) -> None:
    verification = verification_result(attempt)
    if verification is None:
        return
    result["verification_status"] = verification["status"]
    result["measurements_complete"] = measurements_complete
    if verification["status"] == "mismatch":
        result["failure_kind"] = "verification"


def run_command(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
) -> tuple[int, float]:
    start = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        exit_code = process.wait()
    return exit_code, time.monotonic() - start


def validate_ncu_attempt(
    args: argparse.Namespace, attempt: Path, query: int
) -> bool:
    reports = sorted(attempt.glob(f"q{query:02d}_*_rank0.ncu-rep"))
    validation_path = attempt / f"q{query:02d}_ncu_validation.json"
    csv_path = attempt / f"q{query:02d}_ncu_metrics.csv"
    if len(reports) != 1:
        write_json(
            validation_path,
            {
                "status": "fail",
                "reason": f"expected one rank-0 NCU report, found {len(reports)}",
                "reports": [str(path) for path in reports],
            },
        )
        return False
    try:
        validation, raw_csv = inspect_ncu_report(
            reports[0],
            args.ncu_metrics,
            args.ncu_launch_metrics,
            args.ncu_preflight_timeout,
            kernel_substrings=args.ncu_kernels,
        )
    except (OSError, NcuProfileError) as error:
        validation = {"status": "fail", "reason": str(error)}
        raw_csv = ""
    csv_path.write_text(raw_csv)
    write_json(validation_path, validation)
    return validation.get("status") == "pass"


def run_synthesis_stage(
    args: argparse.Namespace,
    build_dir: Path,
    data_root: Path,
    topology: Topology,
    queries: tuple[int, ...],
    synthesis_dir: Path,
    output_root: Path,
    environment: dict[str, str],
    selected_environment: dict[str, str],
) -> bool:
    failed = False
    for query in queries:
        query_dir = synthesis_dir / f"q{query:02d}"
        plan = query_dir / "plan.json"
        if args.resume and args.command != "retry" and is_nonempty_file(plan):
            print(f"SKIP synthesized/synthesis Q{query}: {plan} exists")
            if not args.dry_run:
                append_status(
                    output_root,
                    "synthesized",
                    "synthesis",
                    query,
                    "skip_pass",
                    0,
                    query_dir,
                )
            continue
        command = synthesis_command(
            args, build_dir, data_root, topology, query, synthesis_dir
        )
        query_command_environment, query_selected_environment = query_environment(
            environment, selected_environment, args, query
        )
        if args.dry_run:
            print(
                f"DRY-RUN synthesized/synthesis Q{query}\n"
                f"  environment: {render_runtime_policy(query_selected_environment)}\n"
                f"  {shlex.join(command)}"
            )
            continue
        print(
            f"RUN synthesized/synthesis Q{query}\n"
            f"  environment: {render_runtime_policy(query_selected_environment)}\n"
            f"  {shlex.join(command)}"
        )
        query_dir.mkdir(parents=True, exist_ok=True)
        exit_code, _ = run_command(
            command, query_command_environment, query_dir / "driver.log"
        )
        passed = exit_code == 0 and is_nonempty_file(plan)
        effective_exit_code = exit_code if exit_code != 0 else 1
        append_status(
            output_root,
            "synthesized",
            "synthesis",
            query,
            "pass" if passed else "fail",
            effective_exit_code if not passed else 0,
            query_dir,
        )
        if not passed:
            failed = True
            if not args.keep_going:
                return False
    return not failed


def run_evaluation_cell(
    args: argparse.Namespace,
    build_dir: Path,
    data_root: Path,
    topology: Topology,
    arm: str,
    stage: str,
    query: int,
    query_root: Path,
    plan_dir: Path | None,
    output_root: Path,
    environment: dict[str, str],
    selected_environment: dict[str, str],
) -> bool:
    if args.resume:
        previous = successful_attempt(query_root)
        if previous is not None:
            print(f"SKIP {arm}/{stage} Q{query}: passed in {previous.name}")
            if not args.dry_run:
                append_status(output_root, arm, stage, query, "skip_pass", 0, previous)
            return True

    if (
        not args.dry_run
        and plan_dir is not None
        and not is_nonempty_file(plan_dir / f"q{query:02d}" / "plan.json")
    ):
        record_missing_plan_dependency(
            output_root,
            arm,
            (stage,),
            query,
            plan_dir,
        )
        return False

    if args.dry_run:
        attempt = next_attempt_path(query_root)
        command = benchmark_command(
            args, build_dir, data_root, topology, arm, stage, query, attempt, plan_dir
        )
        print(
            f"DRY-RUN {arm}/{stage} Q{query}\n"
            f"  environment: {render_runtime_policy(selected_environment)}\n"
            f"  {shlex.join(command)}"
        )
        return True

    attempt = next_attempt_dir(query_root)
    command = benchmark_command(
        args, build_dir, data_root, topology, arm, stage, query, attempt, plan_dir
    )
    write_json(
        attempt / "command.json",
        {
            "argv": command,
            "environment": selected_environment,
            "started_utc": utc_now(),
        },
    )
    print(
        f"RUN {arm}/{stage} Q{query}\n"
        f"  environment: {render_runtime_policy(selected_environment)}\n"
        f"  {shlex.join(command)}"
    )
    exit_code, elapsed = run_command(command, environment, attempt / "stdout.log")
    ncu_valid = True
    if stage == "ncu_profiles":
        ncu_valid = validate_ncu_attempt(args, attempt, query)
    expected = expected_artifacts(arm, stage, query, attempt, topology)
    measurements_complete = all(
        (attempt / relative).is_file() and (attempt / relative).stat().st_size > 0
        for relative in expected
    )
    passed = exit_code == 0 and measurements_complete and ncu_valid
    effective_exit_code = exit_code if exit_code != 0 or passed else 1
    result = {
        "arm": arm,
        "stage": stage,
        "query": query,
        "status": "pass" if passed else "fail",
        "exit_code": effective_exit_code,
        "elapsed_seconds": elapsed,
        "memory_reserve_bytes": recorded_memory_reserve_bytes(
            selected_environment, args, query
        ),
        "expected_artifacts": expected,
        "finished_utc": utc_now(),
    }
    add_measurement_verdict(result, attempt, measurements_complete)
    write_json(attempt / "result.json", result)
    append_status(
        output_root,
        arm,
        stage,
        query,
        result["status"],
        effective_exit_code if not passed else 0,
        attempt,
    )
    return passed


def link_or_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def run_combined_timing_breakdown_cell(
    args: argparse.Namespace,
    build_dir: Path,
    data_root: Path,
    topology: Topology,
    arm: str,
    query: int,
    timing_root: Path,
    breakdown_root: Path,
    plan_dir: Path | None,
    campaign_dir: Path,
    environment: dict[str, str],
    selected_environment: dict[str, str],
) -> bool:
    previous_timing = None
    previous_breakdown = None
    if args.resume:
        previous_timing = successful_attempt(timing_root)
        previous_breakdown = successful_attempt(breakdown_root)
        if previous_timing is not None and previous_breakdown is not None:
            print(
                f"SKIP {arm}/timing+breakdown Q{query}: passed in "
                f"{previous_timing.name} and {previous_breakdown.name}"
            )
            if not args.dry_run:
                append_status(
                    campaign_dir,
                    arm,
                    "timing",
                    query,
                    "skip_pass",
                    0,
                    previous_timing,
                )
                append_status(
                    campaign_dir,
                    arm,
                    "breakdown",
                    query,
                    "skip_pass",
                    0,
                    previous_breakdown,
                )
            return True

    if (
        not args.dry_run
        and plan_dir is not None
        and not is_nonempty_file(plan_dir / f"q{query:02d}" / "plan.json")
    ):
        blocked_stages: list[str] = []
        for stage, previous in (
            ("timing", previous_timing),
            ("breakdown", previous_breakdown),
        ):
            if previous is not None:
                print(f"SKIP {arm}/{stage} Q{query}: passed in {previous.name}")
                append_status(
                    campaign_dir,
                    arm,
                    stage,
                    query,
                    "skip_pass",
                    0,
                    previous,
                )
            else:
                blocked_stages.append(stage)
        record_missing_plan_dependency(
            campaign_dir,
            arm,
            tuple(blocked_stages),
            query,
            plan_dir,
        )
        return False

    if args.dry_run:
        timing_attempt = next_attempt_path(timing_root)
        breakdown_attempt = next_attempt_path(breakdown_root)
        command = combined_timing_breakdown_command(
            args,
            build_dir,
            data_root,
            topology,
            arm,
            query,
            timing_attempt,
            breakdown_attempt,
            plan_dir,
        )
        print(
            f"DRY-RUN {arm}/timing+breakdown Q{query}\n"
            f"  environment: {render_runtime_policy(selected_environment)}\n"
            f"  {shlex.join(command)}"
        )
        return True

    timing_attempt = next_attempt_dir(timing_root)
    breakdown_attempt = next_attempt_dir(breakdown_root)
    command = combined_timing_breakdown_command(
        args,
        build_dir,
        data_root,
        topology,
        arm,
        query,
        timing_attempt,
        breakdown_attempt,
        plan_dir,
    )
    command_record = {
        "argv": command,
        "environment": selected_environment,
        "combined_stages": ["timing", "breakdown"],
        "started_utc": utc_now(),
    }
    write_json(timing_attempt / "command.json", command_record)
    write_json(breakdown_attempt / "command.json", command_record)
    print(
        f"RUN {arm}/timing+breakdown Q{query}\n"
        f"  environment: {render_runtime_policy(selected_environment)}\n"
        f"  {shlex.join(command)}"
    )
    exit_code, elapsed = run_command(
        command, environment, timing_attempt / "stdout.log"
    )
    link_or_copy(timing_attempt / "stdout.log", breakdown_attempt / "stdout.log")
    link_or_copy(
        timing_attempt / "logs" / f"q{query:02d}.log",
        breakdown_attempt / "logs" / f"q{query:02d}.log",
    )
    link_or_copy(
        timing_attempt / "verification.json",
        breakdown_attempt / "verification.json",
    )
    if ARMS[arm].uses_ce_pull:
        link_or_copy(timing_attempt / "ce_pull.csv", breakdown_attempt / "ce_pull.csv")

    passed = True
    for stage, attempt in (
        ("timing", timing_attempt),
        ("breakdown", breakdown_attempt),
    ):
        expected = expected_artifacts(arm, stage, query, attempt, topology)
        measurements_complete = all(
            (attempt / relative).is_file() and (attempt / relative).stat().st_size > 0
            for relative in expected
        )
        stage_passed = exit_code == 0 and measurements_complete
        effective_exit_code = exit_code if exit_code != 0 or stage_passed else 1
        result = {
            "arm": arm,
            "stage": stage,
            "combined_stages": ["timing", "breakdown"],
            "query": query,
            "status": "pass" if stage_passed else "fail",
            "exit_code": effective_exit_code,
            "elapsed_seconds": elapsed,
            "memory_reserve_bytes": recorded_memory_reserve_bytes(
                selected_environment, args, query
            ),
            "expected_artifacts": expected,
            "peer_attempt": str(
                breakdown_attempt if stage == "timing" else timing_attempt
            ),
            "finished_utc": utc_now(),
        }
        add_measurement_verdict(result, attempt, measurements_complete)
        write_json(attempt / "result.json", result)
        append_status(
            campaign_dir,
            arm,
            stage,
            query,
            result["status"],
            effective_exit_code if not stage_passed else 0,
            attempt,
        )
        passed = passed and stage_passed
    return passed


def record_missing_plan_dependency(
    campaign_dir: Path,
    arm: str,
    stages: tuple[str, ...],
    query: int,
    plan_dir: Path,
) -> None:
    plan = plan_dir / f"q{query:02d}" / "plan.json"
    rendered_stages = "+".join(stages)
    print(
        f"BLOCK {arm}/{rendered_stages} Q{query}: synthesized plan is missing: "
        f"{plan}"
    )
    for stage in stages:
        append_status(
            campaign_dir,
            arm,
            stage,
            query,
            "blocked",
            66,
            None,
        )


def workflow_files() -> tuple[Path, ...]:
    return (
        SCRIPT_DIR / "run.py",
        LOCAL_DATA_HELPER,
        SCRIPT_DIR / "definitions.py",
        SCRIPT_DIR / "topology.py",
        SCRIPT_DIR / "artifacts.py",
        SCRIPT_DIR / "nsys_preflight.py",
        SCRIPT_DIR / "ncu_profile.py",
        DEFAULT_NCU_METRICS_FILE,
        DEFAULT_NCU_KERNELS_FILE,
        SCRIPT_DIR / "nccl_config_preflight.py",
        REPOSITORY / "script" / "distributed_tpch_common.sh",
        REPOSITORY / "script" / "run_distributed_tpch_bench.sh",
        REPOSITORY / "script" / "run_distributed_tpch_synthesis.sh",
        REPOSITORY / "script" / "profile_distributed_tpch_nsys.sh",
        REPOSITORY / "script" / "profile_distributed_tpch_rank.sh",
        REPOSITORY / "script" / "profile_distributed_tpch_ncu.sh",
        REPOSITORY / "script" / "profile_distributed_tpch_ncu_rank.sh",
    )


def environment_from_manifest(
    selected: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    current_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    if current_cuda != selected.get("CUDA_VISIBLE_DEVICES", "<unset>"):
        raise ConfigurationError(
            "CUDA_VISIBLE_DEVICES differs from the campaign "
            f"({current_cuda!r} != {selected.get('CUDA_VISIBLE_DEVICES')!r})"
        )
    environment = os.environ.copy()
    for key, value in selected.items():
        if value == "<unset>":
            environment.pop(key, None)
        else:
            environment[key] = value
    return environment, selected


def resume_namespace(configuration: dict[str, object]) -> argparse.Namespace:
    saved = configuration.get("arguments")
    if not isinstance(saved, dict):
        raise ArtifactError("campaign manifest is missing resume arguments")
    values = dict(saved)
    values.setdefault("collect_on_verify_failure", False)
    values.setdefault("synthesis_continue_on_verify_failure", False)
    values.setdefault("parquet_decode_threads", DEFAULT_PARQUET_DECODE_THREADS)
    values.setdefault("generate_local_data", False)
    values.setdefault("generated_data_partitions", None)
    values.setdefault("local_data_cache_root", "/tmp/alps-tpch")
    values.setdefault("tpchgen", "/home/scada/.local/bin/tpchgen-cli")
    values.setdefault("ncu_metrics_file", str(DEFAULT_NCU_METRICS_FILE))
    values.setdefault("ncu_metrics", [])
    values.setdefault("ncu_launch_metrics", [])
    values.setdefault("ncu_metrics_sha256", None)
    values.setdefault("ncu_kernels_file", str(DEFAULT_NCU_KERNELS_FILE))
    values.setdefault("ncu_kernels", [])
    values.setdefault("ncu_kernel_expression", None)
    values.setdefault("ncu_kernels_sha256", None)
    values.setdefault("ncu_preflight_timeout", DEFAULT_NCU_PREFLIGHT_TIMEOUT)
    values.setdefault("ncu_capture_timeout", DEFAULT_NCU_CAPTURE_TIMEOUT)
    values["query_memory_reserve_bytes"] = saved_query_memory_reserve_bytes(
        values.get("query_memory_reserve_bytes"), values.get("queries", ())
    )
    values.update(command="resume", resume=True, dry_run=False)
    return argparse.Namespace(**values)


def copy_external_plans(
    source: Path, destination: Path, queries: tuple[int, ...]
) -> None:
    for query in queries:
        query_dir = destination / f"q{query:02d}"
        query_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / f"q{query:02d}.json", query_dir / "plan.json")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    invocation_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    requested_command = args.command
    campaign_dir = resolve_path(args.campaign_dir)
    if args.command == "status":
        try:
            report = inspect_campaign_status(campaign_dir)
        except (ArtifactError, KeyError, TypeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 64
        print_campaign_status(report)
        return campaign_status_exit_code(report)

    retry_cells: tuple[StatusCell, ...] = ()
    retry_reasons: dict[StatusCell, str] = {}
    retry_cell_set: set[StatusCell] = set()
    accepted_code_drift: dict[str, object] = {}
    retry_options = None
    if requested_command == "retry":
        retry_options = {
            "arms": args.arms,
            "stages": args.stages,
            "queries": args.queries,
            "allow_code_drift": args.allow_code_drift,
            "dry_run": args.dry_run,
        }

    manifest: dict[str, object] | None = None
    configuration: dict[str, object] | None = None
    external_plan_dir: Path | None = None
    inherited_nccl_environment = inherited_nccl_cta_environment()
    if inherited_nccl_environment:
        rendered = ", ".join(
            f"{key}={value!r}"
            for key, value in sorted(inherited_nccl_environment.items())
        )
        print(
            "warning: inherited NCCL CTA controls will be ignored because "
            f"experiment arms own CTA policy: {rendered}",
            file=sys.stderr,
        )

    try:
        if args.command == "new":
            if campaign_dir.exists():
                raise ArtifactError(
                    f"campaign directory already exists: {campaign_dir}; "
                    "use the resume command for an existing campaign"
                )
            args.resume = False
            arms = parse_named_list(args.arms, ARMS, "arm")
            stages = parse_named_list(args.stages, STAGES, "stage")
            queries = parse_queries(args.queries)
            resolve_ncu_metrics(args, stages)
            args.query_memory_reserve_bytes = parse_query_memory_reserve_bytes(
                args.query_memory_reserve_bytes, queries
            )
        else:
            manifest = load_campaign(campaign_dir)
            configuration = manifest["configuration"]
            if configuration.get("schema_version") != 4:
                raise ArtifactError(
                    "this runner supports only schema-version 4 campaigns"
                )
            if requested_command == "retry":
                assert retry_options is not None
                retry_cells, retry_reasons = select_retry_cells(
                    campaign_dir,
                    configuration,
                    retry_options["arms"],
                    retry_options["stages"],
                    retry_options["queries"],
                )
                retry_cell_set = set(retry_cells)
                print_retry_selection(retry_cells, retry_reasons)
                if not retry_cells:
                    print("Nothing to retry: every selected cell has passed.")
                    return 0
            args = resume_namespace(configuration)
            args.command = requested_command
            if requested_command == "retry":
                args.dry_run = retry_options["dry_run"]
                args.allow_code_drift = retry_options["allow_code_drift"]
            arms = tuple(configuration["arms"])
            stages = tuple(configuration["stages"])
            queries = tuple(configuration["queries"])

        execution_stages = stages
        if requested_command == "retry":
            _, execution_stages, _ = retry_execution_axes(configuration, retry_cells)

        build_dir = resolve_build_dir(
            Path(args.build_dir) if args.build_dir is not None else None, stages
        )
        source_data_root = resolve_data_root(
            Path(args.tpch_data) if args.tpch_data is not None else None,
            args.scale,
            queries,
            require_tables=not getattr(args, "generate_local_data", False),
        )
        if args.command == "new" and args.physical_plan_dir:
            external_plan_dir = resolve_path(Path(args.physical_plan_dir))
        if args.command == "new":
            validate_plan_configuration(arms, stages, external_plan_dir, queries)

        binaries = {
            name: build_dir / relative for name, relative in BUILD_BINARIES.items()
        }
        topology = resolve_topology(
            args.hosts, args.gpus_per_node, args.ssh_port, binaries["device_probe"]
        )
        validate_generated_data_configuration(args, topology)
        if args.generate_local_data:
            metadata_tool = build_dir / TPCH_METADATA_BINARY
            if not metadata_tool.is_file() or not os.access(metadata_tool, os.X_OK):
                raise ConfigurationError(
                    f"required binary is not executable: {metadata_tool}"
                )
            runtime_data_root = cache_data_root(
                Path(args.local_data_cache_root), args.generated_data_partitions
            )
            runtime_tpchgen = campaign_tpchgen_path(campaign_dir)
        else:
            runtime_data_root = source_data_root
            runtime_tpchgen = None
        nccl_configuration_sources, observed_nccl_configuration = (
            capture_nccl_configuration(
                topology,
                build_dir,
                inspect_remote=not args.dry_run or args.command == "retry",
            )
        )
        if not topology.local and inherited_nccl_environment:
            observed_nccl_configuration["__coordinator__"] = inherited_nccl_environment
        if shutil.which(args.mpirun) is None and not Path(args.mpirun).is_file():
            raise ConfigurationError(f"MPI launcher is not available: {args.mpirun}")
        if "profiles" in execution_stages and shutil.which("nsys") is None:
            raise ConfigurationError("profiles requested but nsys is not on PATH")
        if "ncu_profiles" in execution_stages and shutil.which("ncu") is None:
            raise ConfigurationError("ncu_profiles requested but ncu is not on PATH")

        required_names = required_binary_names(stages)
        binary_hashes = {name: sha256(binaries[name]) for name in required_names}
        if args.command == "new":
            environment, selected_environment = command_environment(
                args, runtime_data_root
            )
        else:
            environment, selected_environment = environment_from_manifest(
                configuration["runtime_environment"]
            )
        remote_executables = required_remote_executables(
            binaries, required_names, stages, selected_environment
        )
        if args.generate_local_data:
            remote_executables = (
                *remote_executables,
                LOCAL_DATA_HELPER,
                build_dir / TPCH_METADATA_BINARY,
                runtime_tpchgen,
            )
        remote_executable_hashes = {
            str(path): sha256(
                Path(args.tpchgen)
                if args.generate_local_data and path == runtime_tpchgen
                else path
            )
            for path in remote_executables
        }
        current_workflow_hashes = {
            str(path.relative_to(REPOSITORY)): sha256(path) for path in workflow_files()
        }
        current_dataset_manifest = dataset_manifest(source_data_root, args.scale)

        plan_hashes: dict[str, str] = {}
        if external_plan_dir:
            plan_hashes = {
                f"q{query:02d}": sha256(external_plan_dir / f"q{query:02d}.json")
                for query in queries
            }

        if args.command == "new":
            git = git_metadata()
            git_worktree_status = git.pop("status", "unavailable")
            configuration = {
                "schema_version": 4,
                "repository": str(REPOSITORY),
                "git": git,
                "scale": args.scale,
                "queries": list(queries),
                "arms": list(arms),
                "stages": list(stages),
                "build_dir": str(build_dir),
                "binary_sha256": binary_hashes,
                "remote_executable_sha256": remote_executable_hashes,
                "workflow_sha256": current_workflow_hashes,
                "tpch_data": str(source_data_root),
                "dataset_manifest": current_dataset_manifest,
                "topology": topology.as_dict(),
                "runtime_environment": selected_environment,
                "nccl_configuration_sources": nccl_configuration_sources,
                "imported_plan_source": (
                    str(external_plan_dir) if external_plan_dir else None
                ),
                "imported_plan_sha256": plan_hashes,
                "arguments": {
                    "scale": args.scale,
                    "arms": list(arms),
                    "stages": list(stages),
                    "queries": list(queries),
                    "build_dir": str(build_dir),
                    "tpch_data": str(source_data_root),
                    "generate_local_data": args.generate_local_data,
                    "generated_data_partitions": args.generated_data_partitions,
                    "local_data_cache_root": str(args.local_data_cache_root),
                    "tpchgen": str(args.tpchgen),
                    "physical_plan_dir": None,
                    "hosts": topology.host_spec if not topology.local else None,
                    "gpus_per_node": topology.gpus_per_node,
                    "ssh_port": topology.ssh_port,
                    "mpirun": selected_environment["MPIRUN"],
                    "synthesis_attempt_timeout": args.synthesis_attempt_timeout,
                    "synthesis_max_restarts": args.synthesis_max_restarts,
                    "synthesis_seed": args.synthesis_seed,
                    "synthesis_arg": list(args.synthesis_arg),
                    "synthesis_continue_on_verify_failure": (
                        args.synthesis_continue_on_verify_failure
                    ),
                    "memory_reserve_bytes": args.memory_reserve_bytes,
                    "query_memory_reserve_bytes": {
                        str(query): reserve
                        for query, reserve in sorted(
                            args.query_memory_reserve_bytes.items()
                        )
                    },
                    "parquet_decode_threads": args.parquet_decode_threads,
                    "nccl_socket_ifname": args.nccl_socket_ifname,
                    "keep_going": args.keep_going,
                    "collect_on_verify_failure": args.collect_on_verify_failure,
                    "profile_preflight_timeout": args.profile_preflight_timeout,
                    "profile_preflight_seconds": args.profile_preflight_seconds,
                    "profile_gpu_metrics_frequency": (
                        args.profile_gpu_metrics_frequency
                    ),
                    "ncu_metrics_file": str(args.ncu_metrics_file),
                    "ncu_metrics": list(args.ncu_metrics),
                    "ncu_launch_metrics": list(args.ncu_launch_metrics),
                    "ncu_metrics_sha256": args.ncu_metrics_sha256,
                    "ncu_kernels_file": str(args.ncu_kernels_file),
                    "ncu_kernels": list(args.ncu_kernels),
                    "ncu_kernel_expression": args.ncu_kernel_expression,
                    "ncu_kernels_sha256": args.ncu_kernels_sha256,
                    "ncu_preflight_timeout": args.ncu_preflight_timeout,
                    "ncu_capture_timeout": args.ncu_capture_timeout,
                },
                "policy": {
                    "combine_timing_breakdown": True,
                    "common_benchmark_args": [
                        *common_benchmark_args(args),
                        *(
                            ["--collect-on-verify-failure"]
                            if args.collect_on_verify_failure
                            else []
                        ),
                    ],
                    "arms": {name: arm_policy(name) for name in arms},
                    "stage_args": {
                        name: list(STAGES[name].benchmark_args) for name in stages
                    },
                },
            }
            fingerprint = create_campaign(
                campaign_dir,
                configuration,
                args.dry_run,
                provenance={
                    "git_worktree_status": git_worktree_status,
                    "observed_nccl_cta_environment": observed_nccl_configuration,
                },
            )
            if not args.dry_run and external_plan_dir:
                copy_external_plans(
                    external_plan_dir,
                    campaign_dir / "arms" / "synthesized" / "synthesis",
                    queries,
                )
            if not args.dry_run and "ncu_profiles" in stages:
                shutil.copy2(
                    args.ncu_metrics_file,
                    campaign_dir / "manifest" / "ncu_metrics.section",
                )
                shutil.copy2(
                    args.ncu_kernels_file,
                    campaign_dir / "manifest" / "ncu_kernels.txt",
                )
        else:
            assert manifest is not None and configuration is not None
            comparisons = {
                "binary hashes": (configuration["binary_sha256"], binary_hashes),
                "remote executable hashes": (
                    configuration["remote_executable_sha256"],
                    remote_executable_hashes,
                ),
                "workflow hashes": (
                    configuration["workflow_sha256"],
                    current_workflow_hashes,
                ),
                "dataset manifest": (
                    configuration["dataset_manifest"],
                    current_dataset_manifest,
                ),
                "NCCL configuration sources": (
                    configuration["nccl_configuration_sources"],
                    nccl_configuration_sources,
                ),
                "topology": (configuration["topology"], topology.as_dict()),
            }
            accepted_code_drift = validate_campaign_comparisons(
                comparisons,
                retry=requested_command == "retry",
                allow_code_drift=(
                    args.allow_code_drift if requested_command == "retry" else False
                ),
            )
            for name, digest in configuration.get("imported_plan_sha256", {}).items():
                plan = (
                    campaign_dir
                    / "arms"
                    / "synthesized"
                    / "synthesis"
                    / name
                    / "plan.json"
                )
                if not plan.is_file() or sha256(plan) != digest:
                    raise ArtifactError(
                        f"imported campaign plan is missing or changed: {plan}"
                    )
            fingerprint = manifest["fingerprint"]
            if requested_command == "retry":
                if accepted_code_drift:
                    print(
                        "Accepted code drift: " + ", ".join(sorted(accepted_code_drift))
                    )
                if not args.dry_run:
                    record_invocation(
                        campaign_dir,
                        "retry",
                        invocation_argv,
                        details={
                            "selected_cells": [
                                {
                                    "arm": arm,
                                    "stage": stage,
                                    "query": query,
                                    "reason": retry_reasons[(arm, stage, query)],
                                }
                                for arm, stage, query in retry_cells
                            ],
                            "accepted_code_drift": accepted_code_drift,
                        },
                    )
                arms, stages, queries = retry_execution_axes(configuration, retry_cells)
            else:
                record_invocation(campaign_dir, "resume", invocation_argv)
    except (
        ArtifactError,
        ConfigurationError,
        TopologyError,
        KeyError,
        TypeError,
    ) as error:
        parser.error(str(error))

    if args.command == "new" and not args.dry_run:
        record_invocation(campaign_dir, "new", invocation_argv)

    case_root = campaign_dir / "arms"
    synthesis_dir = case_root / "synthesized" / "synthesis"
    plan_dir = synthesis_dir
    args.profile_gpu_index = 0
    args.profile_metrics_rank = 0
    if not hasattr(args, "ncu_launch_metrics"):
        args.ncu_launch_metrics = []

    print("Distributed TPC-H reproduction")
    print(f"  fingerprint: {fingerprint}")
    print(f"  scale:       {args.scale}")
    print(f"  queries:     {','.join(map(str, queries))}")
    print(f"  arms:        {','.join(arms)}")
    print(f"  stages:      {','.join(stages)}")
    print(f"  topology:    {topology.label} ({topology.host_spec})")
    print(f"  build:       {build_dir}")
    print(f"  data:        {runtime_data_root}")
    if args.generate_local_data:
        print(f"  source data: {source_data_root}")
        print(f"  partitions:  {args.generated_data_partitions} (host-local on demand)")
    print(f"  campaign:    {campaign_dir}")
    print(f"  reserve:     default={args.memory_reserve_bytes}")
    print(f"  decode:      {args.parquet_decode_threads} threads/rank")
    if args.query_memory_reserve_bytes:
        rendered_reserves = ", ".join(
            f"Q{query}={reserve}"
            for query, reserve in sorted(args.query_memory_reserve_bytes.items())
        )
        print(f"  overrides:   {rendered_reserves}")
    if "synthesis" in stages:
        print(
            "  synth verify: "
            + (
                "warn and continue after mismatch"
                if args.synthesis_continue_on_verify_failure
                else "stop or invalidate plans after mismatch"
            )
        )
    if any(stage != "synthesis" for stage in stages):
        print(
            "  verify data: "
            + (
                "collect measurements after mismatch"
                if args.collect_on_verify_failure
                else "stop current cell on mismatch"
            )
        )

    if not args.dry_run:
        if args.generate_local_data:
            assert runtime_tpchgen is not None
            try:
                stage_campaign_tpchgen(Path(args.tpchgen), runtime_tpchgen)
            except OSError as error:
                print(
                    f"error: cannot stage tpchgen-cli in the campaign: {error}",
                    file=sys.stderr,
                )
                return 2
        try:
            ensure_selected_gpus_idle(topology)
        except ConfigurationError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        try:
            remote = preflight_remote_hosts(
                topology,
                binaries["device_probe"],
                remote_executables,
                source_data_root,
                args.scale,
                queries,
                TPC_H_TABLES,
                campaign_dir,
                "profiles" in stages,
                require_table_data=not args.generate_local_data,
                require_ncu_rank0="ncu_profiles" in stages,
            )
        except TopologyError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        for host, result in remote.items():
            remote_hashes = {
                line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
                for line in result["required_executable_sha256"].splitlines()
            }
            for path, digest in remote_executable_hashes.items():
                if remote_hashes.get(path) != digest:
                    print(
                        f"error: required executable hash mismatch on {host}: {path}",
                        file=sys.stderr,
                    )
                    return 2
        if args.generate_local_data:
            print(
                "Preparing host-local TPC-H data on "
                f"{len(topology.hosts)} host(s)...",
                flush=True,
            )
            try:
                prepared_root = prepare_generated_local_data(
                    args,
                    topology,
                    build_dir,
                    source_data_root,
                    campaign_dir,
                    runtime_tpchgen,
                )
            except ConfigurationError as error:
                print(f"error: {error}", file=sys.stderr)
                return 2
            if prepared_root != runtime_data_root:
                print(
                    "error: generated TPC-H cache root changed during preparation",
                    file=sys.stderr,
                )
                return 2
        profile_gpu_record = None
        if "profiles" in stages:
            profile_gpu_path = campaign_dir / "manifest" / "profile_gpu.json"
            saved_document = None
            saved_selection = None
            if args.command in ("resume", "retry"):
                if profile_gpu_path.exists():
                    try:
                        saved_document = json.loads(profile_gpu_path.read_text())
                        saved_selection = saved_document["selected"]
                    except (
                        OSError,
                        json.JSONDecodeError,
                        KeyError,
                        TypeError,
                    ) as error:
                        print(
                            f"error: cannot load saved profile GPU selection: {error}",
                            file=sys.stderr,
                        )
                        return 2
            try:
                validation = run_profile_gpu_preflight(
                    args, topology, selected_environment, saved_selection
                )
            except ConfigurationError as error:
                print(f"error: {error}", file=sys.stderr)
                return 2
            args.profile_gpu_index = validation["gpu_index"]
            args.profile_metrics_rank = validation["metrics_rank"]
            if saved_document is None:
                profile_gpu_record = {
                    "selected": {
                        "host": validation["host"],
                        "gpu_index": validation["gpu_index"],
                        "metrics_rank": validation["metrics_rank"],
                        "frequency": validation["frequency"],
                    },
                    "validations": [validation],
                }
            else:
                profile_gpu_record = saved_document
                profile_gpu_record.setdefault("validations", []).append(validation)
            write_json(profile_gpu_path, profile_gpu_record)
            print(
                "  profile GPU: "
                f"{validation['host']} GPU {validation['gpu_index']} "
                f"(MPI rank {validation['metrics_rank']})"
            )
        ncu_profile_record = None
        if "ncu_profiles" in stages:
            ncu_profile_path = campaign_dir / "manifest" / "ncu_profile.json"
            try:
                validation = run_ncu_preflight(args, topology, binaries)
            except ConfigurationError as error:
                print(f"error: {error}", file=sys.stderr)
                return 2
            args.ncu_launch_metrics = validation["launch_metrics"]
            ncu_profile_record = {
                "selected": {
                    "host": validation["host"],
                    "rank": validation["rank"],
                    "logical_gpu": validation["logical_gpu"],
                    "metrics": validation["metrics"],
                    "launch_metrics": validation["launch_metrics"],
                    "kernels": validation["kernels"],
                    "kernel_expression": validation["kernel_expression"],
                    "version": validation["version"],
                },
                "validations": [validation],
            }
            if args.command in ("resume", "retry") and ncu_profile_path.is_file():
                try:
                    saved_ncu_profile = json.loads(ncu_profile_path.read_text())
                    saved_selection = saved_ncu_profile["selected"]
                except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
                    print(
                        f"error: cannot load saved NCU selection: {error}",
                        file=sys.stderr,
                    )
                    return 2
                if saved_selection != ncu_profile_record["selected"]:
                    print(
                        "error: rank-0 NCU preflight selection differs from "
                        "the campaign",
                        file=sys.stderr,
                    )
                    return 2
                ncu_profile_record = saved_ncu_profile
                ncu_profile_record.setdefault("validations", []).append(validation)
            write_json(ncu_profile_path, ncu_profile_record)
            print(
                "  NCU profile: "
                f"{validation['host']} rank 0 logical GPU 0, "
                f"{len(validation['metrics'])} requested + "
                f"{len(validation['launch_metrics'])} launch metrics, "
                f"{len(validation['kernels'])} kernel substrings"
            )
        write_json(
            campaign_dir / "manifest" / "preflight.json",
            {
                "timestamp_utc": utc_now(),
                "local_visible_gpus": (
                    topology.gpus_per_node if topology.local else None
                ),
                "local_platform": capture_local_platform(
                    selected_environment["MPIRUN"],
                    "profiles" in stages,
                    "ncu_profiles" in stages,
                ),
                "remote_hosts": remote,
                "nccl_configuration_sources": nccl_configuration_sources,
                "observed_nccl_cta_environment": observed_nccl_configuration,
                "profile_gpu": (
                    profile_gpu_record["selected"] if profile_gpu_record else None
                ),
                "ncu_profile": (
                    ncu_profile_record["selected"] if ncu_profile_record else None
                ),
            },
        )

    success = True
    if "synthesis" in stages:
        synthesis_queries = (
            tuple(
                query
                for query in queries
                if ("synthesized", "synthesis", query) in retry_cell_set
            )
            if args.command == "retry"
            else queries
        )
        if not args.dry_run:
            synthesis_dir.mkdir(parents=True, exist_ok=True)
        success = run_synthesis_stage(
            args,
            build_dir,
            runtime_data_root,
            topology,
            synthesis_queries,
            synthesis_dir,
            campaign_dir,
            environment,
            selected_environment,
        )
        if not success and not args.keep_going:
            return 1

    evaluation_stages = [stage for stage in stages if stage != "synthesis"]
    combine_timing_breakdown = all(
        stage in evaluation_stages for stage in ("timing", "breakdown")
    )
    for arm in arms:
        effective_environment, effective_selected_environment = arm_environment(
            environment, selected_environment, arm
        )
        arm_plan_dir = (
            plan_dir if ARMS[arm].plan_source == PlanSource.SYNTHESIZED else None
        )
        if combine_timing_breakdown:
            for query in queries:
                if args.command == "retry" and not {
                    (arm, "timing", query),
                    (arm, "breakdown", query),
                }.intersection(retry_cell_set):
                    continue
                query_command_environment, query_selected_environment = (
                    query_environment(
                        effective_environment,
                        effective_selected_environment,
                        args,
                        query,
                    )
                )
                passed = run_combined_timing_breakdown_cell(
                    args,
                    build_dir,
                    runtime_data_root,
                    topology,
                    arm,
                    query,
                    case_root / arm / "timing" / f"q{query:02d}",
                    case_root / arm / "breakdown" / f"q{query:02d}",
                    arm_plan_dir,
                    campaign_dir,
                    query_command_environment,
                    query_selected_environment,
                )
                if not passed:
                    success = False
                    if not args.keep_going:
                        return 1
        remaining_stages = [
            stage
            for stage in evaluation_stages
            if not combine_timing_breakdown or stage not in ("timing", "breakdown")
        ]
        for stage in remaining_stages:
            for query in queries:
                if (
                    args.command == "retry"
                    and (arm, stage, query) not in retry_cell_set
                ):
                    continue
                query_command_environment, query_selected_environment = (
                    query_environment(
                        effective_environment,
                        effective_selected_environment,
                        args,
                        query,
                    )
                )
                query_root = case_root / arm / stage / f"q{query:02d}"
                passed = run_evaluation_cell(
                    args,
                    build_dir,
                    runtime_data_root,
                    topology,
                    arm,
                    stage,
                    query,
                    query_root,
                    arm_plan_dir,
                    campaign_dir,
                    query_command_environment,
                    query_selected_environment,
                )
                if not passed:
                    success = False
                    if not args.keep_going:
                        return 1
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
