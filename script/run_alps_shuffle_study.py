#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Run the five-arm ALPS shuffle bandwidth study.

The campaign owns NCCL CTA policy in the same way as distributed_tpch/run.py:
ambient current/deprecated controls are removed, external configuration sources
are rejected, and only explicit arm overrides reach MPI ranks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import socket
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
PREFLIGHT = REPOSITORY / "script" / "distributed_tpch" / "nccl_config_preflight.py"
CONTROLLED_KEYS = (
    "NCCL_MIN_CTAS",
    "NCCL_MAX_CTAS",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_MIN_NRINGS",
    "NCCL_MAX_NRINGS",
)
GPU_COUNTS = (8, 24)


@dataclass(frozen=True)
class Arm:
    name: str
    benchmark_flags: tuple[str, ...]
    environment_overrides: tuple[tuple[str, str], ...] = ()
    odd_count: bool = False


ARMS = (
    Arm("nccl_auto_aligned", ("--backend", "nccl")),
    Arm(
        "p2p_ce_push_aligned",
        ("--backend", "p2p", "--copy", "ce", "--direction", "push"),
    ),
    Arm(
        "nccl_cta64_aligned",
        ("--backend", "nccl"),
        (("NCCL_MIN_CTAS", "64"), ("NCCL_MAX_CTAS", "64")),
    ),
    Arm(
        "nccl_cta64_odd",
        ("--backend", "nccl", "--odd-count"),
        (("NCCL_MIN_CTAS", "64"), ("NCCL_MAX_CTAS", "64")),
        True,
    ),
    Arm(
        "p2p_ce_push_odd",
        (
            "--backend",
            "p2p",
            "--copy",
            "ce",
            "--direction",
            "push",
            "--odd-count",
        ),
        odd_count=True,
    ),
)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_hosts(spec: str, gpus_per_node: int) -> tuple[str, ...]:
    hosts: list[str] = []
    for token in spec.split(","):
        fields = token.strip().split(":")
        if len(fields) not in (1, 2) or not fields[0]:
            raise ValueError(f"invalid host entry: {token!r}")
        if len(fields) == 2 and fields[1] != str(gpus_per_node):
            raise ValueError(
                f"host {fields[0]} declares {fields[1]} slots, expected {gpus_per_node}"
            )
        if fields[0] in hosts:
            raise ValueError(f"duplicate host: {fields[0]}")
        hosts.append(fields[0])
    required = max(GPU_COUNTS) // gpus_per_node
    if max(GPU_COUNTS) % gpus_per_node or len(hosts) < required:
        raise ValueError(
            f"the 24-GPU arm requires at least {required} hosts with {gpus_per_node} GPUs each"
        )
    return tuple(hosts)


def hosts_for_gpu_count(
    hosts: tuple[str, ...], gpus_per_node: int, gpu_count: int
) -> tuple[str, ...]:
    if gpu_count not in GPU_COUNTS or gpu_count % gpus_per_node:
        raise ValueError(f"unsupported GPU count/topology: {gpu_count}/{gpus_per_node}")
    return hosts[: gpu_count // gpus_per_node]


def arm_environment(
    base: Mapping[str, str], arm: Arm
) -> tuple[dict[str, str], dict[str, str]]:
    environment = dict(base)
    effective = {key: "<unset>" for key in CONTROLLED_KEYS}
    for key in CONTROLLED_KEYS:
        environment.pop(key, None)
    for key, value in arm.environment_overrides:
        environment[key] = value
        effective[key] = value
    return environment, effective


def rank_environment_prefix(arm: Arm) -> list[str]:
    prefix = ["env"]
    for key in CONTROLLED_KEYS:
        prefix.extend(("-u", key))
    prefix.extend(f"{key}={value}" for key, value in arm.environment_overrides)
    return prefix


def library_path(build_dir: Path, inherited: str = "") -> str:
    entries = [
        build_dir / "src" / "cuda" / "alps",
        build_dir / "src" / "cuda",
        build_dir / "_deps" / "rmm-build",
        build_dir / "_deps" / "rapids_logger-build",
        Path("/usr/local/cuda/targets/sbsa-linux/lib"),
        Path("/opt/hpcx/ompi/lib"),
    ]
    values = [str(path) for path in entries]
    if inherited:
        values.append(inherited)
    return ":".join(values)


def base_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    for key in CONTROLLED_KEYS:
        environment.pop(key, None)
    environment.update(
        {
            "LD_LIBRARY_PATH": library_path(
                args.build_dir, environment.get("LD_LIBRARY_PATH", "")
            ),
            "NCCL_SOCKET_IFNAME": args.nccl_socket_ifname,
            "NCCL_MNNVL_ENABLE": args.nccl_mnnvl_enable,
            "GAMAX_RMM_RESOURCE": args.rmm_resource,
        }
    )
    return environment


def invocation_dir(
    campaign: Path, gpu_count: int, repetition: int, arm: Arm
) -> Path:
    return campaign / "runs" / f"g{gpu_count}" / f"rep{repetition:02d}" / arm.name


def build_command(
    args: argparse.Namespace,
    hosts: tuple[str, ...],
    gpu_count: int,
    repetition: int,
    arm: Arm,
    output_csv: Path,
) -> list[str]:
    selected_hosts = hosts_for_gpu_count(hosts, args.gpus_per_node, gpu_count)
    host_spec = ",".join(f"{host}:{args.gpus_per_node}" for host in selected_hosts)
    command = [
        str(args.mpirun),
        "-np",
        str(gpu_count),
        "--host",
        host_spec,
        "--mca",
        "plm_rsh_args",
        (
            f"-p {args.ssh_port} -o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/dev/null"
        ),
        "--map-by",
        f"ppr:{args.gpus_per_node}:node",
        "-x",
        "PATH",
        "-x",
        "LD_LIBRARY_PATH",
        "-x",
        "NCCL_SOCKET_IFNAME",
        "-x",
        "NCCL_MNNVL_ENABLE",
        "-x",
        "GAMAX_RMM_RESOURCE",
        *rank_environment_prefix(arm),
        str(args.benchmark),
        *arm.benchmark_flags,
        "--msg-exp-min",
        str(args.msg_exp_min),
        "--msg-exp-max",
        str(args.msg_exp_max),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--tag",
        arm.name,
        "--out",
        str(output_csv),
    ]
    return command


def ssh_command(host: str, port: int, argv: Iterable[str]) -> list[str]:
    return [
        "ssh",
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        host,
        shlex.join(list(argv)),
    ]


def run_capture(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=True, capture_output=True, text=True)


def preflight_host(
    host: str, args: argparse.Namespace, base_env: Mapping[str, str]
) -> dict[str, object]:
    library_dirs = [
        args.build_dir / "src" / "cuda" / "alps",
        args.build_dir / "src" / "cuda",
    ]
    preflight_argv = [
        "env",
        *(
            item
            for key in CONTROLLED_KEYS
            for item in ("-u", key)
        ),
        "python3",
        str(PREFLIGHT),
        "--json",
    ]
    for directory in library_dirs:
        preflight_argv.extend(("--library-dir", str(directory)))
    config_result = run_capture(ssh_command(host, args.ssh_port, preflight_argv))
    config = json.loads(config_result.stdout.strip().splitlines()[-1])
    if config.get("conflicts"):
        raise RuntimeError(
            f"{host}: NCCL CTA configuration conflict: "
            + "; ".join(config["conflicts"])
        )

    check_script = (
        f"test -x {shlex.quote(str(args.benchmark))} && "
        f"test $(nvidia-smi -L | wc -l) -eq {args.gpus_per_node}"
    )
    run_capture(ssh_command(host, args.ssh_port, ["bash", "-c", check_script]))
    active = run_capture(
        ssh_command(
            host,
            args.ssh_port,
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader",
            ],
        )
    ).stdout.strip()
    if active:
        raise RuntimeError(f"{host}: active GPU processes: {active}")
    inventory = run_capture(
        ssh_command(
            host,
            args.ssh_port,
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
        )
    ).stdout.strip()
    remote_hash = run_capture(
        ssh_command(host, args.ssh_port, ["sha256sum", str(args.benchmark)])
    ).stdout.split()[0]
    uname = run_capture(ssh_command(host, args.ssh_port, ["uname", "-a"])).stdout.strip()
    return {
        "configuration": config,
        "gpu_inventory": inventory,
        "benchmark_sha256": remote_hash,
        "uname": uname,
        "controlled_rank_environment": {
            key: base_env.get(key, "<unset>") for key in CONTROLLED_KEYS
        },
    }


def arm_order(repetition: int, gpu_count: int) -> tuple[Arm, ...]:
    offset = (repetition - 1 + (1 if gpu_count == 24 else 0)) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def read_status(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as stream:
        return {
            (int(row["g"]), int(row["repetition"]), row["arm"])
            for row in csv.DictReader(stream)
            if row["status"] == "pass"
        }


def append_status(
    path: Path,
    gpu_count: int,
    repetition: int,
    arm: Arm,
    status: str,
    exit_code: int,
    run_dir: Path,
) -> None:
    need_header = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if need_header:
            writer.writerow(
                ("timestamp_utc", "g", "repetition", "arm", "status", "exit_code", "run_dir")
            )
        writer.writerow(
            (utc_now(), gpu_count, repetition, arm.name, status, exit_code, run_dir)
        )


def enrich_row(
    row: Mapping[str, str], gpu_count: int, repetition: int, arm: Arm
) -> dict[str, str | int]:
    actual = int(row["per_peer_bytes"])
    return {
        "arm": arm.name,
        "repetition": repetition,
        "requested_ctas": "64" if arm.environment_overrides else "auto",
        "nominal_per_peer_bytes": actual + (4 if arm.odd_count else 0),
        **row,
        "g": gpu_count,
    }


def rebuild_raw(campaign: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for metadata_path in sorted((campaign / "runs").glob("g*/rep*/*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "pass":
            continue
        arm = ARM_BY_NAME[metadata["arm"]]
        data_path = metadata_path.parent / "shuffle.csv"
        with data_path.open(newline="") as stream:
            rows.extend(
                enrich_row(row, int(metadata["g"]), int(metadata["repetition"]), arm)
                for row in csv.DictReader(stream)
            )
    raw_path = campaign / "raw.csv"
    if rows:
        fieldnames = list(rows[0])
        with raw_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def write_summary(campaign: Path, rows: list[dict[str, str | int]]) -> None:
    metrics = (
        "comm_ms_median",
        "per_gpu_cross_gbps_median",
        "cross_gpu_gbps_median",
    )
    grouped: dict[tuple[int, str, int], list[dict[str, str | int]]] = {}
    for row in rows:
        key = (int(row["g"]), str(row["arm"]), int(row["nominal_per_peer_bytes"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (gpu_count, arm_name, nominal), group in sorted(grouped.items()):
        record: dict[str, object] = {
            "g": gpu_count,
            "arm": arm_name,
            "nominal_per_peer_bytes": nominal,
            "actual_per_peer_bytes": int(group[0]["per_peer_bytes"]),
            "repetitions": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            record[f"{metric}_mean_across_reps"] = statistics.fmean(values)
            record[f"{metric}_median_across_reps"] = statistics.median(values)
            record[f"{metric}_std_across_reps"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(record)
    if output:
        with (campaign / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)


def validate_matrix(
    campaign: Path, rows: list[dict[str, str | int]], repetitions: int
) -> dict[str, object]:
    point_sets: dict[str, list[int]] = {}
    for gpu_count in GPU_COUNTS:
        for repetition in range(1, repetitions + 1):
            for arm in ARMS:
                key = f"g{gpu_count}/rep{repetition:02d}/{arm.name}"
                point_sets[key] = sorted(
                    int(row["nominal_per_peer_bytes"])
                    for row in rows
                    if int(row["g"]) == gpu_count
                    and int(row["repetition"]) == repetition
                    and row["arm"] == arm.name
                )
    unique = {tuple(points) for points in point_sets.values()}
    expected_runs = len(GPU_COUNTS) * repetitions * len(ARMS)
    report = {
        "complete": len(rows) > 0
        and all(point_sets.values())
        and len(unique) == 1,
        "expected_runs": expected_runs,
        "completed_runs": sum(bool(points) for points in point_sets.values()),
        "common_nominal_points": list(next(iter(unique))) if len(unique) == 1 else [],
        "points_by_run": point_sets,
    }
    (campaign / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def stream_command(
    command: list[str], environment: Mapping[str, str], log_path: Path
) -> int:
    with log_path.open("w") as log:
        log.write("command: " + shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def campaign_configuration(args: argparse.Namespace, hosts: tuple[str, ...]) -> dict[str, object]:
    return {
        "gpu_counts": list(GPU_COUNTS),
        "hosts": list(hosts),
        "gpus_per_node": args.gpus_per_node,
        "ssh_port": args.ssh_port,
        "build_dir": str(args.build_dir),
        "benchmark": str(args.benchmark),
        "benchmark_sha256": file_sha256(args.benchmark),
        "mpirun": str(args.mpirun),
        "msg_exp_min": args.msg_exp_min,
        "msg_exp_max": args.msg_exp_max,
        "iters": args.iters,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "nccl_socket_ifname": args.nccl_socket_ifname,
        "nccl_mnnvl_enable": args.nccl_mnnvl_enable,
        "rmm_resource": args.rmm_resource,
        "controlled_keys": list(CONTROLLED_KEYS),
        "arms": [asdict(arm) for arm in ARMS],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--build-dir", type=Path, default=REPOSITORY / "build")
    parser.add_argument("--hosts", required=True)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--ssh-port", type=int, default=2222)
    parser.add_argument("--mpirun", type=Path, default=Path("/opt/hpcx/ompi/bin/orterun"))
    parser.add_argument("--msg-exp-min", type=int, default=16)
    parser.add_argument("--msg-exp-max", type=int, default=30)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--nccl-socket-ifname", default="eth1")
    parser.add_argument("--nccl-mnnvl-enable", default="1")
    parser.add_argument("--rmm-resource", default="arena")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.build_dir = args.build_dir.resolve()
    args.benchmark = args.build_dir / "src" / "cuda" / "alps" / "alps_shuffle_bench"
    args.campaign = args.campaign.resolve()
    if args.msg_exp_min > args.msg_exp_max:
        parser.error("--msg-exp-min must not exceed --msg-exp-max")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if not args.benchmark.is_file() or not os.access(args.benchmark, os.X_OK):
        parser.error(f"benchmark is not executable: {args.benchmark}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        hosts = parse_hosts(args.hosts, args.gpus_per_node)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    configuration = campaign_configuration(args, hosts)
    manifest_path = args.campaign / "manifest.json"
    if args.campaign.exists() and not args.resume:
        print(f"error: campaign exists; pass --resume: {args.campaign}", file=sys.stderr)
        return 2
    args.campaign.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())["configuration"]
        if saved != configuration:
            print("error: resume configuration differs from manifest", file=sys.stderr)
            return 2

    base_env = base_environment(args)
    try:
        host_reports = {
            host: preflight_host(host, args, base_env) for host in hosts[:6]
        }
    except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: preflight failed: {error}", file=sys.stderr)
        return 2
    manifest = {
        "created_utc": utc_now(),
        "coordinator": socket.gethostname(),
        "configuration": configuration,
        "host_preflight": host_reports,
        "git": {
            "revision": run_capture(["git", "rev-parse", "HEAD"]).stdout.strip(),
            "status": run_capture(["git", "status", "--short"]).stdout.splitlines(),
        },
    }
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    status_path = args.campaign / "status.csv"
    completed = read_status(status_path)
    failures = 0
    for gpu_count in GPU_COUNTS:
        for repetition in range(1, args.repetitions + 1):
            for arm in arm_order(repetition, gpu_count):
                key = (gpu_count, repetition, arm.name)
                if key in completed:
                    print(f"SKIP passed g={gpu_count} rep={repetition} arm={arm.name}")
                    continue
                run_dir = invocation_dir(args.campaign, gpu_count, repetition, arm)
                run_dir.mkdir(parents=True, exist_ok=True)
                output_csv = run_dir / "shuffle.csv"
                if output_csv.exists():
                    output_csv.unlink()
                command = build_command(
                    args, hosts, gpu_count, repetition, arm, output_csv
                )
                arm_env, effective = arm_environment(base_env, arm)
                metadata = {
                    "arm": arm.name,
                    "g": gpu_count,
                    "repetition": repetition,
                    "effective_cta_environment": effective,
                    "command": command,
                    "status": "dry_run" if args.dry_run else "running",
                    "started_utc": utc_now(),
                }
                metadata_path = run_dir / "metadata.json"
                metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
                print(f"RUN g={gpu_count} rep={repetition} arm={arm.name}")
                print("  " + shlex.join(command))
                if args.dry_run:
                    continue
                return_code = stream_command(command, arm_env, run_dir / "run.log")
                passed = return_code == 0 and output_csv.is_file() and output_csv.stat().st_size > 0
                metadata.update(
                    {
                        "status": "pass" if passed else "fail",
                        "exit_code": return_code,
                        "finished_utc": utc_now(),
                    }
                )
                metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
                append_status(
                    status_path,
                    gpu_count,
                    repetition,
                    arm,
                    "pass" if passed else "fail",
                    return_code,
                    run_dir,
                )
                if not passed:
                    failures += 1

    if args.dry_run:
        return 0
    rows = rebuild_raw(args.campaign)
    write_summary(args.campaign, rows)
    validation = validate_matrix(args.campaign, rows, args.repetitions)
    if not validation["complete"]:
        failures += 1
    print(
        f"Campaign complete: {validation['completed_runs']}/{validation['expected_runs']} runs; "
        f"artifacts: {args.campaign}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
