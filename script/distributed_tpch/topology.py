# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Topology parsing and direct-container preflight helpers."""

from __future__ import annotations

import re
import shlex
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path


_HOST_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class TopologyError(ValueError):
    pass


@dataclass(frozen=True)
class Host:
    name: str
    slots: int


@dataclass(frozen=True)
class Topology:
    hosts: tuple[Host, ...]
    local: bool
    ssh_port: int

    @property
    def ranks(self) -> int:
        return sum(host.slots for host in self.hosts)

    @property
    def gpus_per_node(self) -> int:
        return self.hosts[0].slots

    @property
    def host_spec(self) -> str:
        return ",".join(f"{host.name}:{host.slots}" for host in self.hosts)

    @property
    def label(self) -> str:
        kind = "local" if self.local else f"nodes{len(self.hosts)}"
        return f"{kind}_gpn{self.gpus_per_node}_np{self.ranks}"

    def launcher_args(self) -> list[str]:
        args = ["--ranks", str(self.ranks)]
        if not self.local:
            args.extend(
                [
                    "--hosts",
                    self.host_spec,
                    "--ranks-per-node",
                    str(self.gpus_per_node),
                    "--ssh-port",
                    str(self.ssh_port),
                ]
            )
        return args

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "local" if self.local else "explicit_hosts",
            "hosts": [{"name": host.name, "slots": host.slots} for host in self.hosts],
            "ranks": self.ranks,
            "gpus_per_node": self.gpus_per_node,
            "ssh_port": self.ssh_port,
            "label": self.label,
        }


def visible_gpu_count(device_probe: Path) -> int:
    try:
        result = subprocess.run(
            [str(device_probe), "--print-visible-device-count"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise TopologyError(f"CUDA visible-device probe failed: {error}") from error
    value = result.stdout.strip()
    if not value.isdigit() or int(value) < 1:
        raise TopologyError(f"CUDA visible-device probe returned {value!r}")
    return int(value)


def resolve_topology(
    hosts_spec: str | None,
    gpus_per_node: int | None,
    ssh_port: int,
    device_probe: Path,
) -> Topology:
    if hosts_spec is None:
        count = visible_gpu_count(device_probe)
        if gpus_per_node is not None and gpus_per_node != count:
            raise TopologyError(
                f"--gpus-per-node={gpus_per_node} does not match the {count} "
                "locally visible GPUs"
            )
        return Topology((Host(socket.gethostname(), count),), True, ssh_port)

    if not hosts_spec.strip():
        raise TopologyError("--hosts must not be empty")
    parsed: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for token in hosts_spec.split(","):
        token = token.strip()
        match = re.fullmatch(r"([^:]+)(?::([1-9][0-9]*))?", token)
        if not match or not _HOST_RE.fullmatch(match.group(1)):
            raise TopologyError(f"invalid host entry: {token!r}")
        name = match.group(1)
        if name in seen:
            raise TopologyError(f"duplicate host: {name}")
        seen.add(name)
        parsed.append((name, int(match.group(2)) if match.group(2) else None))

    declared = {slots for _, slots in parsed if slots is not None}
    if gpus_per_node is None:
        if any(slots is None for _, slots in parsed):
            raise TopologyError(
                "--gpus-per-node is required when --hosts entries omit slot counts"
            )
        if len(declared) != 1:
            raise TopologyError("selected hosts must use a uniform GPU count")
        gpus_per_node = declared.pop()
    elif any(slots is not None and slots != gpus_per_node for _, slots in parsed):
        raise TopologyError("host slot counts must match the requested --gpus-per-node")

    assert gpus_per_node is not None
    return Topology(
        tuple(Host(name, gpus_per_node) for name, _ in parsed), False, ssh_port
    )


def run_remote(
    host: str, ssh_port: int, argv: list[str]
) -> subprocess.CompletedProcess[str]:
    remote_command = shlex.join(argv)
    return subprocess.run(
        [
            "ssh",
            "-p",
            str(ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            host,
            remote_command,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def preflight_remote_hosts(
    topology: Topology,
    device_probe: Path,
    required_executables: tuple[Path, ...],
    data_root: Path,
    scale: int,
    queries: tuple[int, ...],
    table_names: tuple[str, ...],
    output_root: Path,
    require_nsys: bool,
    require_table_data: bool = True,
    require_ncu_rank0: bool = False,
) -> dict[str, dict[str, str]]:
    if topology.local:
        return {}
    results: dict[str, dict[str, str]] = {}
    golden_paths = [
        data_root / f"sf{scale}" / "golden" / f"q{q:02d}.parquet" for q in queries
    ]
    scale_dir = data_root / f"sf{scale}"
    for host in topology.hosts:
        checks = [scale_dir, *golden_paths]
        command = ["bash", "-c", ""]
        predicates = [f"test -e {shlex.quote(str(path))}" for path in checks]
        if require_table_data:
            predicates.extend(
                f"(test -f {shlex.quote(str(scale_dir / f'{table}.parquet'))} || "
                f"test -d {shlex.quote(str(scale_dir / table))})"
                for table in table_names
            )
        predicates.extend(
            [
                *(f"test -x {shlex.quote(str(path))}" for path in required_executables),
                f"test -f {shlex.quote(str(output_root / 'manifest' / 'campaign.json'))}",
            ]
        )
        if require_nsys:
            predicates.append("command -v nsys >/dev/null")
        if require_ncu_rank0 and host == topology.hosts[0]:
            predicates.append("command -v ncu >/dev/null")
        command[2] = " && ".join(predicates)
        try:
            run_remote(host.name, topology.ssh_port, command)
            count_result = run_remote(
                host.name,
                topology.ssh_port,
                [str(device_probe), "--print-visible-device-count"],
            )
            count = count_result.stdout.strip()
            if count != str(host.slots):
                raise TopologyError(
                    f"{host.name} exposes {count} GPUs, expected {host.slots}"
                )
            hash_result = run_remote(
                host.name,
                topology.ssh_port,
                ["sha256sum", *(str(path) for path in required_executables)],
            )
            process_result = run_remote(
                host.name,
                topology.ssh_port,
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name",
                    "--format=csv,noheader",
                ],
            )
            active_processes = [
                line.strip()
                for line in process_result.stdout.splitlines()
                if line.strip()
            ]
            if active_processes:
                raise TopologyError(
                    f"{host.name} has active GPU processes: {'; '.join(active_processes)}"
                )
            gpu_result = run_remote(
                host.name,
                topology.ssh_port,
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
            )
            uname_result = run_remote(host.name, topology.ssh_port, ["uname", "-a"])
            nsys_version = "not requested"
            if require_nsys:
                nsys_version = run_remote(
                    host.name, topology.ssh_port, ["nsys", "--version"]
                ).stdout.strip()
            ncu_version = "not requested"
            if require_ncu_rank0 and host == topology.hosts[0]:
                ncu_version = run_remote(
                    host.name, topology.ssh_port, ["ncu", "--version"]
                ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise TopologyError(
                f"remote preflight failed on {host.name}: {error}"
            ) from error
        results[host.name] = {
            "visible_gpus": count,
            "required_executable_sha256": hash_result.stdout.strip(),
            "gpu_inventory": gpu_result.stdout.strip(),
            "uname": uname_result.stdout.strip(),
            "nsys_version": nsys_version,
            "ncu_version": ncu_version,
        }
    return results
