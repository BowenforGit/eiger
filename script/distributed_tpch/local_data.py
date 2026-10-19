#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Prepare host-local TPC-H Parquet files for distributed campaigns."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable


REPLICATED_TABLES = ("region", "nation")
SHARDED_TABLES = (
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
)
ALL_TABLES = REPLICATED_TABLES + SHARDED_TABLES
ZERO_SHA256 = "0" * 64
ROW_GROUP_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class GenerationTask:
    table: str
    part: int | None
    output: Path


def cache_data_root(cache_root: Path, partitions: int) -> Path:
    return cache_root / f"parts{partitions}"


def cache_scale_dir(cache_root: Path, scale: int, partitions: int) -> Path:
    return cache_data_root(cache_root, partitions) / f"sf{scale}"


def assigned_part_indices(
    part_count: int, rank: int, world_size: int, strategy: str
) -> tuple[int, ...]:
    if part_count <= 0 or world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid TPC-H partition assignment")
    if strategy == "round_robin":
        return tuple(range(rank, part_count, world_size))
    if strategy != "contiguous_range":
        raise ValueError(f"unknown partition strategy: {strategy}")
    quotient, remainder = divmod(part_count, world_size)
    begin = rank * quotient + (rank * remainder) // world_size
    end = (rank + 1) * quotient + ((rank + 1) * remainder) // world_size
    return tuple(range(begin, end))


def required_tasks(
    cache_root: Path,
    scale: int,
    partitions: int,
    local_ranks: Iterable[int],
    world_size: int,
    strategy: str,
) -> tuple[GenerationTask, ...]:
    scale_dir = cache_scale_dir(cache_root, scale, partitions)
    tasks: list[GenerationTask] = []
    if partitions == 1:
        for table in ALL_TABLES:
            tasks.append(GenerationTask(table, None, scale_dir / f"{table}.parquet"))
        return tuple(tasks)

    for table in REPLICATED_TABLES:
        tasks.append(GenerationTask(table, None, scale_dir / f"{table}.parquet"))
    indices = sorted(
        {
            index
            for rank in local_ranks
            for index in assigned_part_indices(partitions, rank, world_size, strategy)
        }
    )
    for table in SHARDED_TABLES:
        for index in indices:
            part = index + 1
            tasks.append(
                GenerationTask(
                    table,
                    part,
                    scale_dir / table / f"{table}.{part}.parquet",
                )
            )
    return tuple(tasks)


def _generated_path(staging: Path, task: GenerationTask) -> Path:
    if task.part is None:
        return staging / f"{task.table}.parquet"
    return staging / task.table / f"{task.table}.{task.part}.parquet"


def generate_task(
    task: GenerationTask,
    *,
    generator: Path,
    scale: int,
    partitions: int,
    threads: int,
) -> bool:
    task.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = task.output.with_name(f".{task.output.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if task.output.is_file():
            return False
        staging_parent = (
            task.output.parents[1] if task.part is not None else task.output.parent
        )
        temporary = Path(tempfile.mkdtemp(prefix=f".{task.table}-", dir=staging_parent))
        try:
            command = [
                str(generator),
                "parquet",
                "--scale-factor",
                str(scale),
                "--tables",
                task.table,
                "--output-dir",
                str(temporary),
                "--num-threads",
                str(threads),
                "--compression",
                "SNAPPY",
                "--row-group-bytes",
                str(ROW_GROUP_BYTES),
                "--no-progress",
            ]
            if task.part is not None:
                command.extend(["--parts", str(partitions), "--part", str(task.part)])
            subprocess.run(command, check=True)
            generated = _generated_path(temporary, task)
            if not generated.is_file() or generated.stat().st_size == 0:
                raise RuntimeError(f"generator did not create {generated}")
            os.replace(generated, task.output)
            return True
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def _metadata_records(
    metadata_tool: Path, tasks: tuple[GenerationTask, ...]
) -> list[dict]:
    result = subprocess.run(
        [str(metadata_tool), *(str(task.output) for task in tasks)],
        check=True,
        capture_output=True,
        text=True,
    )
    records = json.loads(result.stdout)
    if not isinstance(records, list) or len(records) != len(tasks):
        raise RuntimeError("Parquet metadata output does not match prepared files")
    by_path = {Path(record["path"]).resolve(): record for record in records}
    prepared: list[dict] = []
    scale_dir = (
        tasks[0].output.parents[1]
        if tasks[0].part is not None
        else tasks[0].output.parent
    )
    for task in tasks:
        record = by_path.get(task.output.resolve())
        if record is None:
            raise RuntimeError(f"missing metadata for {task.output}")
        prepared.append(
            {
                "table": task.table,
                "part": task.part,
                "path": str(task.output.relative_to(scale_dir)),
                "bytes": task.output.stat().st_size,
                "rows": record["rows"],
                "row_groups": record["row_groups"],
                "schema": record["schema"],
            }
        )
    return prepared


def prepare_host(args: argparse.Namespace) -> int:
    local_ranks = tuple(int(value) for value in args.local_ranks.split(",") if value)
    tasks = required_tasks(
        args.cache_root,
        args.scale,
        args.partitions,
        local_ranks,
        args.world_size,
        args.strategy,
    )
    if not tasks:
        raise RuntimeError("host has no TPC-H preparation tasks")
    available_cpus = max(1, len(os.sched_getaffinity(0)))
    jobs = min(len(tasks), max(1, available_cpus // 16))
    threads = max(1, available_cpus // jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                generate_task,
                task,
                generator=args.generator,
                scale=args.scale,
                partitions=args.partitions,
                threads=threads,
            )
            for task in tasks
        ]
        for future in futures:
            future.result()
    print(
        json.dumps(_metadata_records(args.metadata_tool, tasks), separators=(",", ":"))
    )
    return 0


def schema_description(fields: list[dict]) -> tuple[list[dict], str]:
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return fields, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_synthetic_manifest(
    records: Iterable[dict],
    *,
    scale: int,
    partitions: int,
    generator: str,
) -> dict:
    unique: dict[tuple[str, int | None], dict] = {}
    for record in records:
        key = (record["table"], record["part"])
        previous = unique.get(key)
        if previous is not None:
            comparable = ("path", "bytes", "rows", "row_groups", "schema")
            if any(previous[name] != record[name] for name in comparable):
                raise RuntimeError(f"inconsistent generated metadata for {key}")
        else:
            unique[key] = record

    tables: dict[str, dict] = {}
    for table in ALL_TABLES:
        expected_parts: tuple[int | None, ...] = (
            (None,)
            if partitions == 1 or table in REPLICATED_TABLES
            else tuple(range(1, partitions + 1))
        )
        entries = []
        for part in expected_parts:
            record = unique.get((table, part))
            if record is None:
                raise RuntimeError(f"missing generated metadata for {table}:{part}")
            entries.append(record)
        schema, fingerprint = schema_description(entries[0]["schema"])
        if any(entry["schema"] != schema for entry in entries[1:]):
            raise RuntimeError(f"generated schema mismatch for table {table}")
        common = {
            "read_policy": "replicated" if table in REPLICATED_TABLES else "sharded",
            "schema": schema,
            "schema_fingerprint": fingerprint,
            "rows": sum(entry["rows"] for entry in entries),
            "bytes": sum(entry["bytes"] for entry in entries),
        }
        manifest_entries = [
            {
                "path": entry["path"],
                "rows": entry["rows"],
                "bytes": entry["bytes"],
                "row_groups": entry["row_groups"],
                "sha256": ZERO_SHA256,
            }
            for entry in entries
        ]
        if len(expected_parts) == 1:
            tables[table] = {**common, "layout": "single", "file": manifest_entries[0]}
        else:
            tables[table] = {
                **common,
                "layout": "partitioned",
                "part_count": partitions,
                "parts": [
                    {"id": part, **entry}
                    for part, entry in zip(expected_parts, manifest_entries)
                ],
            }
    return {
        "format_version": 1,
        "scale_factor": scale,
        "generator": {"path": generator, "mode": "host-local-on-demand"},
        "generation": {
            "parts": partitions,
            "compression": "SNAPPY",
            "row_group_bytes": ROW_GROUP_BYTES,
        },
        "tables": tables,
    }


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def write_manifest_bundle(directory: Path, manifest: dict) -> tuple[Path, Path]:
    manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = directory / "manifest.json"
    success_path = directory / "_SUCCESS"
    atomic_write(manifest_path, manifest_data)
    digest = hashlib.sha256(manifest_data).hexdigest()
    atomic_write(success_path, f"manifest_sha256={digest}\n".encode())
    return manifest_path, success_path


def install_manifest(args: argparse.Namespace) -> int:
    scale_dir = cache_scale_dir(args.cache_root, args.scale, args.partitions)
    atomic_write(scale_dir / "manifest.json", args.manifest.read_bytes())
    atomic_write(scale_dir / "_SUCCESS", args.success.read_bytes())
    golden = scale_dir / "golden"
    if golden.is_symlink():
        if golden.resolve() != args.golden_source.resolve():
            golden.unlink()
    elif golden.exists():
        raise RuntimeError(f"cannot replace non-symlink golden path: {golden}")
    if not golden.exists():
        golden.symlink_to(args.golden_source, target_is_directory=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--cache-root", type=Path, required=True)
    prepare.add_argument("--scale", type=int, required=True)
    prepare.add_argument("--partitions", type=int, required=True)
    prepare.add_argument("--local-ranks", required=True)
    prepare.add_argument("--world-size", type=int, required=True)
    prepare.add_argument(
        "--strategy", choices=("round_robin", "contiguous_range"), required=True
    )
    prepare.add_argument("--generator", type=Path, required=True)
    prepare.add_argument("--metadata-tool", type=Path, required=True)

    install = commands.add_parser("install-manifest")
    install.add_argument("--cache-root", type=Path, required=True)
    install.add_argument("--scale", type=int, required=True)
    install.add_argument("--partitions", type=int, required=True)
    install.add_argument("--manifest", type=Path, required=True)
    install.add_argument("--success", type=Path, required=True)
    install.add_argument("--golden-source", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scale <= 0 or args.partitions <= 0:
        raise RuntimeError("scale and partition count must be positive")
    if args.command == "prepare":
        if args.world_size <= 0:
            raise RuntimeError("world size must be positive")
        return prepare_host(args)
    return install_manifest(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
