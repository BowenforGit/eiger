#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Restartable, strictly sequential TPC-H Parquet generation driver.

The driver launches one tpchgen-cli process for one table partition at a time.
It never launches concurrent generators and never overwrites an existing file.
It records an atomic state file after each completed output.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterator

REPLICATED_TABLES = ("region", "nation")
SHARDED_TABLES = (
    "supplier", "customer", "part", "partsupp", "orders", "lineitem")
ALL_TABLES = REPLICATED_TABLES + SHARDED_TABLES
STATE_NAME = ".generation_state.json"
LOCK_NAME = ".generation.lock"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tables(value: str) -> list[str]:
    tables = [piece.strip() for piece in value.split(",") if piece.strip()]
    unknown = sorted(set(tables) - set(ALL_TABLES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown tables: {unknown}")
    if not tables:
        raise argparse.ArgumentTypeError("table list must not be empty")
    return tables


def expected_outputs(
    staging: Path,
    tables: list[str],
    parts: int,
) -> Iterator[tuple[str, int | None, Path]]:
    for table in tables:
        if table in REPLICATED_TABLES:
            yield table, None, staging / f"{table}.parquet"
            continue
        for part in range(1, parts + 1):
            yield table, part, staging / table / f"{table}.{part}.parquet"


def command_for(
    args: argparse.Namespace,
    table: str,
    part: int | None,
) -> list[str]:
    command = [
        str(args.generator),
        "parquet",
        "--scale-factor", str(args.scale),
        "--tables", table,
        "--output-dir", str(args.staging_dir),
        "--num-threads", str(args.threads),
        "--compression", args.compression,
        "--row-group-bytes", str(args.row_group_bytes),
        "--no-progress",
    ]
    if part is not None:
        command.extend(["--parts", str(args.parts), "--part", str(part)])
    return command


def output_key(table: str, part: int | None) -> str:
    return f"{table}:single" if part is None else f"{table}:{part}"


def load_state(path: Path, configuration: dict) -> dict:
    if not path.exists():
        return {"configuration": configuration, "completed": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read generation state {path}: {error}") from error
    if state.get("configuration") != configuration:
        raise RuntimeError(
            f"{path} was created with different generation settings; "
            "use a different staging directory")
    if not isinstance(state.get("completed"), dict):
        raise RuntimeError(f"{path}: completed must be an object")
    return state


def validate_completed(path: Path, record: dict, key: str) -> bool:
    if not path.is_file():
        raise RuntimeError(
            f"state marks {key} complete but its file is missing: {path}")
    size = path.stat().st_size
    if size != record.get("bytes"):
        raise RuntimeError(
            f"state/file size mismatch for {key}: expected "
            f"{record.get('bytes')}, got {size}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--parts", type=int, default=144)
    parser.add_argument(
        "--tables", type=parse_tables,
        default=list(ALL_TABLES),
        help="comma-separated tables (default: all in safe generation order)")
    parser.add_argument(
        "--staging-dir", type=Path, required=True,
        help="unique staging directory; never the published sfN directory")
    parser.add_argument(
        "--published-dir", type=Path, default=None,
        help="published path to protect (default: /storage/tpch/sf<SF>)")
    parser.add_argument(
        "--generator", type=Path,
        default=Path("/home/scada/.local/bin/tpchgen-cli"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--compression", default="SNAPPY")
    parser.add_argument("--row-group-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument(
        "--stop-after", type=int, default=None,
        help="stop after N newly generated outputs (safe rehearsal aid)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print commands without creating the staging directory")
    args = parser.parse_args()

    if args.scale <= 0 or args.parts <= 0 or args.threads <= 0:
        parser.error("--scale, --parts, and --threads must be positive")
    if args.row_group_bytes <= 0:
        parser.error("--row-group-bytes must be positive")
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")
    if not args.generator.is_file() or not os.access(args.generator, os.X_OK):
        parser.error(f"generator is not executable: {args.generator}")
    generator_version = subprocess.run(
        [str(args.generator), "--version"],
        check=True, capture_output=True, text=True).stdout.strip()

    args.staging_dir = args.staging_dir.resolve()
    published = (
        args.published_dir or Path(f"/storage/tpch/sf{args.scale}")).resolve()
    if args.staging_dir == published:
        parser.error("staging directory must not be the published dataset path")
    if (args.staging_dir / "manifest.json").exists() or \
            (args.staging_dir / "_SUCCESS").exists():
        parser.error("staging directory is already published/completed")

    configuration = {
        "scale_factor": args.scale,
        "parts": args.parts,
        "tables": args.tables,
        "generator": str(args.generator.resolve()),
        "generator_version": generator_version,
        "threads": args.threads,
        "compression": args.compression,
        "row_group_bytes": args.row_group_bytes,
        "mode": "strictly_sequential_one_part_per_process",
    }

    if args.dry_run:
        for table, part, _ in expected_outputs(
                args.staging_dir, args.tables, args.parts):
            print(shlex.join(command_for(args, table, part)))
        return 0

    args.staging_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.staging_dir / LOCK_NAME
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"another generation driver holds {lock_path}") from None

        state_path = args.staging_dir / STATE_NAME
        state = load_state(state_path, configuration)
        generated = 0
        total = sum(
            1 for _ in expected_outputs(
                args.staging_dir, args.tables, args.parts))
        position = 0
        for table, part, output in expected_outputs(
                args.staging_dir, args.tables, args.parts):
            position += 1
            key = output_key(table, part)
            record = state["completed"].get(key)
            if record is not None:
                validate_completed(output, record, key)
                print(f"[{position}/{total}] already complete: {key}", flush=True)
                continue
            if output.exists():
                raise RuntimeError(
                    f"refusing to overwrite untracked output for {key}: {output}")

            command = command_for(args, table, part)
            print(
                f"[{position}/{total}] generating {key}: "
                f"{shlex.join(command)}",
                flush=True)
            subprocess.run(command, check=True)
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(
                    f"generator did not create a non-empty output for {key}: "
                    f"{output}")

            # Hash immediately while the just-written file is likely still in
            # the host page cache. Only one file is read and hashed at a time.
            record = {
                "path": str(output.relative_to(args.staging_dir)),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
            }
            state["completed"][key] = record
            atomic_json(state_path, state)
            generated += 1
            if args.stop_after is not None and generated >= args.stop_after:
                print(
                    f"stopped safely after {generated} new output(s); "
                    f"resume with the same command",
                    flush=True)
                return 0

    print(
        f"generation complete: {len(state['completed'])} outputs in "
        f"{args.staging_dir}")
    print("dataset is still staging; validate and publish a manifest next")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        print(
            f"error: generator exited with status {error.returncode}",
            file=sys.stderr)
        sys.exit(error.returncode or 1)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
