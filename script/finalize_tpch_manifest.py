#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Validate staged TPC-H Parquet outputs and publish their manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from generate_tpch_multifile import (
    ALL_TABLES,
    REPLICATED_TABLES,
    STATE_NAME,
    atomic_json,
    expected_outputs,
    output_key,
    sha256,
)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def schema_description(fields: list[dict]) -> tuple[list[dict], str]:
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return fields, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_publish_destination(
    staging: Path,
    requested: Path | None,
) -> Path | None:
    if requested is None:
        return None
    publish = requested.resolve()
    if os.path.lexists(publish):
        raise RuntimeError(f"refusing to overwrite publish path: {publish}")
    parent = publish.parent
    if not parent.is_dir():
        raise RuntimeError(f"publish parent is not a directory: {parent}")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise RuntimeError(f"publish parent is not writable: {parent}")
    if staging.stat().st_dev != parent.stat().st_dev:
        raise RuntimeError(
            "staging and publish parent are on different filesystems")
    return publish


def inspect_parquet_files(tool: Path, paths: list[Path]) -> dict[Path, dict]:
    result = subprocess.run(
        [str(tool), *(str(path) for path in paths)],
        check=True, capture_output=True, text=True)
    try:
        records = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"metadata tool returned invalid JSON: {error}") from error
    if not isinstance(records, list) or len(records) != len(paths):
        raise RuntimeError(
            "metadata tool output count does not match requested files")
    by_path: dict[Path, dict] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError("metadata tool returned a malformed record")
        path = Path(record["path"]).resolve()
        if path in by_path:
            raise RuntimeError(f"metadata tool returned duplicate path: {path}")
        if not isinstance(record.get("rows"), int) or \
                not isinstance(record.get("row_groups"), int) or \
                not isinstance(record.get("schema"), list):
            raise RuntimeError(
                f"metadata tool returned invalid metadata for {path}")
        by_path[path] = record
    expected = {path.resolve() for path in paths}
    if set(by_path) != expected:
        raise RuntimeError("metadata tool paths do not match requested files")
    return by_path


def inspect_file(
    staging: Path,
    path: Path,
    state_record: dict,
    parquet_metadata: dict,
    *,
    rehash: bool,
) -> tuple[dict, list[dict]]:
    if not path.is_file():
        raise RuntimeError(f"missing generated file: {path}")
    size = path.stat().st_size
    if size != state_record.get("bytes"):
        raise RuntimeError(
            f"state/file size mismatch for {path}: "
            f"{state_record.get('bytes')} != {size}")
    recorded_hash = state_record.get("sha256")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise RuntimeError(f"invalid generation-state SHA-256 for {path}")
    if rehash:
        actual_hash = sha256(path)
        if actual_hash != recorded_hash:
            raise RuntimeError(f"SHA-256 mismatch for {path}")

    entry = {
        "path": str(path.relative_to(staging)),
        "rows": parquet_metadata["rows"],
        "bytes": size,
        "row_groups": parquet_metadata["row_groups"],
        "sha256": recorded_hash,
    }
    return entry, parquet_metadata["schema"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", required=True, type=Path)
    parser.add_argument(
        "--metadata-tool", type=Path,
        default=(
            Path(__file__).resolve().parents[1] /
            "build/src/cuda/tpch/tpch_parquet_metadata"),
        help="repository-native Parquet metadata inspector")
    parser.add_argument(
        "--publish-dir", type=Path, default=None,
        help="atomically rename staging here after writing _SUCCESS")
    parser.add_argument(
        "--rehash", action="store_true",
        help="reread every file and verify its generation-time SHA-256")
    args = parser.parse_args()

    staging = args.staging_dir.resolve()
    metadata_tool = args.metadata_tool.resolve()
    if not metadata_tool.is_file() or not os.access(metadata_tool, os.X_OK):
        parser.error(f"metadata tool is not executable: {metadata_tool}")
    state_path = staging / STATE_NAME
    if not state_path.is_file():
        parser.error(f"missing generation state: {state_path}")
    publish = validate_publish_destination(staging, args.publish_dir)
    if (staging / "manifest.json").exists() or (staging / "_SUCCESS").exists():
        parser.error("staging directory already contains publication metadata")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    configuration = state.get("configuration")
    completed = state.get("completed")
    if not isinstance(configuration, dict) or not isinstance(completed, dict):
        raise RuntimeError("invalid generation state structure")
    if configuration.get("tables") != list(ALL_TABLES):
        raise RuntimeError(
            "publication requires the complete canonical eight-table set in "
            "the safe generation order")
    scale = configuration.get("scale_factor")
    parts = configuration.get("parts")
    if not isinstance(scale, int) or not isinstance(parts, int):
        raise RuntimeError("generation state has invalid scale/parts")

    expected = list(expected_outputs(staging, list(ALL_TABLES), parts))
    expected_keys = {output_key(table, part) for table, part, _ in expected}
    if set(completed) != expected_keys:
        raise RuntimeError(
            "generation state is incomplete or contains unexpected outputs: "
            f"missing={sorted(expected_keys - set(completed))}, "
            f"unexpected={sorted(set(completed) - expected_keys)}")

    table_files: dict[str, list[tuple[int | None, Path, dict]]] = {
        table: [] for table in ALL_TABLES}
    for table, part, path in expected:
        key = output_key(table, part)
        table_files[table].append((part, path, completed[key]))

    manifest_tables: dict[str, dict] = {}
    for table in ALL_TABLES:
        metadata_by_path = inspect_parquet_files(
            metadata_tool,
            [path for _, path, _ in table_files[table]])
        canonical_schema = None
        canonical_fields = None
        canonical_fingerprint = None
        entries: list[tuple[int | None, dict]] = []
        for part, path, record in table_files[table]:
            entry, schema = inspect_file(
                staging, path, record, metadata_by_path[path.resolve()],
                rehash=args.rehash)
            fields, fingerprint = schema_description(schema)
            if canonical_schema is None:
                canonical_schema = schema
                canonical_fields = fields
                canonical_fingerprint = fingerprint
            elif schema != canonical_schema:
                raise RuntimeError(
                    f"schema mismatch in table {table}: {path}")
            entries.append((part, entry))

        total_rows = sum(entry["rows"] for _, entry in entries)
        total_bytes = sum(entry["bytes"] for _, entry in entries)
        common = {
            "read_policy": (
                "replicated" if table in REPLICATED_TABLES else "sharded"),
            "schema": canonical_fields,
            "schema_fingerprint": canonical_fingerprint,
            "rows": total_rows,
            "bytes": total_bytes,
        }
        if table in REPLICATED_TABLES:
            if len(entries) != 1 or entries[0][0] is not None:
                raise RuntimeError(f"replicated table {table} is not single-file")
            manifest_tables[table] = {
                **common,
                "layout": "single",
                "file": entries[0][1],
            }
        else:
            part_entries = []
            for part, entry in entries:
                if part is None:
                    raise RuntimeError(f"sharded table {table} has a single file")
                part_entries.append({"id": part, **entry})
            manifest_tables[table] = {
                **common,
                "layout": "partitioned",
                "part_count": parts,
                "parts": part_entries,
            }

    declared = {
        (staging / entry["path"]).resolve()
        for table in manifest_tables.values()
        for entry in (
            [table["file"]] if table["layout"] == "single" else table["parts"])
    }
    actual = {path.resolve() for path in staging.rglob("*.parquet")}
    if actual != declared:
        raise RuntimeError(
            "staging Parquet files do not exactly match generated outputs: "
            f"missing={sorted(str(path) for path in declared - actual)}, "
            f"unexpected={sorted(str(path) for path in actual - declared)}")

    manifest = {
        "format_version": 1,
        "scale_factor": scale,
        "generator": {
            "path": configuration.get("generator"),
            "version": configuration.get("generator_version"),
            "mode": configuration.get("mode"),
        },
        "generation": {
            "parts": parts,
            "threads": configuration.get("threads"),
            "compression": configuration.get("compression"),
            "row_group_bytes": configuration.get("row_group_bytes"),
        },
        "tables": manifest_tables,
    }
    # Recheck immediately before committing publication metadata. This catches
    # a destination created while the potentially long metadata scan ran and
    # leaves staging retryable.
    publish = validate_publish_destination(staging, publish)
    manifest_path = staging / "manifest.json"
    atomic_json(manifest_path, manifest)
    manifest_hash = sha256(manifest_path)
    atomic_text(
        staging / "_SUCCESS",
        f"manifest_sha256={manifest_hash}\n")
    print(f"validated and completed staging dataset: {staging}")
    print(f"manifest SHA-256: {manifest_hash}")

    if publish is not None:
        os.rename(staging, publish)
        print(f"published atomically: {publish}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
