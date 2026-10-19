#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Strict resolver for legacy single-file and manifest TPC-H datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

TABLES = [
    "customer", "lineitem", "nation", "orders",
    "part", "partsupp", "region", "supplier",
]


class DatasetLayoutError(RuntimeError):
    pass


def _nonnegative_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetLayoutError(f"{context} must be a non-negative integer")
    return value


def _checksum(value: object, context: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise DatasetLayoutError(
            f"{context} must be 64 lowercase hexadecimal characters")
    return value


def _schema_fingerprint(value: object, context: str) -> str:
    if not isinstance(value, list):
        raise DatasetLayoutError(f"{context} must be an array")
    for index, field in enumerate(value, 1):
        field_context = f"{context} field {index}"
        if not isinstance(field, dict) or set(field) != {
                "name", "nullable", "type"}:
            raise DatasetLayoutError(
                f"{field_context} must contain exactly name, nullable, and type")
        if not isinstance(field["name"], str):
            raise DatasetLayoutError(f"{field_context} name must be a string")
        if not isinstance(field["nullable"], bool):
            raise DatasetLayoutError(
                f"{field_context} nullable must be a boolean")
        if not isinstance(field["type"], str):
            raise DatasetLayoutError(f"{field_context} type must be a string")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, relative: object, context: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DatasetLayoutError(f"{context}: path must be a non-empty string")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise DatasetLayoutError(f"{context}: path must be relative: {relative}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise DatasetLayoutError(
            f"{context}: path escapes dataset root: {relative}") from error
    return resolved


def _verify_file(path: Path, entry: dict, context: str) -> None:
    if not path.is_file():
        raise DatasetLayoutError(f"{context}: missing regular file: {path}")
    size = entry.get("bytes")
    size = _nonnegative_int(size, f"{context}: bytes")
    actual = path.stat().st_size
    if actual != size:
        raise DatasetLayoutError(
            f"{context}: size mismatch for {path}: expected {size}, got {actual}")


def load_manifest(sf_dir: str | Path, expected_sf: int) -> dict | None:
    root = Path(sf_dir)
    manifest_path = root / "manifest.json"
    success_path = root / "_SUCCESS"
    if not manifest_path.is_file():
        if success_path.exists():
            raise DatasetLayoutError(
                f"{root}: _SUCCESS exists without manifest.json")
        return None
    if not success_path.is_file():
        raise DatasetLayoutError(
            f"{root}: manifest dataset is incomplete; _SUCCESS is missing")

    manifest_hash = _sha256(manifest_path)
    expected_marker = f"manifest_sha256={manifest_hash}"
    actual_marker = success_path.read_text(encoding="utf-8").strip()
    if actual_marker != expected_marker:
        raise DatasetLayoutError(
            f"{success_path}: completion marker does not match manifest SHA-256")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetLayoutError(
            f"failed to read manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise DatasetLayoutError("manifest root must be an object")
    if manifest.get("format_version") != 1:
        raise DatasetLayoutError(
            f"unsupported manifest format_version: "
            f"{manifest.get('format_version')!r}")
    if manifest.get("scale_factor") != expected_sf:
        raise DatasetLayoutError(
            f"manifest scale_factor mismatch: expected {expected_sf}, "
            f"got {manifest.get('scale_factor')!r}")
    if not isinstance(manifest.get("tables"), dict):
        raise DatasetLayoutError("manifest tables must be an object")
    return manifest


def resolve_table_files(
    sf_dir: str | Path,
    table: str,
    expected_sf: int,
    *,
    allow_legacy_multifile: bool = False,
) -> list[str]:
    root = Path(sf_dir)
    manifest = load_manifest(root, expected_sf)
    single = root / f"{table}.parquet"
    directory = root / table

    if manifest is None:
        if directory.is_dir():
            if single.exists():
                raise DatasetLayoutError(
                    f"ambiguous table {table}: both file and directory exist")
            if not allow_legacy_multifile:
                raise DatasetLayoutError(
                    f"unmanifested multi-file table {table} is disabled")
            parts = sorted(
                path for path in directory.iterdir()
                if path.is_file() and path.suffix == ".parquet")
            if not parts:
                raise DatasetLayoutError(
                    f"legacy table directory has no Parquet files: {directory}")
            return [str(path) for path in parts]
        if not single.is_file():
            raise DatasetLayoutError(f"missing table Parquet file: {single}")
        return [str(single)]

    entry = manifest["tables"].get(table)
    if not isinstance(entry, dict):
        raise DatasetLayoutError(f"manifest has no table entry for {table}")
    layout = entry.get("layout")
    expected_policy = "replicated" if table in ("region", "nation") else "sharded"
    if entry.get("read_policy") != expected_policy:
        raise DatasetLayoutError(
            f"table {table}: read_policy must be {expected_policy}")
    schema = entry.get("schema")
    declared_fingerprint = _checksum(
        entry.get("schema_fingerprint"),
        f"table {table}: schema_fingerprint")
    if _schema_fingerprint(schema, f"table {table}: schema") != \
            declared_fingerprint:
        raise DatasetLayoutError(
            f"table {table}: schema_fingerprint does not match schema")
    table_rows = _nonnegative_int(entry.get("rows"), f"table {table}: rows")
    table_bytes = _nonnegative_int(entry.get("bytes"), f"table {table}: bytes")
    if layout == "single":
        if directory.exists():
            raise DatasetLayoutError(
                f"ambiguous table {table}: manifest declares single but "
                f"directory exists")
        file_entry = entry.get("file")
        if not isinstance(file_entry, dict):
            raise DatasetLayoutError(f"table {table}: file must be an object")
        path = _safe_path(root, file_entry.get("path"), f"table {table}")
        if path != single.resolve():
            raise DatasetLayoutError(
                f"table {table}: single layout must use {single.name}")
        file_rows = _nonnegative_int(
            file_entry.get("rows"), f"table {table}: file rows")
        _nonnegative_int(
            file_entry.get("row_groups"), f"table {table}: file row_groups")
        _checksum(file_entry.get("sha256"), f"table {table}: file sha256")
        _verify_file(path, file_entry, f"table {table}")
        if file_rows != table_rows or file_entry["bytes"] != table_bytes:
            raise DatasetLayoutError(
                f"table {table}: file metadata does not match table totals")
        return [str(path)]

    if layout != "partitioned":
        raise DatasetLayoutError(
            f"table {table}: unsupported layout {layout!r}")
    if single.exists():
        raise DatasetLayoutError(
            f"ambiguous table {table}: partitioned manifest and single file")
    if not directory.is_dir():
        raise DatasetLayoutError(f"missing table directory: {directory}")
    parts = entry.get("parts")
    part_count = entry.get("part_count")
    if (not isinstance(parts, list) or not parts or
            part_count != len(parts)):
        raise DatasetLayoutError(
            f"table {table}: part_count must match a non-empty parts list")

    paths: list[Path] = []
    seen: set[Path] = set()
    for index, part in enumerate(parts, 1):
        if not isinstance(part, dict):
            raise DatasetLayoutError(
                f"table {table} part {index}: entry must be an object")
        if part.get("id") != index:
            raise DatasetLayoutError(
                f"table {table}: part IDs must be ordered and contiguous from 1")
        path = _safe_path(root, part.get("path"), f"table {table} part {index}")
        if path.parent != directory.resolve():
            raise DatasetLayoutError(
                f"table {table} part {index}: path must be directly in "
                f"{directory}")
        if path in seen:
            raise DatasetLayoutError(
                f"table {table}: duplicate part path {path}")
        seen.add(path)
        _nonnegative_int(
            part.get("rows"), f"table {table} part {index}: rows")
        _nonnegative_int(
            part.get("row_groups"),
            f"table {table} part {index}: row_groups")
        _checksum(
            part.get("sha256"), f"table {table} part {index}: sha256")
        _verify_file(path, part, f"table {table} part {index}")
        paths.append(path)

    if sum(part["rows"] for part in parts) != table_rows:
        raise DatasetLayoutError(
            f"table {table}: part rows do not match table total")
    if sum(part["bytes"] for part in parts) != table_bytes:
        raise DatasetLayoutError(
            f"table {table}: part bytes do not match table total")

    actual = {
        path.resolve() for path in directory.iterdir()
        if path.is_file() and path.suffix == ".parquet"
    }
    if actual != seen:
        missing = sorted(str(path) for path in seen - actual)
        unexpected = sorted(str(path) for path in actual - seen)
        raise DatasetLayoutError(
            f"table {table}: directory does not exactly match manifest; "
            f"missing={missing}, unexpected={unexpected}")
    return [str(path) for path in paths]


def resolve_all_tables(
    sf_dir: str | Path,
    expected_sf: int,
    *,
    allow_legacy_multifile: bool = False,
) -> dict[str, list[str]]:
    return {
        table: resolve_table_files(
            sf_dir, table, expected_sf,
            allow_legacy_multifile=allow_legacy_multifile)
        for table in TABLES
    }
