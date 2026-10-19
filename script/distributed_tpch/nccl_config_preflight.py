#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Inspect NCCL configuration sources that can interfere with CTA arms."""

from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Iterable, Mapping


CONTROLLED_KEYS = (
    "NCCL_MIN_CTAS",
    "NCCL_MAX_CTAS",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_MIN_NRINGS",
    "NCCL_MAX_NRINGS",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_controlled_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in CONTROLLED_KEYS:
            entries[key] = value.strip()
    return entries


def configuration_paths(
    environment: Mapping[str, str], home: Path | None = None
) -> tuple[Path, ...]:
    paths = [Path("/etc/nccl.conf")]
    configured = environment.get("NCCL_CONF_FILE")
    paths.append(
        Path(configured) if configured else (home or Path.home()) / ".nccl.conf"
    )
    unique: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if expanded not in unique:
            unique.append(expanded)
    return tuple(unique)


def default_environment_plugin(
    environment: Mapping[str, str],
    library_directories: Iterable[Path] = (),
) -> str | None:
    directories = [*library_directories]
    directories.extend(
        Path(directory)
        for directory in environment.get("LD_LIBRARY_PATH", "").split(":")
        if directory
    )
    for directory in directories:
        matches = sorted(directory.glob("libnccl-env.so*"))
        if matches:
            return str(matches[0])
    resolved = ctypes.util.find_library("nccl-env")
    if resolved:
        return resolved
    return None


def inspect_configuration(
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
    library_directories: Iterable[Path] = (),
) -> dict[str, object]:
    values = os.environ if environment is None else environment
    files: list[dict[str, object]] = []
    conflicts: list[str] = []
    for path in configuration_paths(values, home):
        record: dict[str, object] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            try:
                text = path.read_text()
                entries = parse_controlled_entries(text)
                record.update(
                    sha256=file_sha256(path),
                    controlled_entries=entries,
                )
                if entries:
                    conflicts.append(f"{path} defines " + ", ".join(sorted(entries)))
            except OSError as error:
                record["error"] = str(error)
                conflicts.append(f"cannot inspect {path}: {error}")
        files.append(record)

    plugin_selection = values.get("NCCL_ENV_PLUGIN", "<unset>")
    resolved_plugin = None
    if plugin_selection == "<unset>":
        resolved_plugin = default_environment_plugin(values, library_directories)
        if resolved_plugin:
            conflicts.append(
                "default NCCL environment plugin is available: " + resolved_plugin
            )
    elif plugin_selection.lower() != "none":
        resolved_plugin = plugin_selection
        conflicts.append(
            "NCCL_ENV_PLUGIN may inject uninspectable CTA controls: " + plugin_selection
        )

    return {
        "host": socket.gethostname(),
        "observed_environment": {
            key: values[key] for key in CONTROLLED_KEYS if key in values
        },
        "configuration_files": files,
        "environment_plugin": {
            "selection": plugin_selection,
            "resolved": resolved_plugin,
        },
        "conflicts": conflicts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit one JSON document")
    parser.add_argument(
        "--library-dir",
        action="append",
        default=[],
        type=Path,
        help="additional rank library directory to inspect",
    )
    args = parser.parse_args(argv)
    result = inspect_configuration(library_directories=args.library_dir)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
