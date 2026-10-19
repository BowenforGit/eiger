# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Immutable attempt directories, manifests, and status journaling."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ArtifactError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def create_campaign(
    campaign_dir: Path,
    config: dict[str, Any],
    dry_run: bool,
    provenance: dict[str, Any] | None = None,
) -> str:
    """Create a campaign, refusing to adopt any pre-existing path."""

    fingerprint = canonical_fingerprint(config)
    if campaign_dir.exists():
        raise ArtifactError(
            f"campaign directory already exists: {campaign_dir}; "
            "use the resume command for an existing campaign"
        )
    if dry_run:
        return fingerprint
    campaign_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        campaign_dir / "manifest" / "campaign.json",
        {
            "created_utc": utc_now(),
            "fingerprint": fingerprint,
            "configuration": config,
            "provenance": provenance or {},
        },
    )
    return fingerprint


def load_campaign(campaign_dir: Path) -> dict[str, Any]:
    """Load an existing campaign manifest without accepting configuration overrides."""

    if not campaign_dir.is_dir():
        raise ArtifactError(f"campaign directory does not exist: {campaign_dir}")
    manifest_path = campaign_dir / "manifest" / "campaign.json"
    if not manifest_path.is_file():
        raise ArtifactError(f"campaign manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"cannot read campaign manifest: {error}") from error
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ArtifactError(f"invalid campaign manifest: {manifest_path}")
    if manifest.get("fingerprint") != canonical_fingerprint(configuration):
        raise ArtifactError("campaign manifest fingerprint is invalid")
    return manifest


def record_invocation(
    campaign_dir: Path,
    mode: str,
    argv: list[str],
    details: dict[str, Any] | None = None,
) -> None:
    """Journal each invocation without changing the immutable campaign manifest."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    record: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "mode": mode,
        "argv": argv,
    }
    if details is not None:
        record["details"] = details
    write_json(
        campaign_dir / "invocations" / f"{timestamp}-{os.getpid()}.json",
        record,
    )


def next_attempt_path(query_root: Path) -> Path:
    """Return the next immutable attempt path without creating it."""

    numbers: list[int] = []
    if query_root.exists():
        for path in query_root.glob("attempt-*"):
            try:
                numbers.append(int(path.name.removeprefix("attempt-")))
            except ValueError:
                continue
    return query_root / f"attempt-{max(numbers, default=0) + 1:04d}"


def next_attempt_dir(query_root: Path) -> Path:
    attempt = next_attempt_path(query_root)
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def successful_attempt(query_root: Path) -> Path | None:
    if not query_root.exists():
        return None
    for path in sorted(query_root.glob("attempt-*"), reverse=True):
        result_path = path / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        if result.get("status") != "pass":
            continue
        expected = result.get("expected_artifacts", [])
        if all(
            (path / relative).is_file() and (path / relative).stat().st_size > 0
            for relative in expected
        ):
            return path
    return None


def append_status(
    output_root: Path,
    arm: str,
    stage: str,
    query: int,
    status: str,
    exit_code: int,
    attempt: Path | None,
) -> None:
    path = output_root / "status.csv"
    new_file = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if new_file:
            writer.writerow(
                [
                    "timestamp_utc",
                    "arm",
                    "stage",
                    "query",
                    "status",
                    "exit_code",
                    "attempt",
                ]
            )
        writer.writerow(
            [utc_now(), arm, stage, f"Q{query}", status, exit_code, str(attempt or "")]
        )
