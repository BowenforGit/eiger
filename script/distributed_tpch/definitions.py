# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Canonical experiment-arm and collection-stage definitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class ExecutorKind(str, Enum):
    SYNTHESIS = "synthesis"
    BENCHMARK = "benchmark"
    NSYS = "nsys"
    NCU = "ncu"


class PlanSource(str, Enum):
    IMPLICIT_STOCK = "implicit_stock"
    SYNTHESIZED = "synthesized"


@dataclass(frozen=True)
class ArmDefinition:
    name: str
    description: str
    benchmark_args: tuple[str, ...]
    plan_source: PlanSource = PlanSource.IMPLICIT_STOCK
    variant_of: str | None = None
    environment_overrides: tuple[tuple[str, str], ...] = ()

    @property
    def uses_ce_pull(self) -> bool:
        return any(
            self.benchmark_args[index : index + 2] in (
                ("--communication-backend", "ce-pull"),
                ("--column-exchange-backend", "ce-pull"),
            )
            for index in range(len(self.benchmark_args) - 1)
        )


@dataclass(frozen=True)
class StageDefinition:
    name: str
    description: str
    executor: ExecutorKind
    benchmark_args: tuple[str, ...] = ()


LEGACY_ARGS = (
    "--eq-join",
    "legacy",
    "--group-by",
    "legacy",
    "--semi-join",
    "legacy",
    "--shuffle-alignment",
    "off",
    "--replication-alignment",
    "off",
)


BASE_ARMS: dict[str, ArmDefinition] = {
    "legacy": ArmDefinition(
        "legacy",
        "Legacy operators with the implicit stock physical plan",
        LEGACY_ARGS,
    ),
    "legacy_ce_pull": ArmDefinition(
        "legacy_ce_pull",
        "Legacy operators with CE-pull for every bulk communication path",
        LEGACY_ARGS
        + (
            "--communication-backend",
            "ce-pull",
            "--ce-pull-schedule",
            "round-robin-waves",
            "--ce-pull-slots",
            "2",
        ),
    ),
    "legacy_aligned_both": ArmDefinition(
        "legacy_aligned_both",
        "Legacy operators with shuffle and replication alignment enabled",
        (
            "--eq-join",
            "legacy",
            "--group-by",
            "legacy",
            "--semi-join",
            "legacy",
            "--shuffle-alignment",
            "on",
            "--replication-alignment",
            "on",
        ),
    ),
    "all_co_nccl": ArmDefinition(
        "all_co_nccl",
        "All ColumnOverlap operators with NCCL column exchange",
        (
            "--eq-join",
            "column-overlap",
            "--group-by",
            "column-overlap",
            "--semi-join",
            "column-overlap",
            "--column-exchange-backend",
            "nccl",
            "--column-schedule",
            "interleave",
            "--column-comm-priority",
            "high",
            "--shuffle-alignment",
            "off",
            "--replication-alignment",
            "off",
        ),
    ),
    "all_co_ce_pull": ArmDefinition(
        "all_co_ce_pull",
        "All ColumnOverlap operators inheriting global CE-pull for every bulk communication path",
        (
            "--eq-join",
            "column-overlap",
            "--group-by",
            "column-overlap",
            "--semi-join",
            "column-overlap",
            "--communication-backend",
            "ce-pull",
            "--ce-pull-schedule",
            "round-robin-waves",
            "--ce-pull-slots",
            "2",
            "--column-schedule",
            "interleave",
            "--column-comm-priority",
            "high",
            "--shuffle-alignment",
            "off",
            "--replication-alignment",
            "off",
        ),
    ),
    "group_co_nccl": ArmDefinition(
        "group_co_nccl",
        "ColumnOverlap GroupBy with NCCL; other operators remain Legacy",
        (
            "--eq-join",
            "legacy",
            "--group-by",
            "column-overlap",
            "--semi-join",
            "legacy",
            "--column-exchange-backend",
            "nccl",
            "--column-comm-priority",
            "high",
            "--shuffle-alignment",
            "off",
            "--replication-alignment",
            "off",
        ),
    ),
    "group_co_ce_pull": ArmDefinition(
        "group_co_ce_pull",
        "ColumnOverlap GroupBy with CE-pull; other operators remain Legacy",
        (
            "--eq-join",
            "legacy",
            "--group-by",
            "column-overlap",
            "--semi-join",
            "legacy",
            "--column-exchange-backend",
            "ce-pull",
            "--ce-pull-schedule",
            "round-robin-waves",
            "--ce-pull-slots",
            "2",
            "--column-comm-priority",
            "high",
            "--shuffle-alignment",
            "off",
            "--replication-alignment",
            "off",
        ),
    ),
    "semi_co_nccl": ArmDefinition(
        "semi_co_nccl",
        "ColumnOverlap SemiJoin with NCCL; other operators remain Legacy",
        (
            "--eq-join",
            "legacy",
            "--group-by",
            "legacy",
            "--semi-join",
            "column-overlap",
            "--column-exchange-backend",
            "nccl",
            "--column-comm-priority",
            "high",
            "--shuffle-alignment",
            "off",
            "--replication-alignment",
            "off",
        ),
    ),
    "semi_co_ce_pull": ArmDefinition(
        "semi_co_ce_pull",
        "ColumnOverlap SemiJoin with CE-pull; other operators remain Legacy",
        (
            "--eq-join",
            "legacy",
            "--group-by",
            "legacy",
            "--semi-join",
            "column-overlap",
            "--column-exchange-backend",
            "ce-pull",
            "--ce-pull-schedule",
            "round-robin-waves",
            "--ce-pull-slots",
            "2",
            "--column-comm-priority",
            "high",
            "--shuffle-alignment",
            "off",
            "--replication-alignment",
            "off",
        ),
    ),
    "synthesized": ArmDefinition(
        "synthesized",
        "Guarded per-query synthesized physical plans with Legacy fallbacks",
        LEGACY_ARGS,
        PlanSource.SYNTHESIZED,
    ),
}


NCCL_CTA_CHOICES = (8, 16, 32, 64)
NCCL_CTA_VARIANT_BASES = (
    "legacy",
    "legacy_aligned_both",
    "all_co_nccl",
    "synthesized",
)


ARMS: dict[str, ArmDefinition] = dict(BASE_ARMS)
for base_name in NCCL_CTA_VARIANT_BASES:
    base = BASE_ARMS[base_name]
    for ctas in NCCL_CTA_CHOICES:
        name = f"{base_name}_cta{ctas}"
        ARMS[name] = replace(
            base,
            name=name,
            description=f"{base.description}; fixed NCCL CTAs={ctas}",
            variant_of=base_name,
            environment_overrides=(
                ("NCCL_MIN_CTAS", str(ctas)),
                ("NCCL_MAX_CTAS", str(ctas)),
            ),
        )


STAGES: dict[str, StageDefinition] = {
    "synthesis": StageDefinition(
        "synthesis",
        "Generate guarded per-query physical plans",
        ExecutorKind.SYNTHESIS,
    ),
    "timing": StageDefinition(
        "timing",
        "Ten clean samples; shares load/warmup with breakdown when both are selected",
        ExecutorKind.BENCHMARK,
        (
            "--timing-only",
            "--iterations",
            "10",
            "--warmup-min",
            "3",
            "--warmup-window",
            "5",
            "--warmup-threshold",
            "0.05",
            "--warmup-max",
            "20",
        ),
    ),
    "breakdown": StageDefinition(
        "breakdown",
        "Five instrumented step/phase samples; combines with timing when selected",
        ExecutorKind.BENCHMARK,
        (
            "--steps-only",
            "--step-iterations",
            "5",
            "--warmup-min",
            "3",
            "--warmup-window",
            "5",
            "--warmup-threshold",
            "0.05",
            "--warmup-max",
            "20",
        ),
    ),
    "memory": StageDefinition(
        "memory",
        "One independent verified RMM peak-memory sample",
        ExecutorKind.BENCHMARK,
        ("--memory-profile",),
    ),
    "messages": StageDefinition(
        "messages",
        "One verified trace of exact NCCL API payload sizes (Legacy only)",
        ExecutorKind.BENCHMARK,
        ("--messages-only", "--message-iterations", "1"),
    ),
    "profiles": StageDefinition(
        "profiles",
        "One warmed Nsight Systems capture set per query",
        ExecutorKind.NSYS,
        (
            "--warmup-min",
            "3",
            "--warmup-window",
            "5",
            "--warmup-threshold",
            "0.05",
            "--warmup-max",
            "20",
        ),
    ),
    "ncu_profiles": StageDefinition(
        "ncu_profiles",
        "One warmed rank-0 Nsight Compute capture per query, excluding NCCL kernels",
        ExecutorKind.NCU,
        (
            "--warmup-min",
            "3",
            "--warmup-window",
            "5",
            "--warmup-threshold",
            "0.05",
            "--warmup-max",
            "20",
        ),
    ),
}


DEFAULT_PARQUET_DECODE_THREADS = 8


COMMON_BENCHMARK_ARGS = (
    "--verify",
    "--file-shard-strategy",
    "round_robin",
)


COMBINED_TIMING_BREAKDOWN_ARGS = (
    "--iterations",
    "10",
    "--step-iterations",
    "5",
    "--warmup-min",
    "3",
    "--warmup-window",
    "5",
    "--warmup-threshold",
    "0.05",
    "--warmup-max",
    "20",
)
