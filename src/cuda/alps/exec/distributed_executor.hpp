// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <optional>

#include "alps/collectives/alltoall.hpp"
#include "alps/context.hpp"
#include "alps/operators/column_overlap_eq_join.hpp"
#include "exec/executor.hpp"

namespace alps::exec {

enum class DistributedOperatorImplementation {
    Legacy,
    ColumnOverlap,
};
using DistributedEqJoinImplementation = DistributedOperatorImplementation;
using DistributedGroupByImplementation = DistributedOperatorImplementation;
using DistributedSemiJoinImplementation = DistributedOperatorImplementation;

struct DistributedExecOptions {
    alps::collectives::Backend backend {alps::collectives::Backend::NCCL};
    // Optional ColumnOverlap-only override. The default inherits `backend`, so
    // selecting CE_PULL globally covers both blocking collectives and
    // ColumnOverlap. An explicit value retains mixed-backend A/B experiments.
    std::optional<alps::collectives::Backend> column_exchange_backend;
    alps::collectives::CePullSchedule ce_pull_schedule {
        alps::collectives::CePullSchedule::RoundRobinWaves};
    int ce_pull_slots {2};
    alps::collectives::AlignmentPolicy shuffle_alignment {
        alps::collectives::AlignmentPolicy::Off};
    alps::collectives::AlignmentPolicy replication_alignment {
        alps::collectives::AlignmentPolicy::Off};
    DistributedEqJoinImplementation eq_join_implementation {
        DistributedEqJoinImplementation::Legacy};
    DistributedGroupByImplementation group_by_implementation {
        DistributedGroupByImplementation::Legacy};
    DistributedSemiJoinImplementation semi_join_implementation {
        DistributedSemiJoinImplementation::Legacy};
    alps::operators::ColumnSchedule column_overlap_schedule {
        alps::operators::ColumnSchedule::Interleave};
    bool column_overlap_high_priority {true};
    bool column_overlap_early_reclaim {false};
};

// Execute a physical ExecPlan containing both local GaMax steps and ALPS
// communication steps. Every rank must execute the same ordered plan.
std::unique_ptr<gamax::table> execute_distributed(
    const gamax::exec::ExecPlan& plan,
    const gamax::exec::SourceResolver& source_resolver,
    const gamax::phys::Context& physical_context,
    const alps::Context& distributed_context,
    const DistributedExecOptions& options = {});

// Execute only post-load steps against a reusable benchmark preload state.
// This is the distributed counterpart of gamax::exec::bench_run_steps().
std::unique_ptr<gamax::table> benchmark_run_distributed_steps(
    const gamax::exec::ExecPlan& plan,
    const gamax::exec::BenchExecState& state,
    const gamax::phys::Context& physical_context,
    const alps::Context& distributed_context,
    const DistributedExecOptions& options = {});

} // namespace alps::exec
