// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "exec/exec_plan.hpp"

namespace alps::exec {

struct JoinStrategyOverride {
    gamax::exec::StepKind kind {gamax::exec::StepKind::EqJoin};
    std::vector<std::string> left_key_names;
    std::vector<std::string> right_key_names;
    gamax::exec::JoinDistributionStrategy strategy {
        gamax::exec::JoinDistributionStrategy::Shuffle};
};

struct DistributedPlanOptions {
    int root {0};
    bool gather_final_result {true};
    gamax::exec::JoinDistributionStrategy default_join_strategy {
        gamax::exec::JoinDistributionStrategy::Shuffle};
    // Semantic signatures are stable across unrelated lowering changes. Every
    // configured override must match exactly one local join.
    std::vector<JoinStrategyOverride> join_strategy_overrides;
    gamax::exec::JoinDistributionStrategy cross_join_strategy {
        gamax::exec::JoinDistributionStrategy::BroadcastRight};
    // Empty means every source is a disjoint local shard. A configured
    // resolver may return LocalShard or Replicated for each table.
    std::function<gamax::exec::DistributionKind(const std::string&)>
        source_distribution;
};

// Convert a locally lowered physical plan into a distributed physical plan.
// Logical IR is not modified. Unconfigured sources resolve to disjoint local
// shards; operator kinds are replaced only where global semantics require it.
gamax::exec::ExecPlan plan_for_distributed_execution(
    gamax::exec::ExecPlan local_plan,
    const DistributedPlanOptions& options = {});

} // namespace alps::exec
