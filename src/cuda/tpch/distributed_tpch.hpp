// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <string>

#include <mpi.h>

#include "alps/exec/distributed_planner.hpp"
#include "exec/exec_plan.hpp"
#include "tpch/source_policy.hpp"

namespace gamax::tpch::dist {

alps::exec::DistributedPlanOptions plan_options(
    int query_id,
    SourceReadPoliciesPtr source_read_policies);

// Build the local physical plan before any distributed movement steps are
// inserted. This is the stable boundary at which per-operator physical-plan
// configuration is applied.
gamax::exec::ExecPlan build_local_exec_plan(
    int query_id,
    std::string* local_join_signatures = nullptr);

gamax::exec::ExecPlan distribute_exec_plan(
    int query_id,
    SourceReadPoliciesPtr source_read_policies,
    gamax::exec::ExecPlan local_plan);

// Build the same Spark-guided distributed physical plan used by correctness
// and benchmark runners. When requested, local_join_signatures receives a
// human-readable listing useful for diagnosing stale strategy overrides.
gamax::exec::ExecPlan build_exec_plan(
    int query_id,
    SourceReadPoliciesPtr source_read_policies,
    std::string* local_join_signatures = nullptr);

void preflight_dataset(int scale_factor, MPI_Comm communicator);

} // namespace gamax::tpch::dist
