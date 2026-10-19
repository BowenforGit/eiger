// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <ostream>
#include <string>

#include "exec/exec_plan.hpp"

namespace gamax::exec {

// Render an ExecPlan as a SSA-style text listing. Each step becomes
// one line of the form:
//
//     v<N> = <StepKind>(<input-v-ids>, <payload-summary>)
//
// Plus the `required_sources` list up top. Useful for snapshot tests
// of the lowering pass and for inspecting pipelines before execution.
std::string format_exec_plan(const ExecPlan& plan);
void print_exec_plan(std::ostream& os, const ExecPlan& plan);

// Stable camel-case name for a StepKind ("EqJoin", "GroupBy", ...).
// Exposed for the phys-debug hook in the executor, which uses it both
// as a logging label and as the token matched by the
// `GAMAX_PHYS_DEBUG` env var (case-insensitive).
const char* step_kind_name(StepKind k);

// Stable, bounded semantic label for an individual physical step. Uses column
// names rather than only ordinals where possible so NVTX and benchmark rows are
// meaningful without cross-referencing source code.
std::string format_step_label(
    const ExecPlan& plan,
    gamax::size_type step_index);

} // namespace gamax::exec
