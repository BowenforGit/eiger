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

} // namespace gamax::exec
