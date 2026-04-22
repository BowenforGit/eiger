#include "ir/emit.hpp"

#include <stdexcept>

#include "exec/executor.hpp"
#include "ir/lower_to_exec_plan.hpp"

// Phase 2: `ir::emit` is now a two-line adapter — lower the IR to an
// ExecPlan, then execute that plan against the caller's sources and
// stream. All the per-kind emitter logic that used to live here is
// now split between `ir::lower_to_exec_plan` (IR→ExecPlan) and
// `exec::execute` (ExecPlan→table). Preserved only because many
// tests and TPC-H builders still call `ir::emit(root, ctx)`.

namespace gamax::ir {

std::unique_ptr<gamax::table> emit(const PlanPtr& root, const EmitContext& ctx) {
    if (!root) throw std::runtime_error("ir::emit: null root");
    if (!ctx.catalog) throw std::runtime_error("ir::emit: null catalog");
    if (!ctx.resolve_table) {
        throw std::runtime_error("ir::emit: null resolve_table");
    }
    auto plan = lower_to_exec_plan(root, *ctx.catalog);
    gamax::phys::Context phys_ctx{ctx.stream};
    return exec::execute(plan, ctx.resolve_table, phys_ctx);
}

} // namespace gamax::ir
