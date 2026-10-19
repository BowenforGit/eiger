// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/exec/distributed_executor.hpp"

#include <stdexcept>
#include <utility>

#include "alps/collectives/table_exchange.hpp"
#include "alps/operators/column_overlap_eq_join.hpp"
#include "alps/operators/distributed_relational.hpp"
#include "exec/step_profile.hpp"

namespace alps::exec {
namespace {

using gamax::exec::ExtensionInput;
using gamax::exec::ExtensionResult;
using gamax::exec::Step;
using gamax::exec::StepKind;

const char* exchange_primitive(alps::collectives::Backend backend) {
    switch (backend) {
        case alps::collectives::Backend::CE:
            return "fabric_copy_engine_push";
        case alps::collectives::Backend::CE_PULL:
            return "fabric_copy_engine_pull";
        case alps::collectives::Backend::NCCL:
        default:
            return "nccl_send_recv";
    }
}

alps::collectives::Backend column_exchange_backend(
    const DistributedExecOptions& options)
{
    return options.column_exchange_backend.value_or(options.backend);
}

const char* shuffle_primitive(const DistributedExecOptions& options) {
    if (options.backend == alps::collectives::Backend::CE) {
        return "fabric_copy_engine";
    }
    if (options.backend == alps::collectives::Backend::CE_PULL) {
        return "fabric_copy_engine_pull";
    }
    return options.shuffle_alignment ==
            alps::collectives::AlignmentPolicy::On
        ? "nccl_send_recv_aligned_split" : "nccl_send_recv";
}

const char* replication_primitive(const DistributedExecOptions& options) {
    if (options.backend == alps::collectives::Backend::CE) {
        return "fabric_copy_engine";
    }
    if (options.backend == alps::collectives::Backend::CE_PULL) {
        return "fabric_copy_engine_pull";
    }
    return options.replication_alignment ==
            alps::collectives::AlignmentPolicy::On
        ? "nccl_broadcast_aligned_split" : "nccl_broadcast";
}

void set_alignment_stats(
    gamax::exec::PhaseRange& phase,
    const alps::collectives::AlignmentStats& stats)
{
    if (!stats.applied) return;
    phase.set_alignment(
        stats.rows[0], stats.rows[1], stats.rows[2],
        stats.bytes[0], stats.bytes[1], stats.bytes[2]);
}

void record_implementation(
    const gamax::phys::Context& context,
    const char* requested,
    const char* used,
    const char* fallback_reason = "")
{
    if (context.profile_sink) {
        context.profile_sink->set_active_step_implementation(
            requested, used, fallback_reason);
    }
}

bool is_distributed_step(StepKind kind) {
    switch (kind) {
        case StepKind::PartitionRows:
        case StepKind::ExchangePartitions:
        case StepKind::Replicate:
        case StepKind::GatherToRoot:
        case StepKind::DistributedScalarAgg:
        case StepKind::DistributedGroupBy:
        case StepKind::DistributedDistinct:
        case StepKind::DistributedTopK:
        case StepKind::DistributedSort:
        case StepKind::DistributedEqJoin:
        case StepKind::DistributedSemiJoin:
        case StepKind::DistributedAntiJoin:
        case StepKind::DistributedCrossJoin:
            return true;
        default:
            return false;
    }
}

gamax::table_view require_table(
    const ExtensionInput& input, const char* operation)
{
    if (input.opaque || input.view.num_columns() == 0) {
        throw std::runtime_error(
            std::string("alps::exec::") + operation +
            ": expected a table input");
    }
    return input.view;
}

ExtensionResult execute_step(
    const Step& step,
    const std::vector<ExtensionInput>& inputs,
    const gamax::phys::Context& physical_context,
    const alps::Context& distributed_context,
    const DistributedExecOptions& options)
{
    if (physical_context.stream != distributed_context.stream) {
        throw std::runtime_error(
            "alps::exec: physical and distributed contexts must use the same stream");
    }

    switch (step.kind) {
        case StepKind::PartitionRows: {
            if (inputs.size() != 1) {
                throw std::runtime_error(
                    "alps::exec::PartitionRows expects one input");
            }
            const auto& payload =
                std::get<gamax::exec::PartitionRowsPayload>(step.payload);
            if (payload.strategy != gamax::exec::PartitionStrategy::Hash) {
                throw std::runtime_error(
                    "alps::exec::PartitionRows range strategy is not implemented");
            }
            const auto input = require_table(inputs[0], "PartitionRows");
            gamax::exec::PhaseRange phase(
                physical_context,
                {"shuffle.hash_partition",
                 gamax::exec::PhaseCategory::Compute,
                 gamax::exec::CommunicationPattern::Shuffle,
                 "hash_partition"});
            auto partitioned = alps::collectives::hash_partition(
                input,
                payload.key_ordinals, distributed_context,
                physical_context.stream);
            const auto output = partitioned.table->view();
            phase.set_transfer(
                gamax::exec::table_data_bytes(input),
                gamax::exec::table_data_bytes(output),
                input.num_rows(), output.num_rows(),
                distributed_context.size - 1);
            return ExtensionResult{
                nullptr,
                std::make_shared<alps::collectives::PartitionedTable>(
                    std::move(partitioned))};
        }
        case StepKind::ExchangePartitions: {
            if (inputs.size() != 1 || !inputs[0].opaque) {
                throw std::runtime_error(
                    "alps::exec::ExchangePartitions expects a partition descriptor");
            }
            auto* partitioned = static_cast<
                alps::collectives::PartitionedTable*>(inputs[0].opaque.get());
            const auto input = partitioned->table
                ? partitioned->table->view() : partitioned->borrowed;
            gamax::exec::PhaseRange phase(
                physical_context,
                {"shuffle.exchange",
                 gamax::exec::PhaseCategory::Communication,
                 gamax::exec::CommunicationPattern::Shuffle,
                 shuffle_primitive(options)});
            alps::collectives::AlignmentStats alignment_stats;
            auto result = alps::collectives::exchange_partitions(
                std::move(*partitioned), distributed_context,
                physical_context.stream, options.backend,
                options.shuffle_alignment, &alignment_stats);
            set_alignment_stats(phase, alignment_stats);
            phase.set_transfer(
                gamax::exec::table_data_bytes(input),
                gamax::exec::table_data_bytes(result->view()),
                input.num_rows(), result->num_rows(),
                distributed_context.size - 1);
            return ExtensionResult{std::move(result), nullptr};
        }
        case StepKind::Replicate: {
            if (inputs.size() != 1) {
                throw std::runtime_error("alps::exec::Replicate expects one input");
            }
            const auto input = require_table(inputs[0], "Replicate");
            gamax::exec::PhaseRange phase(
                physical_context,
                {"broadcast.replicate",
                 gamax::exec::PhaseCategory::Communication,
                 gamax::exec::CommunicationPattern::Broadcast,
                 replication_primitive(options)});
            alps::collectives::AlignmentStats alignment_stats;
            auto result = alps::collectives::replicate(
                input, distributed_context,
                physical_context.stream, options.backend,
                options.replication_alignment, &alignment_stats);
            set_alignment_stats(phase, alignment_stats);
            phase.set_transfer(
                gamax::exec::table_data_bytes(input) *
                    (distributed_context.size - 1),
                gamax::exec::table_data_bytes(result->view()) -
                    gamax::exec::table_data_bytes(input),
                static_cast<std::int64_t>(input.num_rows()) *
                    (distributed_context.size - 1),
                result->num_rows() - input.num_rows(),
                distributed_context.size - 1);
            return ExtensionResult{std::move(result), nullptr};
        }
        case StepKind::GatherToRoot: {
            if (inputs.size() != 1) {
                throw std::runtime_error(
                    "alps::exec::GatherToRoot expects one input");
            }
            const auto& payload =
                std::get<gamax::exec::GatherToRootPayload>(step.payload);
            const auto input = require_table(inputs[0], "GatherToRoot");
            gamax::exec::PhaseRange phase(
                physical_context,
                {"gather.to_root",
                 gamax::exec::PhaseCategory::Communication,
                 gamax::exec::CommunicationPattern::Gather,
                 exchange_primitive(options.backend)});
            auto result = alps::collectives::gather_to_root(
                input, payload.root,
                distributed_context, physical_context.stream, options.backend);
            const bool root = distributed_context.rank == payload.root;
            phase.set_transfer(
                root ? 0 : gamax::exec::table_data_bytes(input),
                root ? gamax::exec::table_data_bytes(result->view()) -
                           gamax::exec::table_data_bytes(input) : 0,
                root ? 0 : input.num_rows(),
                root ? result->num_rows() - input.num_rows() : 0,
                distributed_context.size > 1
                    ? (root ? distributed_context.size - 1 : 1) : 0);
            return ExtensionResult{std::move(result), nullptr};
        }
        case StepKind::DistributedScalarAgg: {
            if (inputs.size() != 1) {
                throw std::runtime_error(
                    "alps::exec::DistributedScalarAgg expects one input");
            }
            const auto& payload =
                std::get<gamax::exec::DistributedScalarAggPayload>(step.payload);
            auto result = alps::operators::scalar_aggregate(
                require_table(inputs[0], "DistributedScalarAgg"), payload.specs,
                distributed_context, physical_context);
            return ExtensionResult{std::move(result), nullptr};
        }
        case StepKind::DistributedGroupBy: {
            if (inputs.size() != 1) {
                throw std::runtime_error(
                    "alps::exec::DistributedGroupBy expects one input");
            }
            const auto& payload =
                std::get<gamax::exec::DistributedGroupByPayload>(step.payload);
            if (options.group_by_implementation ==
                DistributedGroupByImplementation::ColumnOverlap) {
                if (payload.hint.groupby_algo != gamax::groupby::HASH) {
                    record_implementation(
                        physical_context, "column_overlap", "legacy",
                        "groupby_algorithm_not_hash");
                } else if (payload.hint.enable_skf) {
                    record_implementation(
                        physical_context, "column_overlap", "legacy",
                        "smart_key_fusion_enabled");
                } else {
                    if (options.backend != alps::collectives::Backend::NCCL &&
                        options.backend != alps::collectives::Backend::CE_PULL) {
                        throw std::invalid_argument(
                            "ColumnOverlap distributed GroupBy requires the "
                            "global backend to be NCCL or CE_PULL");
                    }
                    const auto exchange_backend =
                        column_exchange_backend(options);
                    if (exchange_backend !=
                            alps::collectives::Backend::NCCL &&
                        exchange_backend !=
                            alps::collectives::Backend::CE_PULL) {
                        throw std::invalid_argument(
                            "ColumnOverlap GroupBy transport must be NCCL or "
                            "CE_PULL");
                    }
                    if (options.shuffle_alignment ==
                        alps::collectives::AlignmentPolicy::On) {
                        throw std::invalid_argument(
                            "ColumnOverlap distributed GroupBy does not support "
                            "alignment-aware shuffle");
                    }
                    record_implementation(
                        physical_context, "column_overlap", "column_overlap");
                    auto result =
                        alps::operators::column_overlap_group_by(
                            require_table(inputs[0], "DistributedGroupBy"),
                            payload.n_keys, payload.specs, payload.hint,
                            distributed_context, physical_context,
                            options.column_overlap_high_priority,
                            exchange_backend,
                            options.ce_pull_schedule,
                            options.ce_pull_slots);
                    return ExtensionResult{std::move(result), nullptr};
                }
            } else {
                record_implementation(
                    physical_context, "legacy", "legacy");
            }
            auto result = alps::operators::group_by(
                require_table(inputs[0], "DistributedGroupBy"), payload.n_keys,
                payload.specs, payload.hint, distributed_context,
                physical_context, options.backend, options.shuffle_alignment);
            return ExtensionResult{std::move(result), nullptr};
        }
        case StepKind::DistributedDistinct: {
            if (inputs.size() != 1) {
                throw std::runtime_error(
                    "alps::exec::DistributedDistinct expects one input");
            }
            const auto& payload =
                std::get<gamax::exec::DistributedDistinctPayload>(step.payload);
            auto result = alps::operators::distinct(
                require_table(inputs[0], "DistributedDistinct"),
                payload.key_ordinals, payload.hint, distributed_context,
                physical_context, options.backend, options.shuffle_alignment);
            return ExtensionResult{std::move(result), nullptr};
        }
        case StepKind::DistributedTopK: {
            if (inputs.size() != 1) {
                throw std::runtime_error(
                    "alps::exec::DistributedTopK expects one input");
            }
            const auto& payload =
                std::get<gamax::exec::DistributedTopKPayload>(step.payload);
            auto result = alps::operators::top_k(
                require_table(inputs[0], "DistributedTopK"), payload.keys,
                payload.k, payload.root, payload.hint, distributed_context,
                physical_context, options.backend);
            return ExtensionResult{std::move(result), nullptr};
        }
        case StepKind::DistributedSort: {
            if (inputs.size() != 1) {
                throw std::runtime_error(
                    "alps::exec::DistributedSort expects one input");
            }
            const auto& payload =
                std::get<gamax::exec::DistributedSortPayload>(step.payload);
            auto result = alps::operators::sort(
                require_table(inputs[0], "DistributedSort"), payload.keys,
                payload.root, payload.hint, distributed_context,
                physical_context, options.backend, options.shuffle_alignment,
                options.replication_alignment);
            return ExtensionResult{std::move(result), nullptr};
        }
        case StepKind::DistributedEqJoin: {
            if (inputs.size() != 2) {
                throw std::runtime_error(
                    "alps::exec::DistributedEqJoin expects two inputs");
            }
            const auto& payload =
                std::get<gamax::exec::DistributedEqJoinPayload>(step.payload);
            if (options.eq_join_implementation ==
                    DistributedEqJoinImplementation::ColumnOverlap) {
                if (payload.strategy !=
                    gamax::exec::JoinDistributionStrategy::Shuffle) {
                    record_implementation(
                        physical_context, "column_overlap", "legacy",
                        "distribution_not_shuffle");
                } else if (payload.lkeys.size() != 1 ||
                           payload.rkeys.size() != 1) {
                    record_implementation(
                        physical_context, "column_overlap", "legacy",
                        "multi_key");
                } else if (payload.replicate_left ||
                           payload.replicate_right) {
                    record_implementation(
                        physical_context, "column_overlap", "legacy",
                        "replication_requested");
                } else {
                    if (options.backend != alps::collectives::Backend::NCCL &&
                        options.backend != alps::collectives::Backend::CE_PULL) {
                        throw std::invalid_argument(
                            "ColumnOverlap distributed equality join requires "
                            "the global backend to be NCCL or CE_PULL");
                    }
                    const auto exchange_backend =
                        column_exchange_backend(options);
                    if (exchange_backend !=
                            alps::collectives::Backend::NCCL &&
                        exchange_backend !=
                            alps::collectives::Backend::CE_PULL) {
                        throw std::invalid_argument(
                            "ColumnOverlap equality-join transport must be "
                            "NCCL or CE_PULL");
                    }
                    if (options.shuffle_alignment ==
                            alps::collectives::AlignmentPolicy::On) {
                        throw std::invalid_argument(
                            "ColumnOverlap distributed equality join does not "
                            "support alignment-aware shuffle");
                    }
                    record_implementation(
                        physical_context, "column_overlap", "column_overlap");
                    auto result = alps::operators::column_overlap_eq_join(
                        require_table(inputs[0], "DistributedEqJoin"),
                        payload.lkeys.front(),
                        require_table(inputs[1], "DistributedEqJoin"),
                        payload.rkeys.front(), payload.hint, distributed_context,
                        physical_context.stream, payload.partition_left,
                        payload.partition_right, options.column_overlap_schedule,
                        nullptr, physical_context.profile_sink,
                        options.column_overlap_high_priority,
                        options.column_overlap_early_reclaim,
                        exchange_backend,
                        options.ce_pull_schedule,
                        options.ce_pull_slots);
                    return ExtensionResult{std::move(result), nullptr};
                }
            } else {
                record_implementation(
                    physical_context, "legacy", "legacy");
            }
            auto result = alps::operators::eq_join(
                require_table(inputs[0], "DistributedEqJoin"), payload.lkeys,
                require_table(inputs[1], "DistributedEqJoin"), payload.rkeys,
                payload.strategy, payload.partition_left,
                payload.partition_right, payload.replicate_left,
                payload.replicate_right, payload.hint, distributed_context,
                physical_context, options.backend, options.shuffle_alignment,
                options.replication_alignment);
            return ExtensionResult{std::move(result), nullptr};
        }
        case StepKind::DistributedSemiJoin: {
            if (inputs.size() != 2) {
                throw std::runtime_error(
                    "alps::exec::DistributedSemiJoin expects two inputs");
            }
            const auto& payload =
                std::get<gamax::exec::DistributedSemiJoinPayload>(step.payload);
            if (options.semi_join_implementation ==
                DistributedSemiJoinImplementation::ColumnOverlap) {
                const char* fallback = nullptr;
                if (payload.strategy !=
                    gamax::exec::JoinDistributionStrategy::Shuffle) {
                    fallback = "distribution_not_shuffle";
                } else if (payload.lkeys.size() != 1 ||
                           payload.rkeys.size() != 1) {
                    fallback = "multi_key";
                } else if (payload.replicate_left ||
                           payload.replicate_right) {
                    fallback = "replication_requested";
                } else if (payload.hint.join_algo != gamax::join::HASH) {
                    fallback = "semi_join_algorithm_not_hash";
                } else if (
                    payload.hint.mat_algo == gamax::join::AUTO_MAT ||
                    payload.hint.mat_algo == gamax::join::INDEX_CLUSTER) {
                    fallback = "unsupported_materialization";
                }
                if (!fallback) {
                    if (options.backend != alps::collectives::Backend::NCCL &&
                        options.backend != alps::collectives::Backend::CE_PULL) {
                        throw std::invalid_argument(
                            "ColumnOverlap distributed SemiJoin requires the "
                            "global backend to be NCCL or CE_PULL");
                    }
                    const auto exchange_backend =
                        column_exchange_backend(options);
                    if (exchange_backend !=
                            alps::collectives::Backend::NCCL &&
                        exchange_backend !=
                            alps::collectives::Backend::CE_PULL) {
                        throw std::invalid_argument(
                            "ColumnOverlap SemiJoin transport must be NCCL or "
                            "CE_PULL");
                    }
                    if (options.shuffle_alignment ==
                        alps::collectives::AlignmentPolicy::On) {
                        throw std::invalid_argument(
                            "ColumnOverlap distributed SemiJoin does not support "
                            "alignment-aware shuffle");
                    }
                    record_implementation(
                        physical_context, "column_overlap", "column_overlap");
                    auto result =
                        alps::operators::column_overlap_semi_join(
                            require_table(inputs[0], "DistributedSemiJoin"),
                            payload.lkeys.front(),
                            require_table(inputs[1], "DistributedSemiJoin"),
                            payload.rkeys.front(), payload.partition_left,
                            payload.partition_right, payload.hint,
                            distributed_context, physical_context,
                            options.column_overlap_high_priority,
                            exchange_backend,
                            options.ce_pull_schedule,
                            options.ce_pull_slots);
                    return ExtensionResult{std::move(result), nullptr};
                }
                record_implementation(
                    physical_context, "column_overlap", "legacy", fallback);
            } else {
                record_implementation(
                    physical_context, "legacy", "legacy");
            }
            auto result = alps::operators::semi_join(
                require_table(inputs[0], "DistributedSemiJoin"), payload.lkeys,
                require_table(inputs[1], "DistributedSemiJoin"), payload.rkeys,
                payload.strategy, payload.partition_left,
                payload.partition_right, payload.replicate_left,
                payload.replicate_right, payload.hint, distributed_context,
                physical_context, options.backend, options.shuffle_alignment,
                options.replication_alignment);
            return ExtensionResult{std::move(result), nullptr};
        }
        case StepKind::DistributedAntiJoin: {
            if (inputs.size() != 2) {
                throw std::runtime_error(
                    "alps::exec::DistributedAntiJoin expects two inputs");
            }
            const auto& payload =
                std::get<gamax::exec::DistributedAntiJoinPayload>(step.payload);
            auto result = alps::operators::anti_join(
                require_table(inputs[0], "DistributedAntiJoin"), payload.lkeys,
                require_table(inputs[1], "DistributedAntiJoin"), payload.rkeys,
                payload.strategy, payload.partition_left,
                payload.partition_right, payload.replicate_left,
                payload.replicate_right, payload.hint, distributed_context,
                physical_context, options.backend, options.shuffle_alignment,
                options.replication_alignment);
            return ExtensionResult{std::move(result), nullptr};
        }
        case StepKind::DistributedCrossJoin: {
            if (inputs.size() != 2) {
                throw std::runtime_error(
                    "alps::exec::DistributedCrossJoin expects two inputs");
            }
            const auto& payload =
                std::get<gamax::exec::DistributedCrossJoinPayload>(step.payload);
            auto result = alps::operators::cross_join(
                require_table(inputs[0], "DistributedCrossJoin"),
                require_table(inputs[1], "DistributedCrossJoin"),
                payload.strategy, payload.replicate_left,
                payload.replicate_right, distributed_context,
                physical_context, options.backend,
                options.replication_alignment);
            return ExtensionResult{std::move(result), nullptr};
        }
        default:
            throw std::runtime_error("alps::exec: unhandled distributed step");
    }
}

} // namespace

gamax::exec::StepExtension make_step_extension(
    const alps::Context& distributed_context,
    const DistributedExecOptions& options)
{
    if (options.column_exchange_backend &&
        *options.column_exchange_backend !=
            alps::collectives::Backend::NCCL &&
        *options.column_exchange_backend !=
            alps::collectives::Backend::CE_PULL) {
        throw std::invalid_argument(
            "alps::exec: column_exchange_backend must be NCCL or CE_PULL");
    }
    if (options.backend == alps::collectives::Backend::CE_PULL &&
        (options.shuffle_alignment ==
             alps::collectives::AlignmentPolicy::On ||
         options.replication_alignment ==
             alps::collectives::AlignmentPolicy::On)) {
        throw std::invalid_argument(
            "alps::exec: CE_PULL global communication requires shuffle and "
            "replication alignment to be off");
    }
    switch (options.ce_pull_schedule) {
        case alps::collectives::CePullSchedule::Unordered:
        case alps::collectives::CePullSchedule::RotatedDescriptors:
        case alps::collectives::CePullSchedule::RoundRobinWaves:
            break;
        default:
            throw std::invalid_argument(
                "alps::exec: invalid CE-pull copy schedule");
    }
    if (options.ce_pull_slots < 1) {
        throw std::invalid_argument(
            "alps::exec: CE-pull slot count must be positive");
    }
    const bool uses_ce_pull =
        options.backend == alps::collectives::Backend::CE_PULL ||
        column_exchange_backend(options) ==
            alps::collectives::Backend::CE_PULL;
    if (!uses_ce_pull &&
        options.ce_pull_schedule !=
            alps::collectives::CePullSchedule::RoundRobinWaves) {
        throw std::invalid_argument(
            "alps::exec: a non-default CE-pull schedule requires the "
            "global CE_PULL backend or a CE_PULL ColumnOverlap override");
    }
    if (!uses_ce_pull &&
        options.ce_pull_slots != 2) {
        throw std::invalid_argument(
            "alps::exec: a non-default CE-pull slot count requires the "
            "global CE_PULL backend or a CE_PULL ColumnOverlap override");
    }
    distributed_context.ce_pull_options.slots = options.ce_pull_slots;
    distributed_context.ce_pull_options.schedule = options.ce_pull_schedule;
    distributed_context.ce_pull_options.high_priority_transfer_stream =
        options.column_overlap_high_priority;
    distributed_context.ce_pull_options.prefer_copy_compute_overlap = true;
    distributed_context.ce_pull_options.full_staging = false;
    gamax::exec::StepExtension extension;
    extension.handles = is_distributed_step;
    extension.execute = [&distributed_context, &options](
        const Step& step,
        const std::vector<ExtensionInput>& inputs,
        const gamax::phys::Context& context) {
        return execute_step(
            step, inputs, context, distributed_context, options);
    };
    return extension;
}

std::unique_ptr<gamax::table> execute_distributed(
    const gamax::exec::ExecPlan& plan,
    const gamax::exec::SourceResolver& source_resolver,
    const gamax::phys::Context& physical_context,
    const alps::Context& distributed_context,
    const DistributedExecOptions& options)
{
    auto extension = make_step_extension(distributed_context, options);
    return gamax::exec::execute(
        plan, source_resolver, physical_context, extension);
}

std::unique_ptr<gamax::table> benchmark_run_distributed_steps(
    const gamax::exec::ExecPlan& plan,
    const gamax::exec::BenchExecState& state,
    const gamax::phys::Context& physical_context,
    const alps::Context& distributed_context,
    const DistributedExecOptions& options)
{
    auto extension = make_step_extension(distributed_context, options);
    return gamax::exec::bench_run_steps(
        plan, state, physical_context, extension);
}

} // namespace alps::exec
