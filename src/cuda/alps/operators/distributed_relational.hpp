// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <vector>

#include "alps/collectives/alltoall.hpp"
#include "alps/collectives/ce_pull_shuffle.hpp"
#include "alps/context.hpp"
#include "exec/exec_plan.hpp"
#include "memory/table.hpp"
#include "phys/phys.hpp"

namespace alps::operators {

std::unique_ptr<gamax::table> scalar_aggregate(
    gamax::table_view input,
    const std::vector<gamax::phys::AggSpec>& specs,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context);

std::unique_ptr<gamax::table> group_by(
    gamax::table_view input,
    gamax::size_type n_keys,
    const std::vector<gamax::phys::AggSpec>& specs,
    const gamax::groupby::groupby_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend = alps::collectives::Backend::NCCL,
    alps::collectives::AlignmentPolicy shuffle_alignment =
        alps::collectives::AlignmentPolicy::Off);

// HASH-only whole-column schedule: exchange all key columns first, compute the
// second-stage group-ID plan as soon as they arrive, and overlap that work with
// aggregate-state column exchange.
std::unique_ptr<gamax::table> column_overlap_group_by(
    gamax::table_view input,
    gamax::size_type n_keys,
    const std::vector<gamax::phys::AggSpec>& specs,
    const gamax::groupby::groupby_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    bool communication_high_priority = true,
    alps::collectives::Backend exchange_backend =
        alps::collectives::Backend::NCCL,
    alps::collectives::CePullSchedule ce_pull_schedule =
        alps::collectives::CePullSchedule::RoundRobinWaves,
    int ce_pull_slots = 2);

std::unique_ptr<gamax::table> distinct(
    gamax::table_view input,
    const std::vector<gamax::size_type>& key_ordinals,
    const gamax::groupby::groupby_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend = alps::collectives::Backend::NCCL,
    alps::collectives::AlignmentPolicy shuffle_alignment =
        alps::collectives::AlignmentPolicy::Off);

std::unique_ptr<gamax::table> top_k(
    gamax::table_view input,
    const std::vector<gamax::phys::SortSpec>& keys,
    gamax::size_type k,
    int root,
    const gamax::sort::sort_table_options& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend = alps::collectives::Backend::NCCL);

std::unique_ptr<gamax::table> sort(
    gamax::table_view input,
    const std::vector<gamax::phys::SortSpec>& keys,
    int root,
    const gamax::sort::sort_table_options& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend = alps::collectives::Backend::NCCL,
    alps::collectives::AlignmentPolicy shuffle_alignment =
        alps::collectives::AlignmentPolicy::Off,
    alps::collectives::AlignmentPolicy replication_alignment =
        alps::collectives::AlignmentPolicy::Off);

std::unique_ptr<gamax::table> eq_join(
    gamax::table_view left,
    const std::vector<gamax::size_type>& left_keys,
    gamax::table_view right,
    const std::vector<gamax::size_type>& right_keys,
    gamax::exec::JoinDistributionStrategy strategy,
    bool partition_left,
    bool partition_right,
    bool replicate_left,
    bool replicate_right,
    const gamax::join::join_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend = alps::collectives::Backend::NCCL,
    alps::collectives::AlignmentPolicy shuffle_alignment =
        alps::collectives::AlignmentPolicy::Off,
    alps::collectives::AlignmentPolicy replication_alignment =
        alps::collectives::AlignmentPolicy::Off);

std::unique_ptr<gamax::table> semi_join(
    gamax::table_view left,
    const std::vector<gamax::size_type>& left_keys,
    gamax::table_view right,
    const std::vector<gamax::size_type>& right_keys,
    gamax::exec::JoinDistributionStrategy strategy,
    bool partition_left,
    bool partition_right,
    bool replicate_left,
    bool replicate_right,
    const gamax::join::join_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend = alps::collectives::Backend::NCCL,
    alps::collectives::AlignmentPolicy shuffle_alignment =
        alps::collectives::AlignmentPolicy::Off,
    alps::collectives::AlignmentPolicy replication_alignment =
        alps::collectives::AlignmentPolicy::Off);

// Shuffle-only, single-key HASH semi-join schedule. The right/build key is
// exchanged first and its set is built while left/probe columns are exchanged.
std::unique_ptr<gamax::table> column_overlap_semi_join(
    gamax::table_view left,
    gamax::size_type left_key,
    gamax::table_view right,
    gamax::size_type right_key,
    bool partition_left,
    bool partition_right,
    const gamax::join::join_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    bool communication_high_priority = true,
    alps::collectives::Backend exchange_backend =
        alps::collectives::Backend::NCCL,
    alps::collectives::CePullSchedule ce_pull_schedule =
        alps::collectives::CePullSchedule::RoundRobinWaves,
    int ce_pull_slots = 2);

std::unique_ptr<gamax::table> anti_join(
    gamax::table_view left,
    const std::vector<gamax::size_type>& left_keys,
    gamax::table_view right,
    const std::vector<gamax::size_type>& right_keys,
    gamax::exec::JoinDistributionStrategy strategy,
    bool partition_left,
    bool partition_right,
    bool replicate_left,
    bool replicate_right,
    const gamax::join::join_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend = alps::collectives::Backend::NCCL,
    alps::collectives::AlignmentPolicy shuffle_alignment =
        alps::collectives::AlignmentPolicy::Off,
    alps::collectives::AlignmentPolicy replication_alignment =
        alps::collectives::AlignmentPolicy::Off);

std::unique_ptr<gamax::table> cross_join(
    gamax::table_view left,
    gamax::table_view right,
    gamax::exec::JoinDistributionStrategy strategy,
    bool replicate_left,
    bool replicate_right,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend = alps::collectives::Backend::NCCL,
    alps::collectives::AlignmentPolicy replication_alignment =
        alps::collectives::AlignmentPolicy::Off);

} // namespace alps::operators
