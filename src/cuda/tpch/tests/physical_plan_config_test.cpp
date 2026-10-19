// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "tpch/distributed_tpch.hpp"
#include "tpch/physical_plan_config.hpp"
#include "tpch/source_policy.hpp"

namespace {

namespace physical = gamax::tpch::physical_plan;
using gamax::exec::StepKind;

physical::Focus focus(int query) {
    return physical::Focus{query, 300, 8, "NVIDIA B200", "legacy"};
}

TEST(PhysicalPlanConfig, StockDefaultsAreExplicit) {
    auto plan = gamax::tpch::dist::build_local_exec_plan(3);
    const auto config = physical::make_stock_config(focus(3), plan);
    physical::apply(plan, config);
    for (const auto& step : plan.steps) {
        switch (step.kind) {
            case StepKind::Filter:
                EXPECT_TRUE(std::get<gamax::exec::FilterPayload>(
                    step.payload).hint.fuse_operations);
                break;
            case StepKind::Project:
                EXPECT_TRUE(std::get<gamax::exec::ProjectPayload>(
                    step.payload).hint.fuse_operations);
                break;
            case StepKind::GroupBy:
                EXPECT_EQ(std::get<gamax::exec::GroupByPayload>(
                    step.payload).hint.groupby_algo, gamax::groupby::HASH);
                break;
            case StepKind::Distinct:
                EXPECT_EQ(std::get<gamax::exec::DistinctPayload>(
                    step.payload).hint.groupby_algo, gamax::groupby::HASH);
                break;
            case StepKind::EqJoin: {
                const auto& hint = std::get<gamax::exec::EqJoinPayload>(
                    step.payload).hint;
                EXPECT_EQ(hint.join_algo, gamax::join::HASH);
                EXPECT_EQ(hint.mat_algo, gamax::join::GFUR);
                break;
            }
            default:
                break;
        }
    }
}

TEST(PhysicalPlanConfig, Q13FixedOverrideTargetsOnlyStep6) {
    auto plan = gamax::tpch::dist::build_local_exec_plan(13);
    const auto config =
        physical::make_synthesis_seed_config(focus(13), plan);
    ASSERT_EQ(config.fixed_overrides.size(), 1);
    EXPECT_EQ(config.fixed_overrides[0].step_index, 6);
    EXPECT_EQ(config.fixed_overrides[0].step_kind, "GroupBy");
    physical::apply(plan, config);
    const auto& step6 = std::get<gamax::exec::GroupByPayload>(
        plan.steps[6].payload);
    EXPECT_EQ(step6.hint.groupby_algo, gamax::groupby::PARTITION);
    EXPECT_TRUE(step6.hint.allow_hash_fallback);
    const auto& step12 = std::get<gamax::exec::GroupByPayload>(
        plan.steps[12].payload);
    EXPECT_EQ(step12.hint.groupby_algo, gamax::groupby::HASH);
}

TEST(PhysicalPlanConfig, Q21FixedOverridesMatchApprovedSteps) {
    auto plan = gamax::tpch::dist::build_local_exec_plan(21);
    const auto config =
        physical::make_synthesis_seed_config(focus(21), plan);
    ASSERT_EQ(config.fixed_overrides.size(), 4);
    physical::apply(plan, config);
    for (const int step_index : {6, 12}) {
        const auto& payload = std::get<gamax::exec::GroupByPayload>(
            plan.steps[static_cast<std::size_t>(step_index)].payload);
        EXPECT_EQ(payload.hint.groupby_algo, gamax::groupby::PARTITION);
        EXPECT_TRUE(payload.hint.allow_hash_fallback);
    }
    for (const int step_index : {4, 10}) {
        const auto& payload = std::get<gamax::exec::DistinctPayload>(
            plan.steps[static_cast<std::size_t>(step_index)].payload);
        EXPECT_EQ(payload.hint.groupby_algo, gamax::groupby::SORT);
        EXPECT_TRUE(payload.hint.enable_skf);
    }
    const auto& step26 = std::get<gamax::exec::GroupByPayload>(
        plan.steps[26].payload);
    EXPECT_EQ(step26.hint.groupby_algo, gamax::groupby::HASH);
}

TEST(PhysicalPlanConfig, Q21PartitionCapacityFallbackIsExplicitAndPersistent) {
    auto plan = gamax::tpch::dist::build_local_exec_plan(21);
    auto config =
        physical::make_synthesis_seed_config(focus(21), plan);
    EXPECT_TRUE(physical::fallback_partition_groupbys_to_hash(
        config, "partition_groupby_capacity"));
    ASSERT_EQ(config.seed_fallbacks.size(), 2);
    EXPECT_EQ(config.seed_fallbacks[0].step_index, 6);
    EXPECT_EQ(config.seed_fallbacks[0].from_implementation, "partition_gftr");
    EXPECT_EQ(config.seed_fallbacks[0].to_implementation, "hash");
    EXPECT_EQ(config.seed_fallbacks[0].reason, "partition_groupby_capacity");
    EXPECT_FALSE(physical::fallback_partition_groupbys_to_hash(
        config, "partition_groupby_capacity"));

    physical::apply(plan, config);
    for (const int step_index : {6, 12}) {
        const auto& payload = std::get<gamax::exec::GroupByPayload>(
            plan.steps[static_cast<std::size_t>(step_index)].payload);
        EXPECT_EQ(payload.hint.groupby_algo, gamax::groupby::HASH);
    }

    const auto path = std::filesystem::temp_directory_path() /
        "gamax_q21_seed_fallback_test.json";
    physical::write_json_atomic(config, path.string());
    const auto loaded = physical::read_json(path.string());
    ASSERT_EQ(loaded.seed_fallbacks.size(), 2);
    EXPECT_EQ(loaded.seed_fallbacks[1].step_index, 12);
    EXPECT_EQ(loaded.fixed_overrides[0].implementation, "hash");
    EXPECT_EQ(loaded.fixed_overrides[1].implementation, "hash");
    std::filesystem::remove(path);
}

TEST(PhysicalPlanConfig, PerJoinOverrideSurvivesDistributedPlanning) {
    auto local = gamax::tpch::dist::build_local_exec_plan(3);
    auto config =
        physical::make_synthesis_seed_config(focus(3), local);
    ASSERT_EQ(config.join_overrides.size(), 2);
    config.join_overrides[1].implementation = "partition_gftr";
    physical::apply(local, config);
    auto distributed = gamax::tpch::dist::distribute_exec_plan(
        3, gamax::tpch::make_distributed_tpch_source_read_policies(),
        std::move(local));
    int join = 0;
    for (const auto& step : distributed.steps) {
        if (step.kind != StepKind::DistributedEqJoin) continue;
        const auto& hint =
            std::get<gamax::exec::DistributedEqJoinPayload>(
                step.payload).hint;
        if (join == 0) {
            EXPECT_EQ(hint.join_algo, gamax::join::HASH);
            EXPECT_EQ(hint.mat_algo, gamax::join::GFUR);
        } else if (join == 1) {
            EXPECT_EQ(hint.join_algo, gamax::join::PARTITION);
            EXPECT_EQ(hint.mat_algo, gamax::join::GFTR);
        }
        ++join;
    }
    EXPECT_EQ(join, 2);
}

TEST(PhysicalPlanConfig, FingerprintsSeparateStructureAndHints) {
    auto baseline = gamax::tpch::dist::build_local_exec_plan(3);
    auto changed = gamax::tpch::dist::build_local_exec_plan(3);
    auto config =
        physical::make_synthesis_seed_config(focus(3), changed);
    config.join_overrides[0].implementation = "sort_gfur";
    physical::apply(changed, config);
    EXPECT_EQ(
        physical::structure_fingerprint(baseline),
        physical::structure_fingerprint(changed));
    EXPECT_NE(
        physical::physical_fingerprint(baseline),
        physical::physical_fingerprint(changed));
}

TEST(PhysicalPlanConfig, JsonRoundTripAndStaleGuards) {
    auto plan = gamax::tpch::dist::build_local_exec_plan(3);
    auto config =
        physical::make_synthesis_seed_config(focus(3), plan);
    config.search.seed = 42;
    config.search.coordinate_order = {5, 2};
    const auto path = std::filesystem::temp_directory_path() /
        "gamax_physical_plan_config_test.json";
    physical::write_json_atomic(config, path.string());
    const auto loaded = physical::read_json(path.string());
    EXPECT_EQ(loaded.focus.scale_factor, 300);
    EXPECT_EQ(loaded.focus.ranks, 8);
    EXPECT_EQ(loaded.search.seed, 42);
    EXPECT_EQ(loaded.join_overrides.size(), config.join_overrides.size());
    EXPECT_NO_THROW(physical::validate_focus(loaded, focus(3)));

    auto wrong_focus = focus(3);
    wrong_focus.ranks = 4;
    EXPECT_THROW(
        physical::validate_focus(loaded, wrong_focus),
        std::runtime_error);

    auto stale = loaded;
    stale.join_overrides[0].semantic_label = "stale";
    EXPECT_THROW(physical::apply(plan, stale), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(PhysicalPlanConfig, CoordinateDecisionUsesApprovedNoiseThresholds) {
    EXPECT_EQ(
        physical::initial_candidate_action(100.0, 96.9, 0.03),
        physical::CandidateAction::CommitChallenger);
    EXPECT_EQ(
        physical::initial_candidate_action(100.0, 98.0, 0.03),
        physical::CandidateAction::RunTieBreak);
    EXPECT_EQ(
        physical::initial_candidate_action(100.0, 100.0, 0.03),
        physical::CandidateAction::KeepIncumbent);
    EXPECT_TRUE(
        physical::commit_after_tie_break(100.0, 99.0, 0.01));
    EXPECT_FALSE(
        physical::commit_after_tie_break(100.0, 99.1, 0.01));
}

TEST(PhysicalPlanConfig, CurrentSuiteHasExpected49JoinCoordinates) {
    int total = 0;
    for (int query = 1; query <= 22; ++query) {
        const auto plan =
            gamax::tpch::dist::build_local_exec_plan(query);
        total += static_cast<int>(
            physical::inner_join_coordinates(plan).size());
    }
    EXPECT_EQ(total, 49);
}

} // namespace
