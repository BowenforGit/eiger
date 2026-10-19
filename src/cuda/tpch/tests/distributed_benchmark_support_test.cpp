// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "exec/exec_plan.hpp"
#include "exec/print.hpp"
#include "exec/step_profile.hpp"
#include "tpch/distributed_benchmark_support.hpp"

namespace {

namespace bench = gamax::tpch::benchmark;
namespace gexec = gamax::exec;

TEST(DistributedBenchmarkSupportTest, AdaptiveWarmupConverges) {
    bench::WarmupPolicy policy;
    EXPECT_EQ(
        bench::evaluate_warmup({100, 98, 101, 99}, policy),
        bench::WarmupStatus::Continue);
    EXPECT_EQ(
        bench::evaluate_warmup({100, 98, 101, 99, 100}, policy),
        bench::WarmupStatus::Stable);
}

TEST(DistributedBenchmarkSupportTest, AdaptiveWarmupCapsNoisyRuns) {
    bench::WarmupPolicy policy;
    policy.maximum_runs = 5;
    EXPECT_EQ(
        bench::evaluate_warmup({100, 200, 100, 200, 100}, policy),
        bench::WarmupStatus::MaximumReached);
}

TEST(DistributedBenchmarkSupportTest, RejectsImpossibleWarmupPolicy) {
    bench::WarmupPolicy policy;
    policy.maximum_runs = 4;
    EXPECT_THROW(bench::validate_warmup_policy(policy), std::invalid_argument);
}

TEST(DistributedBenchmarkSupportTest, ComputesSummaryStatistics) {
    const auto stats = bench::summarize({1.0, 2.0, 3.0, 4.0});
    EXPECT_DOUBLE_EQ(stats.minimum, 1.0);
    EXPECT_DOUBLE_EQ(stats.mean, 2.5);
    EXPECT_DOUBLE_EQ(stats.median, 2.5);
    EXPECT_DOUBLE_EQ(stats.maximum, 4.0);
    EXPECT_NEAR(stats.stddev, 1.11803398875, 1e-10);
}

TEST(DistributedBenchmarkSupportTest, DerivesSameSampleResidualPerRank) {
    const auto residual = bench::derive_residual_timing(
        {10.0, 20.0, 30.0}, {8.0, 20.005, 31.0});
    EXPECT_EQ(residual.rank_ms, (std::vector<double>{2.0, 0.0, 0.0}));
    EXPECT_TRUE(residual.negative_beyond_tolerance);

    EXPECT_THROW(
        bench::derive_residual_timing({1.0}, {1.0, 2.0}),
        std::invalid_argument);
}

TEST(DistributedBenchmarkSupportTest, PhaseUnionHandlesDisjointIntervals) {
    EXPECT_FLOAT_EQ(
        gexec::phase_interval_union_ms({{0.0f, 2.0f}, {3.0f, 5.0f}}),
        4.0f);
}

TEST(DistributedBenchmarkSupportTest, PhaseUnionHandlesNestedIntervals) {
    EXPECT_FLOAT_EQ(
        gexec::phase_interval_union_ms(
            {{0.0f, 10.0f}, {2.0f, 4.0f}, {6.0f, 9.0f}}),
        10.0f);
}

TEST(DistributedBenchmarkSupportTest, PhaseUnionHandlesPartialOverlap) {
    EXPECT_FLOAT_EQ(
        gexec::phase_interval_union_ms(
            {{0.0f, 4.0f}, {2.0f, 7.0f}, {8.0f, 9.0f}}),
        8.0f);
}

TEST(DistributedBenchmarkSupportTest, PhaseUnionHandlesFullOverlapAndBoundaries) {
    EXPECT_NEAR(
        gexec::phase_interval_union_ms(
            {{1.0f, 5.0f}, {1.0f, 5.0f}, {5.0f, 6.0f}}),
        5.0f, 1e-6f);
}

TEST(DistributedBenchmarkSupportTest, EscapesCsvFields) {
    EXPECT_EQ(bench::csv_escape("plain"), "plain");
    EXPECT_EQ(bench::csv_escape("a,b"), "\"a,b\"");
    EXPECT_EQ(bench::csv_escape("a\"b"), "\"a\"\"b\"");
}

TEST(DistributedBenchmarkSupportTest, CsvSchemasContainRequiredMetadata) {
    const std::string e2e(bench::e2e_csv_header());
    const std::string steps(bench::steps_csv_header());
    const std::string phases(bench::phases_csv_header());
    const std::string memory(bench::memory_csv_header());
    const std::string messages(bench::messages_csv_header());
    EXPECT_EQ(std::count(e2e.begin(), e2e.end(), ','), 23);
    EXPECT_EQ(std::count(steps.begin(), steps.end(), ','), 26);
    EXPECT_NE(e2e.find("rank_samples_ms"), std::string::npos);
    EXPECT_NE(e2e.find("plan_fingerprint"), std::string::npos);
    EXPECT_NE(steps.find("record_type,rank,step_index"), std::string::npos);
    EXPECT_NE(steps.find("step_label"), std::string::npos);
    EXPECT_NE(
        steps.find(
            "requested_implementation,used_implementation,fallback_reason"),
        std::string::npos);
    EXPECT_EQ(std::count(phases.begin(), phases.end(), ','), 44);
    EXPECT_EQ(std::count(memory.begin(), memory.end(), ','), 22);
    EXPECT_EQ(std::count(messages.begin(), messages.end(), ','), 35);
    EXPECT_NE(e2e.find("physical_plan_preset"), std::string::npos);
    EXPECT_NE(phases.find(
                  "phase_name,column_side,column_ordinal,category,"
                  "communication_pattern"),
              std::string::npos);
    EXPECT_NE(phases.find("bytes_sent,bytes_received,rows_sent,rows_received"),
              std::string::npos);
    EXPECT_NE(
        messages.find("api,direction,peer,root,column_ordinal,column_type"),
        std::string::npos);
    EXPECT_NE(
        messages.find("element_count,element_size,payload_bytes,is_self"),
        std::string::npos);
}

TEST(DistributedBenchmarkSupportTest, RejectsCsvSchemaMismatchBeforeAppend) {
    const auto unique = std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count();
    const auto path = std::filesystem::temp_directory_path() /
        ("gamax_csv_schema_test_" + std::to_string(unique) + ".csv");

    EXPECT_TRUE(bench::csv_needs_header(
        path.string(), bench::phases_csv_header()));
    {
        std::ofstream output(path);
    }
    EXPECT_TRUE(bench::csv_needs_header(
        path.string(), bench::phases_csv_header()));
    {
        std::ofstream output(path);
        output << bench::phases_csv_header() << "\r\n";
    }
    EXPECT_FALSE(bench::csv_needs_header(
        path.string(), bench::phases_csv_header()));
    {
        std::ofstream output(path);
        output << "old,phase,header\n";
    }
    EXPECT_THROW(
        bench::csv_needs_header(path.string(), bench::phases_csv_header()),
        std::runtime_error);

    std::error_code cleanup_error;
    std::filesystem::remove(path, cleanup_error);
}

TEST(DistributedBenchmarkSupportTest, FingerprintIsStableAndSensitive) {
    const auto a = bench::stable_fingerprint("plan-a");
    EXPECT_EQ(a, bench::stable_fingerprint("plan-a"));
    EXPECT_NE(a, bench::stable_fingerprint("plan-b"));
    EXPECT_EQ(bench::fingerprint_hex(a).size(), 16);
}

TEST(DistributedBenchmarkSupportTest, StepLabelUsesPhysicalSemantics) {
    gexec::ExecPlan plan;
    gexec::Value left;
    left.id = 0;
    left.last_used_step = 2;
    left.schema.append(gamax::ir::Field{
        gamax::ir::ColumnId{0}, gamax::data_type{gamax::type_id::INT64},
        false, "orders_key"});
    gexec::Value right;
    right.id = 1;
    right.last_used_step = 2;
    right.schema.append(gamax::ir::Field{
        gamax::ir::ColumnId{1}, gamax::data_type{gamax::type_id::INT64},
        false, "lineitem_key"});
    gexec::Value output;
    output.id = 2;
    output.last_used_step = -1;
    output.schema = left.schema;
    plan.values = {left, right, output};
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 0,
        gexec::SourceTablePayload{"orders", {"orders_key"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 1,
        gexec::SourceTablePayload{"lineitem", {"lineitem_key"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::DistributedEqJoin, {0, 1}, 2,
        gexec::DistributedEqJoinPayload{
            {0}, {0}, gexec::JoinDistributionStrategy::BroadcastLeft,
            false, false, false, false, {}}});

    const auto label = gamax::exec::format_step_label(plan, 2);
    EXPECT_NE(label.find("DistributedEqJoin"), std::string::npos);
    EXPECT_NE(label.find("orders_key"), std::string::npos);
    EXPECT_NE(label.find("lineitem_key"), std::string::npos);
    EXPECT_NE(label.find("broadcast-left"), std::string::npos);
}

TEST(DistributedBenchmarkSupportTest, StepLabelsDescribeDistributionAndOrder) {
    gexec::ExecPlan plan;
    for (int id = 0; id < 4; ++id) {
        gexec::Value value;
        value.id = id;
        value.last_used_step = id == 3 ? -1 : id + 1;
        value.schema.append(gamax::ir::Field{
            gamax::ir::ColumnId{0},
            gamax::data_type{gamax::type_id::INT64}, false, "order_key"});
        plan.values.push_back(std::move(value));
    }
    plan.values[2].distribution.kind =
        gexec::DistributionKind::HashPartitioned;
    plan.values[2].distribution.key_ordinals = {0};
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 0,
        gexec::SourceTablePayload{"orders", {"order_key"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::PartitionRows, {0}, 1,
        gexec::PartitionRowsPayload{
            gexec::PartitionStrategy::Hash, {0}, {}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::ExchangePartitions, {1}, 2,
        gexec::ExchangePartitionsPayload{}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::DistributedTopK, {2}, 3,
        gexec::DistributedTopKPayload{
            {{0, gamax::sort_order::DESCENDING}}, 10, {}, 0}});

    const auto partition = gamax::exec::format_step_label(plan, 1);
    const auto exchange = gamax::exec::format_step_label(plan, 2);
    const auto topk = gamax::exec::format_step_label(plan, 3);
    EXPECT_NE(partition.find("strategy=hash"), std::string::npos);
    EXPECT_NE(partition.find("order_key"), std::string::npos);
    EXPECT_NE(exchange.find("hash-partitioned"), std::string::npos);
    EXPECT_NE(exchange.find("order_key"), std::string::npos);
    EXPECT_NE(topk.find("order_key DESC"), std::string::npos);
    EXPECT_NE(topk.find("k=10"), std::string::npos);
    EXPECT_NE(topk.find("root=0"), std::string::npos);
}

TEST(DistributedBenchmarkSupportTest, GroupByLabelsUseValueRelativeColumns) {
    gexec::ExecPlan plan;
    gexec::Value input;
    input.id = 0;
    input.last_used_step = 0;
    for (const auto* name : {"group_key", "quantity", "price"}) {
        input.schema.append(gamax::ir::Field{
            gamax::ir::ColumnId{static_cast<gamax::size_type>(
                input.schema.size())},
            gamax::data_type{gamax::type_id::INT64}, false, name});
    }
    gexec::Value output;
    output.id = 1;
    output.last_used_step = -1;
    plan.values = {input, output};
    const std::vector<gamax::phys::AggSpec> specs = {
        {0, gamax::groupby::aggregation_id::SUM},
        {1, gamax::groupby::aggregation_id::MAX}};

    plan.steps.push_back(gexec::Step{
        gexec::StepKind::GroupBy, {0}, 1,
        gexec::GroupByPayload{1, specs, {}}});
    EXPECT_EQ(
        gamax::exec::format_step_label(plan, 0),
        "GroupBy(keys=[group_key];aggs=[sum(quantity),max(price)])");

    plan.steps[0].kind = gexec::StepKind::DistributedGroupBy;
    plan.steps[0].payload = gexec::DistributedGroupByPayload{1, specs, {}};
    EXPECT_EQ(
        gamax::exec::format_step_label(plan, 0),
        "DistributedGroupBy(keys=[group_key];aggs=[sum(quantity),max(price)])");
}

} // namespace
