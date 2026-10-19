// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "exec/exec_plan.hpp"

namespace gamax::tpch::physical_plan {

inline constexpr int kSchemaVersion = 1;

struct Focus {
    int query_id {0};
    int scale_factor {0};
    int ranks {0};
    std::string gpu_model;
    std::string eq_join_backend {"legacy"};
};

struct Defaults {
    std::string filter {"nofuse"};
    std::string project {"nofuse"};
    std::string group_by {"hash"};
    std::string distinct {"hash"};
    std::string sort {"no_skf"};
    std::string top_k {"no_skf"};
    std::string eq_join {"hash_gfur"};
    std::string semi_join {"hash_gfur"};
    std::string anti_join {"hash_gfur"};
};

struct StepOverride {
    int step_index {-1};
    std::string step_kind;
    std::string semantic_label;
    std::string implementation;
};

struct CandidateRecord {
    int pass {0};
    int coordinate_position {0};
    int step_index {-1};
    std::string implementation;
    std::string status;
    double median_ms {0.0};
    int samples {0};
    std::string detail;
};

struct SeedFallback {
    int step_index {-1};
    std::string step_kind;
    std::string from_implementation;
    std::string to_implementation;
    std::string reason;
};

struct SearchState {
    std::uint64_t seed {0};
    int pass {0};
    int coordinate_position {0};
    int max_passes {3};
    bool completed {false};
    bool converged {false};
    std::optional<int> active_step_index;
    std::string active_implementation;
    std::vector<int> coordinate_order;
    std::vector<CandidateRecord> records;
};

struct Config {
    int schema_version {kSchemaVersion};
    std::string preset {"synthesis_seed"};
    Focus focus;
    Defaults defaults;
    std::string structure_fingerprint;
    std::vector<StepOverride> fixed_overrides;
    std::vector<StepOverride> join_overrides;
    std::vector<SeedFallback> seed_fallbacks;
    SearchState search;
};

struct JoinCoordinate {
    int step_index {-1};
    std::string step_kind;
    std::string semantic_label;
};

enum class CandidateAction {
    KeepIncumbent,
    CommitChallenger,
    RunTieBreak,
};

CandidateAction initial_candidate_action(
    double incumbent_median_ms,
    double challenger_median_ms,
    double direct_commit_fraction);
bool commit_after_tie_break(
    double incumbent_median_ms,
    double challenger_median_ms,
    double tie_commit_fraction);

const std::vector<std::string>& join_candidates();

std::string structure_fingerprint(const gamax::exec::ExecPlan& plan);
std::string physical_fingerprint(const gamax::exec::ExecPlan& plan);
std::vector<JoinCoordinate> inner_join_coordinates(
    const gamax::exec::ExecPlan& local_plan);

Config make_stock_config(
    const Focus& focus,
    const gamax::exec::ExecPlan& local_plan);
Config make_synthesis_seed_config(
    const Focus& focus,
    const gamax::exec::ExecPlan& local_plan);
bool fallback_partition_groupbys_to_hash(
    Config& config,
    const std::string& reason);

void validate_focus(
    const Config& config,
    const Focus& actual);
void apply(
    gamax::exec::ExecPlan& local_plan,
    const Config& config);

Config read_json(const std::string& path);
void write_json_atomic(
    const Config& config,
    const std::string& path);

} // namespace gamax::tpch::physical_plan
