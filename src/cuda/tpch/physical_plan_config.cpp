// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "tpch/physical_plan_config.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>

#include "exec/print.hpp"
#include "tpch/distributed_benchmark_support.hpp"

namespace gamax::tpch::physical_plan {
namespace {

using gamax::exec::StepKind;

[[noreturn]] void config_error(const std::string& message) {
    throw std::runtime_error("physical-plan config: " + message);
}

gamax::join::join_hint parse_join(const std::string& value) {
    gamax::join::join_hint hint;
    if (value == "hash_gfur") {
        hint.join_algo = gamax::join::HASH;
        hint.mat_algo = gamax::join::GFUR;
    } else if (value == "sort_gfur") {
        hint.join_algo = gamax::join::SORT;
        hint.mat_algo = gamax::join::GFUR;
    } else if (value == "sort_gftr") {
        hint.join_algo = gamax::join::SORT;
        hint.mat_algo = gamax::join::GFTR;
    } else if (value == "partition_gfur") {
        hint.join_algo = gamax::join::PARTITION;
        hint.mat_algo = gamax::join::GFUR;
    } else if (value == "partition_gftr") {
        hint.join_algo = gamax::join::PARTITION;
        hint.mat_algo = gamax::join::GFTR;
    } else {
        config_error("unknown join implementation '" + value + "'");
    }
    return hint;
}

void apply_group_by(
    gamax::groupby::groupby_hint& hint,
    const std::string& value)
{
    if (value == "hash") {
        hint.groupby_algo = gamax::groupby::HASH;
        hint.enable_skf = false;
    } else if (value == "partition_gftr") {
        hint.groupby_algo = gamax::groupby::PARTITION;
        hint.enable_skf = false;
        hint.allow_hash_fallback = true;
    } else {
        config_error("unknown group-by implementation '" + value + "'");
    }
}

void apply_distinct(
    gamax::groupby::groupby_hint& hint,
    const std::string& value)
{
    if (value == "hash") {
        hint.groupby_algo = gamax::groupby::HASH;
        hint.enable_skf = false;
    } else if (value == "sort_skf") {
        hint.groupby_algo = gamax::groupby::SORT;
        hint.enable_skf = true;
    } else {
        config_error("unknown distinct implementation '" + value + "'");
    }
}

void apply_defaults(
    gamax::exec::ExecPlan& plan,
    const Defaults& defaults)
{
    for (auto& step : plan.steps) {
        switch (step.kind) {
            case StepKind::Filter: {
                auto& hint =
                    std::get<gamax::exec::FilterPayload>(step.payload).hint;
                if (defaults.filter == "fuse") hint.fuse_operations = true;
                else if (defaults.filter == "nofuse") hint.fuse_operations = false;
                else config_error("unknown filter default '" + defaults.filter + "'");
                break;
            }
            case StepKind::Project: {
                auto& hint =
                    std::get<gamax::exec::ProjectPayload>(step.payload).hint;
                if (defaults.project == "fuse") hint.fuse_operations = true;
                else if (defaults.project == "nofuse") hint.fuse_operations = false;
                else config_error("unknown project default '" + defaults.project + "'");
                break;
            }
            case StepKind::GroupBy:
                apply_group_by(
                    std::get<gamax::exec::GroupByPayload>(step.payload).hint,
                    defaults.group_by);
                break;
            case StepKind::Distinct:
                apply_distinct(
                    std::get<gamax::exec::DistinctPayload>(step.payload).hint,
                    defaults.distinct);
                break;
            case StepKind::Sort: {
                auto& hint =
                    std::get<gamax::exec::SortPayload>(step.payload).hint;
                if (defaults.sort == "no_skf") hint.smart_keys_fusion = false;
                else if (defaults.sort == "skf") hint.smart_keys_fusion = true;
                else config_error("unknown sort default '" + defaults.sort + "'");
                break;
            }
            case StepKind::TopK: {
                auto& hint =
                    std::get<gamax::exec::TopKPayload>(step.payload).hint;
                if (defaults.top_k == "no_skf") hint.smart_keys_fusion = false;
                else if (defaults.top_k == "skf") hint.smart_keys_fusion = true;
                else config_error("unknown top-k default '" + defaults.top_k + "'");
                break;
            }
            case StepKind::EqJoin:
                std::get<gamax::exec::EqJoinPayload>(step.payload).hint =
                    parse_join(defaults.eq_join);
                break;
            case StepKind::SemiJoin:
                std::get<gamax::exec::SemiJoinPayload>(step.payload).hint =
                    parse_join(defaults.semi_join);
                break;
            case StepKind::AntiJoin:
                std::get<gamax::exec::AntiJoinPayload>(step.payload).hint =
                    parse_join(defaults.anti_join);
                break;
            default:
                break;
        }
    }
}

void apply_override(
    gamax::exec::ExecPlan& plan,
    const StepOverride& override)
{
    if (override.step_index < 0 ||
        static_cast<std::size_t>(override.step_index) >= plan.steps.size()) {
        config_error(
            "step " + std::to_string(override.step_index) + " is out of range");
    }
    auto& step = plan.steps[static_cast<std::size_t>(override.step_index)];
    const std::string actual_kind = gamax::exec::step_kind_name(step.kind);
    if (actual_kind != override.step_kind) {
        config_error(
            "step " + std::to_string(override.step_index) + " kind is " +
            actual_kind + ", expected " + override.step_kind);
    }
    const std::string actual_label = gamax::exec::format_step_label(
        plan, override.step_index);
    if (actual_label != override.semantic_label) {
        config_error(
            "step " + std::to_string(override.step_index) +
            " semantic label mismatch; actual '" + actual_label +
            "', expected '" + override.semantic_label + "'");
    }

    switch (step.kind) {
        case StepKind::EqJoin:
            std::get<gamax::exec::EqJoinPayload>(step.payload).hint =
                parse_join(override.implementation);
            break;
        case StepKind::GroupBy:
            apply_group_by(
                std::get<gamax::exec::GroupByPayload>(step.payload).hint,
                override.implementation);
            break;
        case StepKind::Distinct:
            apply_distinct(
                std::get<gamax::exec::DistinctPayload>(step.payload).hint,
                override.implementation);
            break;
        default:
            config_error(
                "step override for unsupported kind " + actual_kind);
    }
}

StepOverride make_override(
    const gamax::exec::ExecPlan& plan,
    int step_index,
    StepKind expected_kind,
    std::string implementation)
{
    if (step_index < 0 ||
        static_cast<std::size_t>(step_index) >= plan.steps.size()) {
        config_error(
            "hardwired step " + std::to_string(step_index) + " is out of range");
    }
    const auto& step = plan.steps[static_cast<std::size_t>(step_index)];
    if (step.kind != expected_kind) {
        config_error(
            "hardwired step " + std::to_string(step_index) + " is " +
            gamax::exec::step_kind_name(step.kind) + ", expected " +
            gamax::exec::step_kind_name(expected_kind));
    }
    return StepOverride{
        step_index,
        gamax::exec::step_kind_name(step.kind),
        gamax::exec::format_step_label(plan, step_index),
        std::move(implementation)};
}

void normalize_hints(gamax::exec::ExecPlan& plan) {
    Defaults normalized;
    normalized.filter = "fuse";
    normalized.project = "fuse";
    normalized.group_by = "hash";
    normalized.distinct = "hash";
    normalized.sort = "no_skf";
    normalized.top_k = "no_skf";
    normalized.eq_join = "hash_gfur";
    normalized.semi_join = "hash_gfur";
    normalized.anti_join = "hash_gfur";
    apply_defaults(plan, normalized);
}

const rapidjson::Value& require_member(
    const rapidjson::Value& object,
    const char* name)
{
    if (!object.IsObject() || !object.HasMember(name)) {
        config_error(std::string("missing JSON member '") + name + "'");
    }
    return object[name];
}

std::string require_string(
    const rapidjson::Value& object,
    const char* name)
{
    const auto& value = require_member(object, name);
    if (!value.IsString()) {
        config_error(std::string("JSON member '") + name + "' must be a string");
    }
    return value.GetString();
}

int require_int(const rapidjson::Value& object, const char* name) {
    const auto& value = require_member(object, name);
    if (!value.IsInt()) {
        config_error(std::string("JSON member '") + name + "' must be an integer");
    }
    return value.GetInt();
}

bool require_bool(const rapidjson::Value& object, const char* name) {
    const auto& value = require_member(object, name);
    if (!value.IsBool()) {
        config_error(std::string("JSON member '") + name + "' must be boolean");
    }
    return value.GetBool();
}

StepOverride read_override(const rapidjson::Value& value) {
    return StepOverride{
        require_int(value, "step_index"),
        require_string(value, "step_kind"),
        require_string(value, "semantic_label"),
        require_string(value, "implementation")};
}

template <typename Writer>
void write_string_member(
    Writer& writer,
    const char* name,
    const std::string& value)
{
    writer.Key(name);
    writer.String(value.c_str());
}

template <typename Writer>
void write_override(Writer& writer, const StepOverride& value) {
    writer.StartObject();
    writer.Key("step_index");
    writer.Int(value.step_index);
    write_string_member(writer, "step_kind", value.step_kind);
    write_string_member(writer, "semantic_label", value.semantic_label);
    write_string_member(writer, "implementation", value.implementation);
    writer.EndObject();
}

} // namespace

CandidateAction initial_candidate_action(
    double incumbent_median_ms,
    double challenger_median_ms,
    double direct_commit_fraction)
{
    if (!(incumbent_median_ms > 0.0) ||
        !(challenger_median_ms >= 0.0) ||
        !(direct_commit_fraction >= 0.0)) {
        config_error("invalid candidate decision input");
    }
    if (challenger_median_ms >= incumbent_median_ms) {
        return CandidateAction::KeepIncumbent;
    }
    const double improvement =
        (incumbent_median_ms - challenger_median_ms) /
        incumbent_median_ms;
    return improvement >= direct_commit_fraction
        ? CandidateAction::CommitChallenger
        : CandidateAction::RunTieBreak;
}

bool commit_after_tie_break(
    double incumbent_median_ms,
    double challenger_median_ms,
    double tie_commit_fraction)
{
    if (!(incumbent_median_ms > 0.0) ||
        !(challenger_median_ms >= 0.0) ||
        !(tie_commit_fraction >= 0.0)) {
        config_error("invalid tie-break decision input");
    }
    const double improvement =
        (incumbent_median_ms - challenger_median_ms) /
        incumbent_median_ms;
    return improvement >= tie_commit_fraction;
}

const std::vector<std::string>& join_candidates() {
    static const std::vector<std::string> values{
        "hash_gfur",
        "sort_gfur",
        "sort_gftr",
        "partition_gfur",
        "partition_gftr"};
    return values;
}

std::string structure_fingerprint(const gamax::exec::ExecPlan& plan) {
    auto normalized = plan;
    normalize_hints(normalized);
    return benchmark::fingerprint_hex(benchmark::stable_fingerprint(
        gamax::exec::format_exec_plan(normalized)));
}

std::string physical_fingerprint(const gamax::exec::ExecPlan& plan) {
    return benchmark::fingerprint_hex(benchmark::stable_fingerprint(
        gamax::exec::format_exec_plan(plan)));
}

std::vector<JoinCoordinate> inner_join_coordinates(
    const gamax::exec::ExecPlan& local_plan)
{
    std::vector<JoinCoordinate> result;
    for (std::size_t index = 0; index < local_plan.steps.size(); ++index) {
        const auto& step = local_plan.steps[index];
        if (step.kind != StepKind::EqJoin) continue;
        result.push_back(JoinCoordinate{
            static_cast<int>(index),
            gamax::exec::step_kind_name(step.kind),
            gamax::exec::format_step_label(
                local_plan, static_cast<gamax::size_type>(index))});
    }
    return result;
}

Config make_stock_config(
    const Focus& focus,
    const gamax::exec::ExecPlan& local_plan)
{
    Config config;
    config.preset = "stock_distributed_default";
    config.focus = focus;
    config.defaults.filter = "fuse";
    config.defaults.project = "fuse";
    config.structure_fingerprint = structure_fingerprint(local_plan);
    config.search.completed = true;
    config.search.converged = true;
    return config;
}

Config make_synthesis_seed_config(
    const Focus& focus,
    const gamax::exec::ExecPlan& local_plan)
{
    Config config;
    config.preset = "synthesis_seed";
    config.focus = focus;
    config.structure_fingerprint = structure_fingerprint(local_plan);
    if (focus.query_id == 13) {
        config.fixed_overrides.push_back(make_override(
            local_plan, 6, StepKind::GroupBy, "partition_gftr"));
    }
    if (focus.query_id == 21) {
        config.fixed_overrides.push_back(make_override(
            local_plan, 6, StepKind::GroupBy, "partition_gftr"));
        config.fixed_overrides.push_back(make_override(
            local_plan, 12, StepKind::GroupBy, "partition_gftr"));
        config.fixed_overrides.push_back(make_override(
            local_plan, 4, StepKind::Distinct, "sort_skf"));
        config.fixed_overrides.push_back(make_override(
            local_plan, 10, StepKind::Distinct, "sort_skf"));
    }
    for (const auto& coordinate : inner_join_coordinates(local_plan)) {
        config.join_overrides.push_back(StepOverride{
            coordinate.step_index,
            coordinate.step_kind,
            coordinate.semantic_label,
            "hash_gfur"});
    }
    return config;
}

bool fallback_partition_groupbys_to_hash(
    Config& config,
    const std::string& reason)
{
    bool changed = false;
    for (auto& value : config.fixed_overrides) {
        if (value.step_kind != "GroupBy" ||
            (value.implementation != "partition_gftr" &&
             value.implementation != "partition_gfur")) {
            continue;
        }
        config.seed_fallbacks.push_back(SeedFallback{
            value.step_index,
            value.step_kind,
            value.implementation,
            "hash",
            reason});
        value.implementation = "hash";
        changed = true;
    }
    return changed;
}

void validate_focus(const Config& config, const Focus& actual) {
    if (config.schema_version != kSchemaVersion) {
        config_error(
            "unsupported schema version " +
            std::to_string(config.schema_version));
    }
    if (config.focus.query_id != actual.query_id) {
        config_error("query ID does not match the requested query");
    }
    if (config.focus.scale_factor != actual.scale_factor) {
        config_error("scale factor does not match the requested run");
    }
    if (config.focus.ranks != actual.ranks) {
        config_error("MPI rank count does not match the requested run");
    }
    if (config.focus.eq_join_backend != actual.eq_join_backend) {
        config_error("equijoin backend does not match the requested run");
    }
    if (!config.focus.gpu_model.empty() &&
        config.focus.gpu_model != actual.gpu_model) {
        config_error(
            "GPU model '" + actual.gpu_model + "' does not match config '" +
            config.focus.gpu_model + "'");
    }
}

void apply(gamax::exec::ExecPlan& local_plan, const Config& config) {
    const auto actual_structure = structure_fingerprint(local_plan);
    if (actual_structure != config.structure_fingerprint) {
        config_error(
            "stale structural fingerprint; actual " + actual_structure +
            ", expected " + config.structure_fingerprint);
    }
    apply_defaults(local_plan, config.defaults);
    for (const auto& value : config.fixed_overrides) {
        apply_override(local_plan, value);
    }
    for (const auto& value : config.join_overrides) {
        apply_override(local_plan, value);
    }
}

Config read_json(const std::string& path) {
    std::ifstream input(path);
    if (!input) config_error("cannot open '" + path + "'");
    rapidjson::IStreamWrapper stream(input);
    rapidjson::Document document;
    document.ParseStream(stream);
    if (document.HasParseError()) {
        config_error(
            "invalid JSON in '" + path + "': " +
            rapidjson::GetParseError_En(document.GetParseError()));
    }

    Config config;
    config.schema_version = require_int(document, "schema_version");
    config.preset = require_string(document, "preset");
    const auto& focus = require_member(document, "focus");
    config.focus = Focus{
        require_int(focus, "query_id"),
        require_int(focus, "scale_factor"),
        require_int(focus, "ranks"),
        require_string(focus, "gpu_model"),
        require_string(focus, "eq_join_backend")};
    const auto& defaults = require_member(document, "defaults");
    config.defaults = Defaults{
        require_string(defaults, "filter"),
        require_string(defaults, "project"),
        require_string(defaults, "group_by"),
        require_string(defaults, "distinct"),
        require_string(defaults, "sort"),
        require_string(defaults, "top_k"),
        require_string(defaults, "eq_join"),
        require_string(defaults, "semi_join"),
        require_string(defaults, "anti_join")};
    config.structure_fingerprint =
        require_string(document, "structure_fingerprint");

    auto read_overrides = [&](const char* name, auto& output) {
        const auto& values = require_member(document, name);
        if (!values.IsArray()) {
            config_error(std::string("JSON member '") + name + "' must be an array");
        }
        for (const auto& value : values.GetArray()) {
            output.push_back(read_override(value));
        }
    };
    read_overrides("fixed_overrides", config.fixed_overrides);
    read_overrides("join_overrides", config.join_overrides);

    if (document.HasMember("seed_fallbacks")) {
        const auto& fallbacks = document["seed_fallbacks"];
        if (!fallbacks.IsArray()) {
            config_error("JSON member 'seed_fallbacks' must be an array");
        }
        for (const auto& value : fallbacks.GetArray()) {
            config.seed_fallbacks.push_back(SeedFallback{
                require_int(value, "step_index"),
                require_string(value, "step_kind"),
                require_string(value, "from_implementation"),
                require_string(value, "to_implementation"),
                require_string(value, "reason")});
        }
    }

    const auto& search = require_member(document, "search");
    const auto& seed = require_member(search, "seed");
    if (!seed.IsUint64()) config_error("search.seed must be uint64");
    config.search.seed = seed.GetUint64();
    config.search.pass = require_int(search, "pass");
    config.search.coordinate_position =
        require_int(search, "coordinate_position");
    config.search.max_passes = require_int(search, "max_passes");
    config.search.completed = require_bool(search, "completed");
    config.search.converged = require_bool(search, "converged");
    if (search.HasMember("active_step_index") &&
        !search["active_step_index"].IsNull()) {
        if (!search["active_step_index"].IsInt()) {
            config_error("search.active_step_index must be integer or null");
        }
        config.search.active_step_index =
            search["active_step_index"].GetInt();
    }
    config.search.active_implementation =
        require_string(search, "active_implementation");
    const auto& order = require_member(search, "coordinate_order");
    if (!order.IsArray()) config_error("search.coordinate_order must be an array");
    for (const auto& value : order.GetArray()) {
        if (!value.IsInt()) config_error("coordinate_order values must be integers");
        config.search.coordinate_order.push_back(value.GetInt());
    }
    const auto& records = require_member(search, "records");
    if (!records.IsArray()) config_error("search.records must be an array");
    for (const auto& value : records.GetArray()) {
        config.search.records.push_back(CandidateRecord{
            require_int(value, "pass"),
            require_int(value, "coordinate_position"),
            require_int(value, "step_index"),
            require_string(value, "implementation"),
            require_string(value, "status"),
            require_member(value, "median_ms").GetDouble(),
            require_int(value, "samples"),
            require_string(value, "detail")});
    }
    return config;
}

void write_json_atomic(const Config& config, const std::string& path) {
    const std::filesystem::path output_path(path);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    const auto temporary = output_path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) config_error("cannot write '" + temporary + "'");
    rapidjson::OStreamWrapper stream(output);
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(stream);
    writer.SetIndent(' ', 2);
    writer.StartObject();
    writer.Key("schema_version");
    writer.Int(config.schema_version);
    write_string_member(writer, "preset", config.preset);
    writer.Key("focus");
    writer.StartObject();
    writer.Key("query_id");
    writer.Int(config.focus.query_id);
    writer.Key("scale_factor");
    writer.Int(config.focus.scale_factor);
    writer.Key("ranks");
    writer.Int(config.focus.ranks);
    write_string_member(writer, "gpu_model", config.focus.gpu_model);
    write_string_member(
        writer, "eq_join_backend", config.focus.eq_join_backend);
    writer.EndObject();
    writer.Key("defaults");
    writer.StartObject();
    write_string_member(writer, "filter", config.defaults.filter);
    write_string_member(writer, "project", config.defaults.project);
    write_string_member(writer, "group_by", config.defaults.group_by);
    write_string_member(writer, "distinct", config.defaults.distinct);
    write_string_member(writer, "sort", config.defaults.sort);
    write_string_member(writer, "top_k", config.defaults.top_k);
    write_string_member(writer, "eq_join", config.defaults.eq_join);
    write_string_member(writer, "semi_join", config.defaults.semi_join);
    write_string_member(writer, "anti_join", config.defaults.anti_join);
    writer.EndObject();
    write_string_member(
        writer, "structure_fingerprint", config.structure_fingerprint);
    writer.Key("fixed_overrides");
    writer.StartArray();
    for (const auto& value : config.fixed_overrides) {
        write_override(writer, value);
    }
    writer.EndArray();
    writer.Key("join_overrides");
    writer.StartArray();
    for (const auto& value : config.join_overrides) {
        write_override(writer, value);
    }
    writer.EndArray();
    writer.Key("seed_fallbacks");
    writer.StartArray();
    for (const auto& value : config.seed_fallbacks) {
        writer.StartObject();
        writer.Key("step_index");
        writer.Int(value.step_index);
        write_string_member(writer, "step_kind", value.step_kind);
        write_string_member(
            writer, "from_implementation", value.from_implementation);
        write_string_member(
            writer, "to_implementation", value.to_implementation);
        write_string_member(writer, "reason", value.reason);
        writer.EndObject();
    }
    writer.EndArray();
    writer.Key("search");
    writer.StartObject();
    writer.Key("seed");
    writer.Uint64(config.search.seed);
    writer.Key("pass");
    writer.Int(config.search.pass);
    writer.Key("coordinate_position");
    writer.Int(config.search.coordinate_position);
    writer.Key("max_passes");
    writer.Int(config.search.max_passes);
    writer.Key("completed");
    writer.Bool(config.search.completed);
    writer.Key("converged");
    writer.Bool(config.search.converged);
    writer.Key("active_step_index");
    if (config.search.active_step_index) {
        writer.Int(*config.search.active_step_index);
    } else {
        writer.Null();
    }
    write_string_member(
        writer, "active_implementation",
        config.search.active_implementation);
    writer.Key("coordinate_order");
    writer.StartArray();
    for (const int value : config.search.coordinate_order) writer.Int(value);
    writer.EndArray();
    writer.Key("records");
    writer.StartArray();
    for (const auto& value : config.search.records) {
        writer.StartObject();
        writer.Key("pass");
        writer.Int(value.pass);
        writer.Key("coordinate_position");
        writer.Int(value.coordinate_position);
        writer.Key("step_index");
        writer.Int(value.step_index);
        write_string_member(
            writer, "implementation", value.implementation);
        write_string_member(writer, "status", value.status);
        writer.Key("median_ms");
        writer.Double(value.median_ms);
        writer.Key("samples");
        writer.Int(value.samples);
        write_string_member(writer, "detail", value.detail);
        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();
    writer.EndObject();
    output.flush();
    if (!output) config_error("failed while writing '" + temporary + "'");
    output.close();
    std::filesystem::rename(temporary, output_path);
}

} // namespace gamax::tpch::physical_plan
