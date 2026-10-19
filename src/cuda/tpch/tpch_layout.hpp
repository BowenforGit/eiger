// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tpch/source_policy.hpp"

namespace gamax::tpch {

enum class FileShardStrategy {
    RoundRobin,
    ContiguousRange,
};

const char* file_shard_strategy_name(FileShardStrategy strategy) noexcept;
FileShardStrategy parse_file_shard_strategy(const std::string& value);
std::string sha256_file_hex(const std::string& path);

std::vector<std::size_t> assigned_part_indices(
    std::size_t part_count,
    int shard_rank,
    int shard_count,
    FileShardStrategy strategy);

struct ResolvedTpchFiles {
    std::vector<std::string> paths;
    std::string schema_path;
    bool file_partitioned {false};
    std::string dataset_identity;
};

ResolvedTpchFiles resolve_tpch_files(
    const std::string& data_dir,
    const std::string& table_name,
    int expected_scale_factor,
    int shard_rank,
    int shard_count,
    SourceReadPolicy read_policy,
    FileShardStrategy strategy,
    bool allow_legacy_multifile,
    bool validate_complete_directory = false);

void preflight_tpch_dataset(
    const std::string& data_dir,
    int expected_scale_factor,
    bool allow_legacy_multifile = false);

} // namespace gamax::tpch
