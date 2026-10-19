// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>

#include "tpch/source_policy.hpp"
#include "tpch/tpch.hpp"

namespace {

void require_ok(const arrow::Status& status) {
    ASSERT_TRUE(status.ok()) << status.ToString();
}

void write_int64_parquet(
    const std::filesystem::path& path,
    const std::string& column_name,
    const std::vector<int64_t>& values,
    int64_t row_group_size)
{
    arrow::Int64Builder builder;
    require_ok(builder.AppendValues(values));
    std::shared_ptr<arrow::Array> array;
    require_ok(builder.Finish(&array));
    auto table = arrow::Table::Make(
        arrow::schema({arrow::field(column_name, arrow::int64())}), {array});
    auto output = arrow::io::FileOutputStream::Open(path.string());
    ASSERT_TRUE(output.ok()) << output.status().ToString();
    require_ok(parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), *output, row_group_size));
    require_ok((*output)->Close());
}

class TpchReaderPolicyTest : public ::testing::Test {
protected:
    void SetUp() override {
        char pattern[] = "/tmp/gamax_tpch_reader_policy.XXXXXX";
        const char* created = mkdtemp(pattern);
        ASSERT_NE(created, nullptr);
        root_ = created;
        sf_ = root_ / "sf1";
        std::filesystem::create_directories(sf_ / "nation");

        write_int64_parquet(
            sf_ / "region.parquet", "r_regionkey",
            {0, 1, 2, 3, 4, 5}, 2);
        write_int64_parquet(
            sf_ / "nation" / "nation.0.parquet", "n_nationkey",
            {0, 1, 2}, 3);
        write_int64_parquet(
            sf_ / "nation" / "nation.1.parquet", "n_nationkey",
            {3, 4}, 2);
        write_int64_parquet(
            sf_ / "nation" / "nation.2.parquet", "n_nationkey",
            {5, 6, 7, 8}, 4);

        const char* old = std::getenv("TPCH_DATA_PATH");
        if (old) old_data_path_ = old;
        setenv("TPCH_DATA_PATH", root_.c_str(), 1);
    }

    void TearDown() override {
        if (old_data_path_.empty()) {
            unsetenv("TPCH_DATA_PATH");
        } else {
            setenv("TPCH_DATA_PATH", old_data_path_.c_str(), 1);
        }
        std::filesystem::remove_all(root_);
    }

    gamax::tpch::tpch_params params(
        gamax::tpch::SourceReadPoliciesPtr policies) const
    {
        gamax::tpch::tpch_params result(1);
        result.set_row_group_shard(1, 2);
        result.allow_legacy_multifile = true;
        result.set_source_read_policies(std::move(policies));
        return result;
    }

    std::filesystem::path root_;
    std::filesystem::path sf_;
    std::string old_data_path_;
};

TEST_F(TpchReaderPolicyTest, DistributedDefaultsReplicateOnlySmallDimensions) {
    const auto policies =
        gamax::tpch::make_distributed_tpch_source_read_policies();
    EXPECT_EQ(
        policies->for_table("region"),
        gamax::tpch::SourceReadPolicy::Replicated);
    EXPECT_EQ(
        policies->for_table("nation"),
        gamax::tpch::SourceReadPolicy::Replicated);
    EXPECT_EQ(
        policies->for_table("lineitem"),
        gamax::tpch::SourceReadPolicy::Sharded);
}

TEST_F(TpchReaderPolicyTest, ReplicatedSingleFileReadsEveryRowGroup) {
    gamax::tpch::TpchTableReader replicated(
        params(gamax::tpch::make_distributed_tpch_source_read_policies()));
    auto full = replicated.load("region", {"r_regionkey"});
    EXPECT_EQ(full->num_rows(), 6);

    gamax::tpch::TpchTableReader sharded(
        params(gamax::tpch::make_sharded_source_read_policies()));
    auto subset = sharded.load("region", {"r_regionkey"});
    EXPECT_EQ(subset->num_rows(), 2);
}

TEST_F(TpchReaderPolicyTest, ReplicatedMultiFileReadsEveryPart) {
    gamax::tpch::TpchTableReader replicated(
        params(gamax::tpch::make_distributed_tpch_source_read_policies()));
    auto full = replicated.load("nation", {"n_nationkey"});
    EXPECT_EQ(full->num_rows(), 9);

    gamax::tpch::TpchTableReader sharded(
        params(gamax::tpch::make_sharded_source_read_policies()));
    auto subset = sharded.load("nation", {"n_nationkey"});
    EXPECT_EQ(subset->num_rows(), 2);
}

TEST_F(TpchReaderPolicyTest, MultiFileSchemaMismatchIsRejected) {
    std::filesystem::create_directories(sf_ / "orders");
    write_int64_parquet(
        sf_ / "orders" / "orders.0.parquet", "o_orderkey", {1, 2}, 2);
    write_int64_parquet(
        sf_ / "orders" / "orders.1.parquet", "wrong_column", {3, 4}, 2);

    gamax::tpch::tpch_params reader_params(1);
    reader_params.allow_legacy_multifile = true;
    gamax::tpch::TpchTableReader reader(reader_params);
    EXPECT_THROW(reader.load("orders", {"o_orderkey"}), std::runtime_error);
}

TEST(TpchFileAssignmentTest, RoundRobinSupportsNvl72Counts) {
    using gamax::tpch::FileShardStrategy;
    const auto rank_zero_36 = gamax::tpch::assigned_part_indices(
        144, 0, 36, FileShardStrategy::RoundRobin);
    EXPECT_EQ(
        rank_zero_36,
        (std::vector<std::size_t>{0, 36, 72, 108}));

    const auto rank_71 = gamax::tpch::assigned_part_indices(
        144, 71, 72, FileShardStrategy::RoundRobin);
    EXPECT_EQ(rank_71, (std::vector<std::size_t>{71, 143}));
}

TEST(TpchFileAssignmentTest, ContiguousRangeSupportsNvl72Counts) {
    using gamax::tpch::FileShardStrategy;
    const auto rank_zero_36 = gamax::tpch::assigned_part_indices(
        144, 0, 36, FileShardStrategy::ContiguousRange);
    EXPECT_EQ(
        rank_zero_36,
        (std::vector<std::size_t>{0, 1, 2, 3}));

    const auto rank_71 = gamax::tpch::assigned_part_indices(
        144, 71, 72, FileShardStrategy::ContiguousRange);
    EXPECT_EQ(rank_71, (std::vector<std::size_t>{142, 143}));
}

TEST(TpchFileAssignmentTest, StrategiesCoverNonDivisiblePartCountsExactlyOnce) {
    using gamax::tpch::FileShardStrategy;
    for (const auto strategy : {
             FileShardStrategy::RoundRobin,
             FileShardStrategy::ContiguousRange}) {
        std::vector<int> visits(11, 0);
        std::size_t minimum = 11;
        std::size_t maximum = 0;
        for (int rank = 0; rank < 4; ++rank) {
            const auto assigned = gamax::tpch::assigned_part_indices(
                visits.size(), rank, 4, strategy);
            minimum = std::min(minimum, assigned.size());
            maximum = std::max(maximum, assigned.size());
            for (const auto index : assigned) {
                ASSERT_LT(index, visits.size());
                ++visits[index];
            }
        }
        EXPECT_LE(maximum - minimum, 1);
        for (const int count : visits) EXPECT_EQ(count, 1);
    }
}

TEST(TpchFileAssignmentTest, MoreRanksThanPartsProduceTypedEmptyAssignments) {
    using gamax::tpch::FileShardStrategy;
    EXPECT_TRUE(gamax::tpch::assigned_part_indices(
        2, 3, 4, FileShardStrategy::RoundRobin).empty());
    EXPECT_TRUE(gamax::tpch::assigned_part_indices(
        2, 0, 4, FileShardStrategy::ContiguousRange).empty());
}

TEST(TpchManifestLayoutTest, UsesManifestNumericOrderForBothStrategies) {
    char pattern[] = "/tmp/gamax_tpch_manifest_layout.XXXXXX";
    const char* created = mkdtemp(pattern);
    ASSERT_NE(created, nullptr);
    const std::filesystem::path root(created);
    const auto table_dir = root / "lineitem";
    std::filesystem::create_directories(table_dir);

    std::vector<std::filesystem::path> paths;
    for (const int part : {1, 2, 10, 11}) {
        auto path = table_dir /
            ("lineitem." + std::to_string(part) + ".parquet");
        write_int64_parquet(path, "l_orderkey", {part}, 1);
        paths.push_back(std::move(path));
    }

    const std::string zeros(64, '0');
    const std::string schema_fingerprint =
        "f284397aa53d9beb4d278cd37cc6fa01ad2e46a2db3b0c98b8a6df8d7d234c31";
    std::uintmax_t total_bytes = 0;
    for (const auto& path : paths) {
        total_bytes += std::filesystem::file_size(path);
    }
    const auto manifest_path = root / "manifest.json";
    {
        std::ofstream manifest(manifest_path);
        manifest
            << R"({"format_version":1,"scale_factor":1000,"tables":{"lineitem":{)"
            << R"("layout":"partitioned","read_policy":"sharded",)"
            << R"("schema":[{"name":"l_orderkey","nullable":true,"type":"int64"}],)"
            << R"("schema_fingerprint":")" << schema_fingerprint
            << R"(","rows":4,"bytes":)" << total_bytes
            << R"(,"part_count":4,"parts":[)";
        for (std::size_t index = 0; index < paths.size(); ++index) {
            if (index != 0) manifest << ",";
            manifest
                << R"({"id":)" << (index + 1)
                << R"(,"path":"lineitem/)" << paths[index].filename().string()
                << R"(","rows":1,"bytes":)"
                << std::filesystem::file_size(paths[index])
                << R"(,"row_groups":1,"sha256":")"
                << zeros << R"("})";
        }
        manifest << "]}}}";
    }
    {
        std::ofstream success(root / "_SUCCESS");
        success << "manifest_sha256="
                << gamax::tpch::sha256_file_hex(manifest_path.string())
                << "\n";
    }

    const auto round_robin = gamax::tpch::resolve_tpch_files(
        root.string(), "lineitem", 1000, 0, 2,
        gamax::tpch::SourceReadPolicy::Sharded,
        gamax::tpch::FileShardStrategy::RoundRobin, false);
    ASSERT_EQ(round_robin.paths.size(), 2);
    EXPECT_EQ(round_robin.paths[0], paths[0].string());
    EXPECT_EQ(round_robin.paths[1], paths[2].string());

    const auto range = gamax::tpch::resolve_tpch_files(
        root.string(), "lineitem", 1000, 0, 2,
        gamax::tpch::SourceReadPolicy::Sharded,
        gamax::tpch::FileShardStrategy::ContiguousRange, false);
    ASSERT_EQ(range.paths.size(), 2);
    EXPECT_EQ(range.paths[0], paths[0].string());
    EXPECT_EQ(range.paths[1], paths[1].string());

    // A host-local cache only contains the parts assigned to ranks on that
    // host.  The global manifest may therefore name files that are present on
    // other hosts, and normal reads must validate only the locally assigned
    // paths.
    std::filesystem::remove(paths[1]);
    std::filesystem::remove(paths[3]);
    const auto partial_host = gamax::tpch::resolve_tpch_files(
        root.string(), "lineitem", 1000, 0, 2,
        gamax::tpch::SourceReadPolicy::Sharded,
        gamax::tpch::FileShardStrategy::RoundRobin, false);
    ASSERT_EQ(partial_host.paths.size(), 2);
    EXPECT_EQ(partial_host.paths[0], paths[0].string());
    EXPECT_EQ(partial_host.paths[1], paths[2].string());

    std::filesystem::remove_all(root);
}

TEST(TpchManifestLayoutTest, RejectsSchemaDeclarationFingerprintMismatch) {
    char pattern[] = "/tmp/gamax_tpch_manifest_schema_decl.XXXXXX";
    const char* created = mkdtemp(pattern);
    ASSERT_NE(created, nullptr);
    const std::filesystem::path root(created);
    const auto table_dir = root / "lineitem";
    std::filesystem::create_directories(table_dir);
    const auto path = table_dir / "lineitem.1.parquet";
    write_int64_parquet(path, "l_orderkey", {1}, 1);

    const std::string zeros(64, '0');
    const auto bytes = std::filesystem::file_size(path);
    const auto manifest_path = root / "manifest.json";
    {
        std::ofstream manifest(manifest_path);
        manifest
            << R"({"format_version":1,"scale_factor":1,"tables":{"lineitem":{)"
            << R"("layout":"partitioned","read_policy":"sharded",)"
            << R"("schema":[{"name":"l_orderkey","nullable":true,"type":"int64"}],)"
            << R"("schema_fingerprint":")" << zeros
            << R"(","rows":1,"bytes":)" << bytes
            << R"(,"part_count":1,"parts":[{"id":1,)"
            << R"("path":"lineitem/lineitem.1.parquet","rows":1,"bytes":)"
            << bytes << R"(,"row_groups":1,"sha256":")"
            << zeros << R"("}]}}})";
    }
    {
        std::ofstream success(root / "_SUCCESS");
        success << "manifest_sha256="
                << gamax::tpch::sha256_file_hex(manifest_path.string())
                << "\n";
    }

    EXPECT_THROW(
        gamax::tpch::resolve_tpch_files(
            root.string(), "lineitem", 1, 0, 1,
            gamax::tpch::SourceReadPolicy::Sharded,
            gamax::tpch::FileShardStrategy::RoundRobin, false),
        std::runtime_error);
    std::filesystem::remove_all(root);
}

TEST(TpchManifestLayoutTest, RejectsParquetSchemaDifferentFromManifest) {
    char pattern[] = "/tmp/gamax_tpch_manifest_schema_file.XXXXXX";
    const char* created = mkdtemp(pattern);
    ASSERT_NE(created, nullptr);
    const std::filesystem::path root(created);
    const auto table_dir = root / "lineitem";
    std::filesystem::create_directories(table_dir);
    const auto path = table_dir / "lineitem.1.parquet";
    write_int64_parquet(path, "wrong_column", {1}, 1);

    const std::string zeros(64, '0');
    const std::string declared_fingerprint =
        "f284397aa53d9beb4d278cd37cc6fa01ad2e46a2db3b0c98b8a6df8d7d234c31";
    const auto bytes = std::filesystem::file_size(path);
    const auto manifest_path = root / "manifest.json";
    {
        std::ofstream manifest(manifest_path);
        manifest
            << R"({"format_version":1,"scale_factor":1,"tables":{"lineitem":{)"
            << R"("layout":"partitioned","read_policy":"sharded",)"
            << R"("schema":[{"name":"l_orderkey","nullable":true,"type":"int64"}],)"
            << R"("schema_fingerprint":")" << declared_fingerprint
            << R"(","rows":1,"bytes":)" << bytes
            << R"(,"part_count":1,"parts":[{"id":1,)"
            << R"("path":"lineitem/lineitem.1.parquet","rows":1,"bytes":)"
            << bytes << R"(,"row_groups":1,"sha256":")"
            << zeros << R"("}]}}})";
    }
    {
        std::ofstream success(root / "_SUCCESS");
        success << "manifest_sha256="
                << gamax::tpch::sha256_file_hex(manifest_path.string())
                << "\n";
    }

    EXPECT_THROW(
        gamax::tpch::resolve_tpch_files(
            root.string(), "lineitem", 1, 0, 1,
            gamax::tpch::SourceReadPolicy::Sharded,
            gamax::tpch::FileShardStrategy::RoundRobin, false),
        std::runtime_error);
    std::filesystem::remove_all(root);
}

TEST(TpchManifestLayoutTest, RejectsUnmanifestedMultiFileByDefault) {
    char pattern[] = "/tmp/gamax_tpch_legacy_layout.XXXXXX";
    const char* created = mkdtemp(pattern);
    ASSERT_NE(created, nullptr);
    const std::filesystem::path root(created);
    std::filesystem::create_directories(root / "orders");
    std::ofstream(root / "orders" / "orders.1.parquet") << "not parquet";

    EXPECT_THROW(
        gamax::tpch::resolve_tpch_files(
            root.string(), "orders", 1, 0, 1,
            gamax::tpch::SourceReadPolicy::Sharded,
            gamax::tpch::FileShardStrategy::RoundRobin, false),
        std::runtime_error);
    std::filesystem::remove_all(root);
}

} // namespace
