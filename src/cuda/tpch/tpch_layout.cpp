// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "tpch/tpch_layout.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <openssl/evp.h>
#include <parquet/arrow/reader.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace gamax::tpch {
namespace {

namespace fs = std::filesystem;

std::string read_text_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open '" + path.string() + "'");
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("Failed to read '" + path.string() + "'");
    }
    return contents.str();
}

std::string trim_ascii(std::string value) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!value.empty() && is_space(value.front())) value.erase(value.begin());
    while (!value.empty() && is_space(value.back())) value.pop_back();
    return value;
}

std::string sha256_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open '" + path.string() +
                                 "' for SHA-256");
    }

    EVP_MD_CTX* raw = EVP_MD_CTX_new();
    if (raw == nullptr) {
        throw std::runtime_error("EVP_MD_CTX_new failed");
    }
    struct ContextGuard {
        EVP_MD_CTX* value;
        ~ContextGuard() { EVP_MD_CTX_free(value); }
    } guard {raw};

    if (EVP_DigestInit_ex(raw, EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("EVP_DigestInit_ex(SHA-256) failed");
    }
    std::array<char, 1 << 16> buffer {};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0 &&
            EVP_DigestUpdate(raw, buffer.data(), static_cast<std::size_t>(count)) != 1) {
            throw std::runtime_error("EVP_DigestUpdate(SHA-256) failed");
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("Failed while hashing '" + path.string() + "'");
    }

    std::array<unsigned char, EVP_MAX_MD_SIZE> digest {};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(raw, digest.data(), &size) != 1) {
        throw std::runtime_error("EVP_DigestFinal_ex(SHA-256) failed");
    }
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < size; ++i) {
        result << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return result.str();
}

std::string sha256_text(std::string_view text) {
    EVP_MD_CTX* raw = EVP_MD_CTX_new();
    if (raw == nullptr) {
        throw std::runtime_error("EVP_MD_CTX_new failed");
    }
    struct ContextGuard {
        EVP_MD_CTX* value;
        ~ContextGuard() { EVP_MD_CTX_free(value); }
    } guard {raw};

    if (EVP_DigestInit_ex(raw, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(raw, text.data(), text.size()) != 1) {
        throw std::runtime_error("Failed to hash TPC-H schema description");
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest {};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(raw, digest.data(), &size) != 1) {
        throw std::runtime_error("Failed to finish TPC-H schema SHA-256");
    }
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < size; ++i) {
        result << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return result.str();
}

const rapidjson::Value& required_member(
    const rapidjson::Value& object,
    const char* name,
    rapidjson::Type type,
    std::string_view context)
{
    if (!object.IsObject() || !object.HasMember(name) ||
        object[name].GetType() != type) {
        throw std::runtime_error(
            "TPC-H manifest " + std::string(context) + " requires '" +
            name + "' with the expected type");
    }
    return object[name];
}

std::uint64_t required_uint64(
    const rapidjson::Value& object,
    const char* name,
    std::string_view context)
{
    if (!object.IsObject() || !object.HasMember(name) ||
        !object[name].IsUint64()) {
        throw std::runtime_error(
            "TPC-H manifest " + std::string(context) +
            " requires non-negative integer '" + name + "'");
    }
    return object[name].GetUint64();
}

std::string required_string(
    const rapidjson::Value& object,
    const char* name,
    std::string_view context)
{
    const auto& value =
        required_member(object, name, rapidjson::kStringType, context);
    return value.GetString();
}

bool required_bool(
    const rapidjson::Value& object,
    const char* name,
    std::string_view context)
{
    if (!object.IsObject() || !object.HasMember(name) ||
        !object[name].IsBool()) {
        throw std::runtime_error(
            "TPC-H manifest " + std::string(context) + " requires boolean '" +
            name + "'");
    }
    return object[name].GetBool();
}

bool is_lowercase_sha256(std::string_view value) {
    return value.size() == 64 &&
        std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
}

void require_sha256(
    const rapidjson::Value& object,
    std::string_view context)
{
    const std::string value = required_string(object, "sha256", context);
    if (!is_lowercase_sha256(value)) {
        throw std::runtime_error(
            "TPC-H manifest " + std::string(context) +
            " sha256 must be 64 lowercase hexadecimal characters");
    }
}

std::string manifest_schema_fingerprint(
    const rapidjson::Value& schema,
    std::string_view context)
{
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartArray();
    for (rapidjson::SizeType index = 0; index < schema.Size(); ++index) {
        const auto& field = schema[index];
        const std::string field_context =
            std::string(context) + " schema field " +
            std::to_string(index + 1);
        if (!field.IsObject() || field.MemberCount() != 3) {
            throw std::runtime_error(
                "TPC-H manifest " + field_context +
                " must contain exactly name, nullable, and type");
        }
        const std::string name =
            required_string(field, "name", field_context);
        const bool nullable =
            required_bool(field, "nullable", field_context);
        const std::string type =
            required_string(field, "type", field_context);
        writer.StartObject();
        // This is the same sorted-key, compact JSON representation used by
        // script/finalize_tpch_manifest.py.
        writer.Key("name");
        writer.String(name.c_str());
        writer.Key("nullable");
        writer.Bool(nullable);
        writer.Key("type");
        writer.String(type.c_str());
        writer.EndObject();
    }
    writer.EndArray();
    return sha256_text(
        std::string_view(buffer.GetString(), buffer.GetSize()));
}

std::string arrow_schema_fingerprint(const arrow::Schema& schema) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartArray();
    for (const auto& field : schema.fields()) {
        writer.StartObject();
        writer.Key("name");
        writer.String(field->name().c_str());
        writer.Key("nullable");
        writer.Bool(field->nullable());
        writer.Key("type");
        const std::string type = field->type()->ToString();
        writer.String(type.c_str());
        writer.EndObject();
    }
    writer.EndArray();
    return sha256_text(
        std::string_view(buffer.GetString(), buffer.GetSize()));
}

fs::path safe_manifest_path(
    const fs::path& data_dir,
    const rapidjson::Value& entry,
    std::string_view context)
{
    const auto& path_value =
        required_member(entry, "path", rapidjson::kStringType, context);
    fs::path relative(path_value.GetString());
    if (relative.empty() || relative.is_absolute()) {
        throw std::runtime_error(
            "TPC-H manifest " + std::string(context) +
            " path must be non-empty and relative");
    }
    relative = relative.lexically_normal();
    if (relative.empty() || *relative.begin() == "..") {
        throw std::runtime_error(
            "TPC-H manifest " + std::string(context) +
            " path escapes the dataset directory");
    }
    return (data_dir / relative).lexically_normal();
}

void verify_declared_file(
    const fs::path& path,
    std::uint64_t declared_bytes,
    std::string_view context)
{
    std::error_code error;
    if (!fs::is_regular_file(path, error)) {
        throw std::runtime_error(
            "TPC-H manifest " + std::string(context) +
            " file is missing or not regular: " + path.string());
    }
    const auto actual = fs::file_size(path, error);
    if (error || actual != declared_bytes) {
        throw std::runtime_error(
            "TPC-H manifest " + std::string(context) +
            " byte-size mismatch for '" + path.string() + "': expected " +
            std::to_string(declared_bytes) + ", got " +
            (error ? error.message() : std::to_string(actual)));
    }
}

std::shared_ptr<arrow::Schema> read_arrow_schema(const fs::path& path) {
    auto input = arrow::io::ReadableFile::Open(path.string());
    if (!input.ok()) {
        throw std::runtime_error(
            "TPC-H preflight failed to open '" + path.string() + "': " +
            input.status().ToString());
    }
    parquet::arrow::FileReaderBuilder builder;
    auto status = builder.Open(*input);
    if (!status.ok()) {
        throw std::runtime_error(
            "TPC-H preflight failed to parse Parquet file '" + path.string() +
            "': " + status.ToString());
    }
    std::unique_ptr<parquet::arrow::FileReader> reader;
    status = builder.Build(&reader);
    if (!status.ok()) {
        throw std::runtime_error(
            "TPC-H preflight failed to build Parquet reader for '" +
            path.string() + "': " + status.ToString());
    }
    std::shared_ptr<arrow::Schema> schema;
    status = reader->GetSchema(&schema);
    if (!status.ok()) {
        throw std::runtime_error(
            "TPC-H preflight failed to read schema for '" + path.string() +
            "': " + status.ToString());
    }
    return schema;
}

void verify_declared_schema(
    const fs::path& path,
    std::string_view declared_fingerprint,
    std::string_view table_name)
{
    const auto schema = read_arrow_schema(path);
    const std::string actual_fingerprint =
        arrow_schema_fingerprint(*schema);
    if (actual_fingerprint != declared_fingerprint) {
        throw std::runtime_error(
            "TPC-H manifest schema mismatch for table '" +
            std::string(table_name) + "' file '" + path.string() +
            "': declared fingerprint " + std::string(declared_fingerprint) +
            ", actual " + actual_fingerprint);
    }
}

ResolvedTpchFiles resolve_manifest_table(
    const fs::path& data_dir,
    const std::string& table_name,
    int expected_scale_factor,
    int shard_rank,
    int shard_count,
    SourceReadPolicy read_policy,
    FileShardStrategy strategy,
    bool validate_complete_directory)
{
    const fs::path manifest_path = data_dir / "manifest.json";
    const fs::path success_path = data_dir / "_SUCCESS";
    if (!fs::is_regular_file(success_path)) {
        throw std::runtime_error(
            "TPC-H manifest dataset is incomplete: missing " +
            success_path.string());
    }

    const std::string manifest_hash = sha256_file(manifest_path);
    const std::string expected_marker =
        "manifest_sha256=" + manifest_hash;
    if (trim_ascii(read_text_file(success_path)) != expected_marker) {
        throw std::runtime_error(
            "TPC-H dataset completion marker does not match manifest SHA-256: " +
            success_path.string());
    }

    const std::string manifest_text = read_text_file(manifest_path);
    rapidjson::Document manifest;
    manifest.Parse(manifest_text.data(), manifest_text.size());
    if (manifest.HasParseError()) {
        throw std::runtime_error(
            "Failed to parse TPC-H manifest '" + manifest_path.string() +
            "' at byte " + std::to_string(manifest.GetErrorOffset()) + ": " +
            rapidjson::GetParseError_En(manifest.GetParseError()));
    }
    if (!manifest.IsObject()) {
        throw std::runtime_error("TPC-H manifest root must be an object");
    }
    const auto format_version =
        required_uint64(manifest, "format_version", "root");
    if (format_version != 1) {
        throw std::runtime_error(
            "Unsupported TPC-H manifest format_version " +
            std::to_string(format_version));
    }
    const auto scale_factor =
        required_uint64(manifest, "scale_factor", "root");
    if (scale_factor != static_cast<std::uint64_t>(expected_scale_factor)) {
        throw std::runtime_error(
            "TPC-H manifest scale_factor mismatch: expected " +
            std::to_string(expected_scale_factor) + ", got " +
            std::to_string(scale_factor));
    }

    const auto& tables =
        required_member(manifest, "tables", rapidjson::kObjectType, "root");
    if (!tables.HasMember(table_name.c_str()) ||
        !tables[table_name.c_str()].IsObject()) {
        throw std::runtime_error(
            "TPC-H manifest has no table entry for '" + table_name + "'");
    }
    const auto& table = tables[table_name.c_str()];
    const std::string expected_policy =
        table_name == "region" || table_name == "nation"
        ? "replicated" : "sharded";
    const std::string declared_policy =
        required_string(table, "read_policy", "table '" + table_name + "'");
    if (declared_policy != expected_policy) {
        throw std::runtime_error(
            "TPC-H manifest table '" + table_name +
            "' read_policy mismatch: expected " + expected_policy +
            ", got " + declared_policy);
    }
    const auto& declared_schema = required_member(
        table, "schema", rapidjson::kArrayType, "table '" + table_name + "'");
    const std::string schema_fingerprint = required_string(
        table, "schema_fingerprint", "table '" + table_name + "'");
    if (!is_lowercase_sha256(schema_fingerprint)) {
        throw std::runtime_error(
            "TPC-H manifest table '" + table_name +
            "' has invalid schema_fingerprint");
    }
    const std::string described_fingerprint = manifest_schema_fingerprint(
        declared_schema, "table '" + table_name + "'");
    if (described_fingerprint != schema_fingerprint) {
        throw std::runtime_error(
            "TPC-H manifest table '" + table_name +
            "' schema_fingerprint does not match its schema declaration");
    }
    const auto declared_table_rows =
        required_uint64(table, "rows", "table '" + table_name + "'");
    const auto declared_table_bytes =
        required_uint64(table, "bytes", "table '" + table_name + "'");
    const auto& layout_value =
        required_member(table, "layout", rapidjson::kStringType,
                        "table '" + table_name + "'");
    const std::string layout = layout_value.GetString();

    const fs::path single_file = data_dir / (table_name + ".parquet");
    const fs::path table_dir = data_dir / table_name;
    if (layout == "single") {
        if (fs::exists(table_dir)) {
            throw std::runtime_error(
                "Ambiguous TPC-H table layout for '" + table_name +
                "': manifest declares single file but directory exists");
        }
        const auto& file =
            required_member(table, "file", rapidjson::kObjectType,
                            "table '" + table_name + "'");
        const fs::path path =
            safe_manifest_path(data_dir, file, "table '" + table_name + "' file");
        if (path != single_file.lexically_normal()) {
            throw std::runtime_error(
                "TPC-H manifest single table '" + table_name +
                "' must use path '" + single_file.string() + "'");
        }
        const auto bytes =
            required_uint64(file, "bytes", "table '" + table_name + "' file");
        const auto rows =
            required_uint64(file, "rows", "table '" + table_name + "' file");
        (void)required_uint64(
            file, "row_groups", "table '" + table_name + "' file");
        require_sha256(file, "table '" + table_name + "' file");
        if (bytes != declared_table_bytes || rows != declared_table_rows) {
            throw std::runtime_error(
                "TPC-H manifest single table '" + table_name +
                "' file metadata does not match table totals");
        }
        verify_declared_file(path, bytes, "table '" + table_name + "' file");
        verify_declared_schema(path, schema_fingerprint, table_name);
        return ResolvedTpchFiles{
            {path.string()}, path.string(), false, manifest_hash};
    }

    if (layout != "partitioned") {
        throw std::runtime_error(
            "TPC-H manifest table '" + table_name +
            "' has unsupported layout '" + layout + "'");
    }
    if (fs::exists(single_file)) {
        throw std::runtime_error(
            "Ambiguous TPC-H table layout for '" + table_name +
            "': manifest declares partitioned table but single file exists");
    }
    if (!fs::is_directory(table_dir)) {
        throw std::runtime_error(
            "TPC-H manifest partitioned table directory is missing: " +
            table_dir.string());
    }

    const auto part_count =
        required_uint64(table, "part_count", "table '" + table_name + "'");
    const auto& parts =
        required_member(table, "parts", rapidjson::kArrayType,
                        "table '" + table_name + "'");
    if (part_count == 0 || part_count != parts.Size()) {
        throw std::runtime_error(
            "TPC-H manifest table '" + table_name +
            "' part_count does not match its non-empty parts array");
    }
    if (part_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "TPC-H manifest table '" + table_name + "' has too many parts");
    }

    std::vector<fs::path> ordered_paths;
    std::vector<std::uint64_t> ordered_bytes;
    ordered_paths.reserve(parts.Size());
    ordered_bytes.reserve(parts.Size());
    std::set<std::string> unique_paths;
    std::uint64_t summed_rows = 0;
    std::uint64_t summed_bytes = 0;
    for (rapidjson::SizeType index = 0; index < parts.Size(); ++index) {
        const auto& part = parts[index];
        const std::string context =
            "table '" + table_name + "' part " + std::to_string(index + 1);
        const auto id = required_uint64(part, "id", context);
        if (id != static_cast<std::uint64_t>(index) + 1) {
            throw std::runtime_error(
                "TPC-H manifest " + context +
                " IDs must be ordered and contiguous from 1");
        }
        fs::path path = safe_manifest_path(data_dir, part, context);
        if (path.parent_path() != table_dir.lexically_normal()) {
            throw std::runtime_error(
                "TPC-H manifest " + context +
                " path must be directly inside " + table_dir.string());
        }
        if (!unique_paths.insert(path.string()).second) {
            throw std::runtime_error(
                "TPC-H manifest table '" + table_name +
                "' contains duplicate part path '" + path.string() + "'");
        }
        const auto bytes = required_uint64(part, "bytes", context);
        const auto rows = required_uint64(part, "rows", context);
        (void)required_uint64(part, "row_groups", context);
        require_sha256(part, context);
        if (std::numeric_limits<std::uint64_t>::max() - summed_rows < rows ||
            std::numeric_limits<std::uint64_t>::max() - summed_bytes < bytes) {
            throw std::runtime_error(
                "TPC-H manifest table '" + table_name +
                "' row/byte totals overflow uint64");
        }
        summed_rows += rows;
        summed_bytes += bytes;
        ordered_paths.push_back(std::move(path));
        ordered_bytes.push_back(bytes);
    }
    if (summed_rows != declared_table_rows ||
        summed_bytes != declared_table_bytes) {
        throw std::runtime_error(
            "TPC-H manifest table '" + table_name +
            "' part metadata does not match table totals");
    }

    if (validate_complete_directory) {
        std::set<std::string> actual_paths;
        for (const auto& entry : fs::directory_iterator(table_dir)) {
            if (entry.is_regular_file() &&
                entry.path().extension() == ".parquet") {
                actual_paths.insert(entry.path().lexically_normal().string());
            }
        }
        if (actual_paths != unique_paths) {
            throw std::runtime_error(
                "TPC-H manifest table '" + table_name +
                "' directory does not exactly match declared Parquet parts");
        }
    }

    std::vector<std::size_t> assigned;
    if (read_policy == SourceReadPolicy::Replicated || shard_count == 1) {
        assigned.resize(ordered_paths.size());
        for (std::size_t i = 0; i < assigned.size(); ++i) assigned[i] = i;
    } else {
        assigned = assigned_part_indices(
            ordered_paths.size(), shard_rank, shard_count, strategy);
    }

    std::vector<std::string> paths;
    paths.reserve(assigned.size());
    for (const auto index : assigned) {
        verify_declared_file(
            ordered_paths[index], ordered_bytes[index],
            "table '" + table_name + "' part " + std::to_string(index + 1));
        paths.push_back(ordered_paths[index].string());
    }
    const std::string schema_path = paths.empty()
        ? ordered_paths.front().string() : paths.front();
    verify_declared_schema(schema_path, schema_fingerprint, table_name);
    return ResolvedTpchFiles{
        std::move(paths), schema_path, true, manifest_hash};
}

ResolvedTpchFiles resolve_legacy_table(
    const fs::path& data_dir,
    const std::string& table_name,
    int shard_rank,
    int shard_count,
    SourceReadPolicy read_policy,
    FileShardStrategy strategy,
    bool allow_legacy_multifile)
{
    const fs::path table_dir = data_dir / table_name;
    const fs::path single_file = data_dir / (table_name + ".parquet");
    if (fs::is_directory(table_dir)) {
        if (fs::exists(single_file)) {
            throw std::runtime_error(
                "Ambiguous TPC-H table layout for '" + table_name +
                "': both single file and part directory exist");
        }
        if (!allow_legacy_multifile) {
            throw std::runtime_error(
                "Unmanifested multi-file TPC-H table '" + table_name +
                "' is disabled; publish manifest.json and _SUCCESS or "
                "explicitly enable legacy multi-file compatibility");
        }
        std::vector<std::string> all_parts;
        for (const auto& entry : fs::directory_iterator(table_dir)) {
            if (entry.is_regular_file() &&
                entry.path().extension() == ".parquet") {
                all_parts.push_back(entry.path().string());
            }
        }
        std::sort(all_parts.begin(), all_parts.end());
        if (all_parts.empty()) {
            throw std::runtime_error(
                "TPC-H table part directory contains no parquet files: " +
                table_dir.string());
        }
        std::vector<std::string> paths;
        if (read_policy == SourceReadPolicy::Replicated || shard_count == 1) {
            paths = all_parts;
        } else {
            for (const auto index : assigned_part_indices(
                     all_parts.size(), shard_rank, shard_count, strategy)) {
                paths.push_back(all_parts[index]);
            }
        }
        return ResolvedTpchFiles{
            std::move(paths), all_parts.front(), true, "legacy-multifile"};
    }

    if (!fs::is_regular_file(single_file)) {
        throw std::runtime_error(
            "TPC-H table parquet file does not exist: " +
            single_file.string() +
            ". Expected a legacy single file or a completed manifest dataset");
    }
    return ResolvedTpchFiles{
        {single_file.string()}, single_file.string(), false, "legacy-single"};
}

} // namespace

const char* file_shard_strategy_name(FileShardStrategy strategy) noexcept {
    switch (strategy) {
        case FileShardStrategy::RoundRobin: return "round_robin";
        case FileShardStrategy::ContiguousRange: return "contiguous_range";
    }
    return "unknown";
}

FileShardStrategy parse_file_shard_strategy(const std::string& value) {
    if (value == "round_robin" || value == "round-robin") {
        return FileShardStrategy::RoundRobin;
    }
    if (value == "contiguous_range" || value == "contiguous-range" ||
        value == "range") {
        return FileShardStrategy::ContiguousRange;
    }
    throw std::runtime_error(
        "Unknown file shard strategy '" + value +
        "'; expected round_robin or contiguous_range");
}

std::string sha256_file_hex(const std::string& path) {
    return sha256_file(path);
}

std::vector<std::size_t> assigned_part_indices(
    std::size_t part_count,
    int shard_rank,
    int shard_count,
    FileShardStrategy strategy)
{
    if (shard_count <= 0 || shard_rank < 0 || shard_rank >= shard_count) {
        throw std::runtime_error(
            "Invalid TPC-H file shard: rank=" + std::to_string(shard_rank) +
            " count=" + std::to_string(shard_count));
    }
    std::vector<std::size_t> result;
    if (strategy == FileShardStrategy::RoundRobin) {
        for (std::size_t index = static_cast<std::size_t>(shard_rank);
             index < part_count;
             index += static_cast<std::size_t>(shard_count)) {
            result.push_back(index);
        }
        return result;
    }

    const auto count = static_cast<std::size_t>(shard_count);
    const auto rank = static_cast<std::size_t>(shard_rank);
    // floor(rank * part_count / count), expressed as quotient/remainder.
    // The small remainder product uses 128-bit arithmetic to avoid overflow.
    const auto quotient = part_count / count;
    const auto remainder = part_count % count;
    const auto scaled_remainder = [&](std::size_t boundary) {
        return static_cast<std::size_t>(
            (static_cast<unsigned __int128>(boundary) * remainder) / count);
    };
    const auto begin = rank * quotient + scaled_remainder(rank);
    const auto end =
        (rank + 1) * quotient + scaled_remainder(rank + 1);
    const auto length = end - begin;
    result.reserve(length);
    for (std::size_t index = begin; index < end; ++index) {
        result.push_back(index);
    }
    return result;
}

ResolvedTpchFiles resolve_tpch_files(
    const std::string& data_dir,
    const std::string& table_name,
    int expected_scale_factor,
    int shard_rank,
    int shard_count,
    SourceReadPolicy read_policy,
    FileShardStrategy strategy,
    bool allow_legacy_multifile,
    bool validate_complete_directory)
{
    const fs::path root(data_dir);
    if (fs::is_regular_file(root / "manifest.json")) {
        return resolve_manifest_table(
            root, table_name, expected_scale_factor, shard_rank, shard_count,
            read_policy, strategy, validate_complete_directory);
    }
    if (fs::exists(root / "_SUCCESS")) {
        throw std::runtime_error(
            "TPC-H dataset has _SUCCESS but no manifest.json: " + data_dir);
    }
    return resolve_legacy_table(
        root, table_name, shard_rank, shard_count, read_policy, strategy,
        allow_legacy_multifile);
}

void preflight_tpch_dataset(
    const std::string& data_dir,
    int expected_scale_factor,
    bool allow_legacy_multifile)
{
    static constexpr std::array<const char*, 8> tables {
        "customer", "lineitem", "nation", "orders",
        "part", "partsupp", "region", "supplier"};
    for (const char* table : tables) {
        const auto resolved = resolve_tpch_files(
            data_dir, table, expected_scale_factor, 0, 1,
            SourceReadPolicy::Sharded, FileShardStrategy::RoundRobin,
            allow_legacy_multifile, true);
        std::shared_ptr<arrow::Schema> canonical;
        for (const auto& path : resolved.paths) {
            auto schema = read_arrow_schema(path);
            if (canonical == nullptr) {
                canonical = std::move(schema);
            } else if (!canonical->Equals(*schema, /*check_metadata=*/true)) {
                throw std::runtime_error(
                    "TPC-H preflight schema mismatch for table '" +
                    std::string(table) + "' file '" + path + "': expected " +
                    canonical->ToString() + ", got " + schema->ToString());
            }
        }
    }
}

} // namespace gamax::tpch
