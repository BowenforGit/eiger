// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace {

struct Metadata {
    std::string path;
    std::int64_t rows;
    int row_groups;
    std::shared_ptr<arrow::Schema> schema;
};

Metadata inspect(const std::string& path) {
    auto input = arrow::io::ReadableFile::Open(path);
    if (!input.ok()) {
        throw std::runtime_error(
            "failed to open '" + path + "': " + input.status().ToString());
    }
    parquet::arrow::FileReaderBuilder builder;
    auto status = builder.Open(*input);
    if (!status.ok()) {
        throw std::runtime_error(
            "failed to parse '" + path + "': " + status.ToString());
    }
    std::unique_ptr<parquet::arrow::FileReader> reader;
    status = builder.Build(&reader);
    if (!status.ok()) {
        throw std::runtime_error(
            "failed to build reader for '" + path + "': " +
            status.ToString());
    }
    std::shared_ptr<arrow::Schema> schema;
    status = reader->GetSchema(&schema);
    if (!status.ok()) {
        throw std::runtime_error(
            "failed to read schema for '" + path + "': " +
            status.ToString());
    }
    const auto parquet_metadata = reader->parquet_reader()->metadata();
    return Metadata{
        path,
        parquet_metadata->num_rows(),
        parquet_metadata->num_row_groups(),
        std::move(schema)};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " PARQUET [PARQUET ...]\n";
        return 2;
    }
    try {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.StartArray();
        for (int index = 1; index < argc; ++index) {
            const auto metadata = inspect(argv[index]);
            writer.StartObject();
            writer.Key("path");
            writer.String(metadata.path.c_str());
            writer.Key("rows");
            writer.Int64(metadata.rows);
            writer.Key("row_groups");
            writer.Int(metadata.row_groups);
            writer.Key("schema");
            writer.StartArray();
            for (const auto& field : metadata.schema->fields()) {
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
            writer.EndObject();
        }
        writer.EndArray();
        std::cout << buffer.GetString() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
