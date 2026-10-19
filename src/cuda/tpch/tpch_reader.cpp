// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "tpch/tpch.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <string>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/util/thread_pool.h>
#include <parquet/arrow/reader.h>
#include <parquet/properties.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string_column.hpp"
#include "tpch/tpch_layout.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

namespace gamax::tpch {

namespace {

// ---------------------------------------------------------------------------
// Arrow type → gamax type_id mapping
// ---------------------------------------------------------------------------
gamax::type_id arrow_type_to_gamax(const std::shared_ptr<arrow::DataType>& arrow_type) {
    switch (arrow_type->id()) {
        case arrow::Type::BOOL:    return gamax::type_id::BOOL8;
        case arrow::Type::INT8:    return gamax::type_id::INT8;
        case arrow::Type::INT16:   return gamax::type_id::INT16;
        case arrow::Type::INT32:   return gamax::type_id::INT32;
        case arrow::Type::INT64:   return gamax::type_id::INT64;
        case arrow::Type::UINT8:   return gamax::type_id::UINT8;
        case arrow::Type::UINT16:  return gamax::type_id::UINT16;
        case arrow::Type::UINT32:  return gamax::type_id::UINT32;
        case arrow::Type::UINT64:  return gamax::type_id::UINT64;
        case arrow::Type::FLOAT:   return gamax::type_id::FLOAT32;
        case arrow::Type::DOUBLE:  return gamax::type_id::FLOAT64;
        case arrow::Type::DATE32:  return gamax::type_id::INT32;   // days since epoch → int32
        case arrow::Type::DATE64:  return gamax::type_id::INT64;   // milliseconds since epoch → int64
        case arrow::Type::DECIMAL128:
        case arrow::Type::DECIMAL256: {
            auto decimal_type = std::static_pointer_cast<arrow::DecimalType>(arrow_type);
            if (decimal_type->scale() == 0) {
                return gamax::type_id::INT64;
            }
            return gamax::type_id::FLOAT64;
        }
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
            return gamax::type_id::STRING;
        default:
            throw std::runtime_error(
                "Unsupported Arrow type: " + arrow_type->ToString());
    }
}

// ---------------------------------------------------------------------------
// Flatten a chunked array into a single contiguous Arrow array
// ---------------------------------------------------------------------------
std::shared_ptr<arrow::Array> flatten_chunked_array(
    const std::shared_ptr<arrow::ChunkedArray>& chunked)
{
    if (chunked->num_chunks() == 0) {
        throw std::runtime_error("ChunkedArray has no chunks");
    }
    if (chunked->num_chunks() == 1) {
        return chunked->chunk(0);
    }
    // Concatenate all chunks into a single array
    auto result = arrow::Concatenate(chunked->chunks());
    if (!result.ok()) {
        throw std::runtime_error(
            "Failed to concatenate chunked array: " + result.status().ToString());
    }
    return *result;
}

// ---------------------------------------------------------------------------
// Pinned host memory cache for converted columns
// ---------------------------------------------------------------------------
struct PinnedColumnData {
    void* pinned_ptr = nullptr;
    gamax::size_type num_rows = 0;
    gamax::data_type type;
    size_t byte_size = 0;

    PinnedColumnData() = default;

    PinnedColumnData(void* ptr, gamax::size_type rows, gamax::data_type t, size_t bytes)
        : pinned_ptr(ptr), num_rows(rows), type(t), byte_size(bytes) {}

    ~PinnedColumnData() {
        if (pinned_ptr) cudaFreeHost(pinned_ptr);
    }

    PinnedColumnData(const PinnedColumnData&) = delete;
    PinnedColumnData& operator=(const PinnedColumnData&) = delete;

    PinnedColumnData(PinnedColumnData&& o) noexcept
        : pinned_ptr(o.pinned_ptr), num_rows(o.num_rows), type(o.type), byte_size(o.byte_size) {
        o.pinned_ptr = nullptr;
    }

    PinnedColumnData& operator=(PinnedColumnData&& o) noexcept {
        if (this != &o) {
            if (pinned_ptr) cudaFreeHost(pinned_ptr);
            pinned_ptr = o.pinned_ptr;
            num_rows = o.num_rows;
            type = o.type;
            byte_size = o.byte_size;
            o.pinned_ptr = nullptr;
        }
        return *this;
    }
};

// Allocate pinned memory and copy host data into it
PinnedColumnData make_pinned_cache(
    const void* host_data,
    gamax::size_type num_rows,
    gamax::data_type type,
    size_t byte_size)
{
    void* pinned = nullptr;
    cudaMallocHost(&pinned, byte_size);
    std::memcpy(pinned, host_data, byte_size);
    return PinnedColumnData(pinned, num_rows, type, byte_size);
}

// ---------------------------------------------------------------------------
// Convert Arrow array to gamax column + populate pinned cache
// ---------------------------------------------------------------------------

// Functor for type-dispatched direct-copy conversion
struct direct_copy_functor {
    template <typename T>
    std::pair<std::unique_ptr<gamax::column>, PinnedColumnData> operator()(
        const std::shared_ptr<arrow::Array>& array,
        cudaStream_t stream) const
    {
        auto num_rows = static_cast<gamax::size_type>(array->length());
        const T* host_data = array->data()->GetValues<T>(1);
        size_t byte_size = num_rows * sizeof(T);
        auto cache = make_pinned_cache(host_data, num_rows,
            gamax::data_type{gamax::type_to_id<T>()}, byte_size);
        // H2D from pinned memory (faster DMA)
        auto col = gamax::make_column_from_host<T>(static_cast<T*>(cache.pinned_ptr), num_rows, stream);
        return {std::move(col), std::move(cache)};
    }
};

// Convert Decimal128 column → int64 (scale == 0)
std::pair<std::unique_ptr<gamax::column>, PinnedColumnData> convert_decimal_to_int64(
    const std::shared_ptr<arrow::Array>& array,
    cudaStream_t stream)
{
    auto decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(array);
    auto num_rows = static_cast<gamax::size_type>(array->length());
    size_t byte_size = num_rows * sizeof(int64_t);

    // Allocate pinned memory directly for conversion
    void* pinned = nullptr;
    cudaMallocHost(&pinned, byte_size);
    auto* buf = static_cast<int64_t*>(pinned);

    for (gamax::size_type i = 0; i < num_rows; ++i) {
        arrow::Decimal128 val(decimal_array->GetValue(i));
        auto result = val.ToInteger<int64_t>();
        if (!result.ok()) {
            cudaFreeHost(pinned);
            throw std::runtime_error(
                "Decimal128 value at row " + std::to_string(i) +
                " does not fit in int64: " + result.status().ToString());
        }
        buf[i] = *result;
    }

    auto col = gamax::make_column_from_host<int64_t>(buf, num_rows, stream);
    PinnedColumnData cache(pinned, num_rows, gamax::data_type{gamax::type_id::INT64}, byte_size);
    return {std::move(col), std::move(cache)};
}

// Convert days-since-1970-01-01 → int32 YYYYMMDD (e.g. 9404 → 19951001).
// Uses C++20 <chrono> calendar types: `sys_days` is the Unix-epoch-based
// time_point at days resolution, and `year_month_day` converts it to
// calendar fields.
inline int32_t days_since_epoch_to_yyyymmdd(int32_t days) {
    using namespace std::chrono;
    sys_days sd{std::chrono::days{days}};
    year_month_day ymd{sd};
    const int y = static_cast<int>(ymd.year());
    const unsigned m = static_cast<unsigned>(ymd.month());
    const unsigned d = static_cast<unsigned>(ymd.day());
    return y * 10000 + static_cast<int32_t>(m) * 100 + static_cast<int32_t>(d);
}

// Convert Arrow DATE32 column → int32 YYYYMMDD. We store dates as
// packed decimal YYYYMMDD rather than days-since-epoch so query-side
// literals are human-readable (e.g. `lit(19951001)`) and year/month/day
// extraction via `year_of` / similar ops is a pair of integer divides
// instead of a conditional epoch-offset table.
std::pair<std::unique_ptr<gamax::column>, PinnedColumnData> convert_date32_to_yyyymmdd(
    const std::shared_ptr<arrow::Array>& array,
    cudaStream_t stream)
{
    auto date_array = std::static_pointer_cast<arrow::Date32Array>(array);
    auto num_rows = static_cast<gamax::size_type>(array->length());
    size_t byte_size = num_rows * sizeof(int32_t);

    void* pinned = nullptr;
    cudaMallocHost(&pinned, byte_size);
    auto* buf = static_cast<int32_t*>(pinned);

    const int32_t* src = date_array->raw_values();
    for (gamax::size_type i = 0; i < num_rows; ++i) {
        buf[i] = days_since_epoch_to_yyyymmdd(src[i]);
    }

    auto col = gamax::make_column_from_host<int32_t>(buf, num_rows, stream);
    PinnedColumnData cache(pinned, num_rows, gamax::data_type{gamax::type_id::INT32}, byte_size);
    return {std::move(col), std::move(cache)};
}

// Convert Decimal128 column → double (scale != 0)
std::pair<std::unique_ptr<gamax::column>, PinnedColumnData> convert_decimal_to_double(
    const std::shared_ptr<arrow::Array>& array,
    int32_t scale,
    cudaStream_t stream)
{
    auto decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(array);
    auto num_rows = static_cast<gamax::size_type>(array->length());
    size_t byte_size = num_rows * sizeof(double);

    void* pinned = nullptr;
    cudaMallocHost(&pinned, byte_size);
    auto* buf = static_cast<double*>(pinned);

    for (gamax::size_type i = 0; i < num_rows; ++i) {
        arrow::Decimal128 val(decimal_array->GetValue(i));
        buf[i] = val.ToDouble(scale);
    }

    auto col = gamax::make_column_from_host<double>(buf, num_rows, stream);
    PinnedColumnData cache(pinned, num_rows, gamax::data_type{gamax::type_id::FLOAT64}, byte_size);
    return {std::move(col), std::move(cache)};
}

// Main conversion: Arrow array → gamax column + pinned cache entry
std::pair<std::unique_ptr<gamax::column>, PinnedColumnData> arrow_array_to_gamax_column_cached(
    const std::shared_ptr<arrow::Array>& array,
    const std::shared_ptr<arrow::DataType>& arrow_type,
    cudaStream_t stream)
{
    gamax::type_id target_type = arrow_type_to_gamax(arrow_type);

    if (arrow_type->id() == arrow::Type::DECIMAL128 ||
        arrow_type->id() == arrow::Type::DECIMAL256)
    {
        auto decimal_type = std::static_pointer_cast<arrow::DecimalType>(arrow_type);
        if (decimal_type->scale() == 0) {
            return convert_decimal_to_int64(array, stream);
        } else {
            return convert_decimal_to_double(array, decimal_type->scale(), stream);
        }
    }

    if (arrow_type->id() == arrow::Type::DATE32) {
        return convert_date32_to_yyyymmdd(array, stream);
    }

    return gamax::type_dispatcher(
        gamax::data_type{target_type},
        direct_copy_functor{},
        array, stream);
}

// Create a GPU column from an existing pinned cache entry
std::unique_ptr<gamax::column> column_from_pinned_cache(
    const PinnedColumnData& cache,
    cudaStream_t stream)
{
    gamax::device_buffer buf(cache.byte_size, stream);
    cudaMemcpyAsync(buf.data(), cache.pinned_ptr, cache.byte_size,
                    cudaMemcpyHostToDevice, stream);
    return std::make_unique<gamax::column>(cache.type, cache.num_rows, std::move(buf));
}

// ---------------------------------------------------------------------------
// Pinned host memory cache for string columns (offsets + data)
// ---------------------------------------------------------------------------
struct PinnedStringColumnData {
    void* pinned_offsets = nullptr;   // int64_t offsets, (num_rows + 1) entries
    void* pinned_data = nullptr;      // flat char data
    gamax::size_type num_rows = 0;
    size_t offsets_byte_size = 0;
    size_t data_byte_size = 0;

    PinnedStringColumnData() = default;

    PinnedStringColumnData(void* offsets, void* data,
                           gamax::size_type rows,
                           size_t off_bytes, size_t dat_bytes)
        : pinned_offsets(offsets), pinned_data(data)
        , num_rows(rows)
        , offsets_byte_size(off_bytes), data_byte_size(dat_bytes) {}

    ~PinnedStringColumnData() {
        if (pinned_offsets) cudaFreeHost(pinned_offsets);
        if (pinned_data) cudaFreeHost(pinned_data);
    }

    PinnedStringColumnData(const PinnedStringColumnData&) = delete;
    PinnedStringColumnData& operator=(const PinnedStringColumnData&) = delete;

    PinnedStringColumnData(PinnedStringColumnData&& o) noexcept
        : pinned_offsets(o.pinned_offsets), pinned_data(o.pinned_data)
        , num_rows(o.num_rows)
        , offsets_byte_size(o.offsets_byte_size), data_byte_size(o.data_byte_size) {
        o.pinned_offsets = nullptr;
        o.pinned_data = nullptr;
    }

    PinnedStringColumnData& operator=(PinnedStringColumnData&& o) noexcept {
        if (this != &o) {
            if (pinned_offsets) cudaFreeHost(pinned_offsets);
            if (pinned_data) cudaFreeHost(pinned_data);
            pinned_offsets = o.pinned_offsets;
            pinned_data = o.pinned_data;
            num_rows = o.num_rows;
            offsets_byte_size = o.offsets_byte_size;
            data_byte_size = o.data_byte_size;
            o.pinned_offsets = nullptr;
            o.pinned_data = nullptr;
        }
        return *this;
    }
};

// Convert an Arrow chunked STRING / LARGE_STRING column → pinned cache.
//
// Why chunked, not flat: at large scale a single column's total bytes
// can exceed INT32_MAX (2 GiB), and arrow::StringArray uses int32
// offsets — so `arrow::Concatenate` on the chunks overflows with the
// "offset overflow ... consider casting to large_string" error. Arrow's
// Parquet reader already splits long string columns into multiple
// chunks that each fit int32. We aggregate those chunks into one
// contiguous int64-offset buffer pair directly, which is what the GPU
// string_column format expects anyway. This path handles single-chunk
// inputs as a trivial case.
PinnedStringColumnData convert_arrow_chunked_string_to_pinned(
    const std::shared_ptr<arrow::ChunkedArray>& chunked)
{
    const bool is_large = (chunked->type()->id() == arrow::Type::LARGE_STRING);
    gamax::size_type num_rows = 0;
    int64_t total_bytes = 0;
    for (int c = 0; c < chunked->num_chunks(); ++c) {
        auto chk = chunked->chunk(c);
        num_rows += static_cast<gamax::size_type>(chk->length());
        if (is_large) {
            auto la = std::static_pointer_cast<arrow::LargeStringArray>(chk);
            const int64_t* off = la->raw_value_offsets();
            total_bytes += off[la->length()] - off[0];
        } else {
            auto sa = std::static_pointer_cast<arrow::StringArray>(chk);
            const int32_t* off = sa->raw_value_offsets();
            total_bytes += static_cast<int64_t>(off[sa->length()] - off[0]);
        }
    }

    size_t offsets_byte_size =
        static_cast<size_t>(num_rows + 1) * sizeof(int64_t);
    void* pinned_offsets = nullptr;
    cudaMallocHost(&pinned_offsets, offsets_byte_size);
    auto* dst_offsets = static_cast<int64_t*>(pinned_offsets);
    dst_offsets[0] = 0;

    size_t data_byte_size = static_cast<size_t>(total_bytes);
    void* pinned_data = nullptr;
    if (data_byte_size > 0) {
        cudaMallocHost(&pinned_data, data_byte_size);
    }

    gamax::size_type row_cursor = 0;
    int64_t byte_cursor = 0;
    for (int c = 0; c < chunked->num_chunks(); ++c) {
        auto chk = chunked->chunk(c);
        auto chk_len = static_cast<gamax::size_type>(chk->length());
        int64_t chunk_bytes;
        const uint8_t* src_data_ptr;
        if (is_large) {
            auto la = std::static_pointer_cast<arrow::LargeStringArray>(chk);
            const int64_t* src_off = la->raw_value_offsets();
            const int64_t chunk_start = src_off[0];
            for (gamax::size_type i = 0; i < chk_len; ++i) {
                dst_offsets[row_cursor + i + 1] =
                    byte_cursor + (src_off[i + 1] - chunk_start);
            }
            chunk_bytes = src_off[chk_len] - chunk_start;
            src_data_ptr = la->value_data()->data() + chunk_start;
        } else {
            auto sa = std::static_pointer_cast<arrow::StringArray>(chk);
            const int32_t* src_off = sa->raw_value_offsets();
            const int32_t chunk_start = src_off[0];
            for (gamax::size_type i = 0; i < chk_len; ++i) {
                dst_offsets[row_cursor + i + 1] = byte_cursor +
                    static_cast<int64_t>(src_off[i + 1] - chunk_start);
            }
            chunk_bytes = static_cast<int64_t>(src_off[chk_len] - chunk_start);
            src_data_ptr = sa->value_data()->data() + chunk_start;
        }
        if (chunk_bytes > 0 && pinned_data) {
            std::memcpy(static_cast<char*>(pinned_data) + byte_cursor,
                        src_data_ptr, static_cast<size_t>(chunk_bytes));
        }
        byte_cursor += chunk_bytes;
        row_cursor += chk_len;
    }

    return PinnedStringColumnData(
        pinned_offsets, pinned_data, num_rows,
        offsets_byte_size, data_byte_size);
}

// Create a GPU string_column from a pinned string cache entry
std::unique_ptr<gamax::column> string_column_from_pinned_cache(
    const PinnedStringColumnData& cache,
    cudaStream_t stream)
{
    // Offsets → device meta buffer
    gamax::device_buffer meta_buf(cache.offsets_byte_size, stream);
    cudaMemcpyAsync(meta_buf.data(), cache.pinned_offsets, cache.offsets_byte_size,
                    cudaMemcpyHostToDevice, stream);

    // Data → device data buffer
    gamax::device_buffer data_buf(
        cache.data_byte_size > 0 ? cache.data_byte_size : 1, stream);
    if (cache.data_byte_size > 0) {
        cudaMemcpyAsync(data_buf.data(), cache.pinned_data, cache.data_byte_size,
                        cudaMemcpyHostToDevice, stream);
    }

    return std::make_unique<gamax::string_column>(
        std::move(data_buf), std::move(meta_buf),
        cache.num_rows, gamax::string_impl_id::ARROW_STRING);
}

std::unique_ptr<gamax::column> make_empty_column_for_arrow_type(
    const std::shared_ptr<arrow::DataType>& arrow_type,
    cudaStream_t stream)
{
    const auto target_type = arrow_type_to_gamax(arrow_type);
    if (target_type == gamax::type_id::STRING) {
        int64_t zero = 0;
        gamax::device_buffer meta_buf(sizeof(int64_t), stream);
        cudaMemcpyAsync(meta_buf.data(), &zero, sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream);
        gamax::device_buffer data_buf(1, stream);
        return std::make_unique<gamax::string_column>(
            std::move(data_buf), std::move(meta_buf),
            gamax::size_type{0}, gamax::string_impl_id::ARROW_STRING);
    }
    return gamax::make_column(gamax::data_type{target_type}, 0, stream);
}

// ---------------------------------------------------------------------------
// Open a Parquet file and build a column-projecting, thread-parallel
// FileReader. Pre-buffering is on by default in ArrowReaderProperties;
// we just flip use_threads so per-column decode runs on the Arrow CPU
// thread pool. The reader keeps the file open and the footer parsed, so
// subsequent ReadTable calls with different column index subsets are
// cheap.
// ---------------------------------------------------------------------------
struct OpenedParquet {
    std::unique_ptr<parquet::arrow::FileReader> reader;
    std::shared_ptr<arrow::Schema> schema;
};

OpenedParquet open_parquet_for_projection(const std::string& file_path) {
    auto maybe_infile = arrow::io::ReadableFile::Open(file_path);
    if (!maybe_infile.ok()) {
        throw std::runtime_error(
            "Failed to open Parquet file '" + file_path + "': " +
            maybe_infile.status().ToString());
    }

    parquet::ArrowReaderProperties arrow_props(/*use_threads=*/true);

    parquet::arrow::FileReaderBuilder builder;
    auto st = builder.Open(*maybe_infile);
    if (!st.ok()) {
        throw std::runtime_error(
            "FileReaderBuilder::Open failed for '" + file_path + "': " +
            st.ToString());
    }
    builder.memory_pool(arrow::default_memory_pool());
    builder.properties(arrow_props);

    std::unique_ptr<parquet::arrow::FileReader> reader;
    st = builder.Build(&reader);
    if (!st.ok()) {
        throw std::runtime_error(
            "FileReaderBuilder::Build failed for '" + file_path + "': " +
            st.ToString());
    }
    reader->set_use_threads(true);

    std::shared_ptr<arrow::Schema> schema;
    st = reader->GetSchema(&schema);
    if (!st.ok()) {
        throw std::runtime_error(
            "Failed to read schema from '" + file_path + "': " +
            st.ToString());
    }

    return OpenedParquet{std::move(reader), std::move(schema)};
}

// ---------------------------------------------------------------------------
// Resolve the data directory from environment variable
// ---------------------------------------------------------------------------
std::string get_tpch_data_dir(gamax::size_type scale_factor) {
    const char* data_path = std::getenv("TPCH_DATA_PATH");
    if (data_path == nullptr || std::string(data_path).empty()) {
        throw std::runtime_error(
            "Environment variable TPCH_DATA_PATH is not set. "
            "Please set it to the root directory containing TPC-H Parquet data "
            "(e.g., export TPCH_DATA_PATH=/path/to/tpch_data).");
    }

    std::filesystem::path base_path(data_path);
    std::filesystem::path sf_dir = base_path / ("sf" + std::to_string(scale_factor));

    if (!std::filesystem::exists(sf_dir)) {
        throw std::runtime_error(
            "TPC-H data directory does not exist: " + sf_dir.string() + ". "
            "Expected directory structure: $TPCH_DATA_PATH/sf<N>/<table>.parquet");
    }

    return sf_dir.string();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// TpchTableReader::Impl — holds per-table Parquet readers + per-column
// pinned-host caches. The Parquet reader is opened lazily on first access
// and retained so subsequent loads for the same table only pay for the
// columns not yet cached.
// ---------------------------------------------------------------------------
struct TpchTableReader::Impl {
    struct FileHandle {
        std::string path;
        std::unique_ptr<parquet::arrow::FileReader> reader;
    };
    struct TableHandle {
        std::vector<FileHandle> files;
        std::shared_ptr<arrow::Schema> schema;
        bool file_partitioned {false};
    };
    std::unordered_map<std::string, TableHandle> table_handles;
    std::unordered_map<std::string, PinnedColumnData> pinned_column_cache;
    std::unordered_map<std::string, PinnedStringColumnData> pinned_string_column_cache;
};

std::string source_cache_suffix(
    const tpch_params& params, SourceReadPolicy read_policy)
{
    std::string suffix = "#policy=";
    suffix += source_read_policy_name(read_policy);
    if (!params.row_group_sharded() ||
        read_policy == SourceReadPolicy::Replicated) {
        return suffix;
    }
    return suffix + "#shard=" + std::to_string(params.shard_rank) +
           "/" + std::to_string(params.shard_count) +
           "#files=" + file_shard_strategy_name(params.file_shard_strategy);
}

std::vector<int> assigned_row_groups(
    const parquet::arrow::FileReader& reader,
    const tpch_params& params,
    SourceReadPolicy read_policy)
{
    std::vector<int> groups;
    if (!params.row_group_sharded() ||
        read_policy == SourceReadPolicy::Replicated) {
        return groups;
    }
    const int n = reader.num_row_groups();
    for (int i = 0; i < n; ++i) {
        if ((i % params.shard_count) == params.shard_rank) {
            groups.push_back(i);
        }
    }
    return groups;
}

// ---------------------------------------------------------------------------
// TpchTableReader constructors / destructor / move
// ---------------------------------------------------------------------------
TpchTableReader::TpchTableReader(const tpch_params& params)
    : params_(params)
    , impl_(std::make_unique<Impl>())
{
    auto status = arrow::SetCpuThreadPoolCapacity(
        params_.parquet_decode_threads);
    if (!status.ok()) {
        throw std::runtime_error(
            "Failed to set Arrow CPU thread-pool capacity to " +
            std::to_string(params_.parquet_decode_threads) + ": " +
            status.ToString());
    }
}

TpchTableReader::~TpchTableReader() = default;

TpchTableReader::TpchTableReader(TpchTableReader&&) noexcept = default;
TpchTableReader& TpchTableReader::operator=(TpchTableReader&&) noexcept = default;

// ---------------------------------------------------------------------------
// TpchTableReader::load
//
// Strategy: the per-column pinned cache is the source of truth. On a
// load() call we first emit cache hits, then batch all *distinct*
// uncached columns into a single `ReadTable({indices}, &tbl)` Parquet
// read. That read is column-projecting (skips decode of untouched
// columns) and parallel (use_threads was set on the reader). Because a
// TPC-H query typically touches a small subset of lineitem, this turns
// a ~2.4 GiB full decode into a few-hundred-MiB projected decode on
// the first call.
// ---------------------------------------------------------------------------
std::unique_ptr<gamax::table> TpchTableReader::load(
    const std::string& table_name,
    std::vector<std::string> column_names)
{
    const auto read_policy = params_.source_read_policy(table_name);

    // Lazily open + cache the Parquet reader for this table.
    auto handle_it = impl_->table_handles.find(table_name);
    if (handle_it == impl_->table_handles.end()) {
        std::string data_dir = get_tpch_data_dir(params_.scale_factor);
        auto resolved = resolve_tpch_files(
            data_dir, table_name, static_cast<int>(params_.scale_factor),
            params_.shard_rank, params_.shard_count, read_policy,
            params_.file_shard_strategy, params_.allow_legacy_multifile);
        Impl::TableHandle h;
        h.file_partitioned = resolved.file_partitioned;

        auto schema_open = open_parquet_for_projection(resolved.schema_path);
        h.schema = std::move(schema_open.schema);

        for (const auto& file_path : resolved.paths) {
            std::cout << "[TpchTableReader] Opening Parquet file: "
                      << file_path << " ("
                      << source_read_policy_name(read_policy) << ")" << std::endl;
            auto op = open_parquet_for_projection(file_path);
            if (!h.schema->Equals(*op.schema, /*check_metadata=*/true)) {
                throw std::runtime_error(
                    "Parquet schema mismatch for table '" + table_name +
                    "' file '" + file_path + "': expected " +
                    h.schema->ToString() + ", got " + op.schema->ToString());
            }
            h.files.push_back(Impl::FileHandle{
                file_path, std::move(op.reader)});
        }
        handle_it = impl_->table_handles.emplace(table_name, std::move(h)).first;
    }
    const auto& handle = handle_it->second;

    std::vector<std::unique_ptr<gamax::column>> gpu_columns(column_names.size());
    const std::string shard_suffix =
        source_cache_suffix(params_, read_policy);

    // Partition: cache hits vs. missing (dedup'd by column name, since
    // a caller could legitimately request the same column twice).
    std::unordered_map<std::string, std::vector<size_t>> missing;
    for (size_t i = 0; i < column_names.size(); ++i) {
        std::string cache_key = table_name + "." + column_names[i] + shard_suffix;
        auto pinned_it = impl_->pinned_column_cache.find(cache_key);
        if (pinned_it != impl_->pinned_column_cache.end()) {
            gpu_columns[i] = column_from_pinned_cache(pinned_it->second, params_.stream);
            continue;
        }
        auto str_it = impl_->pinned_string_column_cache.find(cache_key);
        if (str_it != impl_->pinned_string_column_cache.end()) {
            gpu_columns[i] = string_column_from_pinned_cache(str_it->second, params_.stream);
            continue;
        }
        missing[column_names[i]].push_back(i);
    }

    if (!missing.empty()) {
        // Resolve requested names → Parquet schema indices. If any name
        // doesn't exist, fail early with the list of available columns.
        std::vector<int> col_indices;
        std::vector<std::string> ordered_missing;
        col_indices.reserve(missing.size());
        ordered_missing.reserve(missing.size());
        for (const auto& kv : missing) {
            const std::string& name = kv.first;
            int idx = handle.schema->GetFieldIndex(name);
            if (idx < 0) {
                std::string available;
                for (int i = 0; i < handle.schema->num_fields(); ++i) {
                    if (i > 0) available += ", ";
                    available += handle.schema->field(i)->name();
                }
                throw std::runtime_error(
                    "Column '" + name + "' not found in table '" + table_name + "'. "
                    "Available columns: " + available);
            }
            col_indices.push_back(idx);
            ordered_missing.push_back(name);
        }

        auto make_empty_result = [&]() {
            for (size_t mi = 0; mi < ordered_missing.size(); ++mi) {
                const std::string& name = ordered_missing[mi];
                const std::string cache_key =
                    table_name + "." + name + shard_suffix;
                auto arrow_type = handle.schema
                    ->field(handle.schema->GetFieldIndex(name))->type();
                for (size_t pos : missing.at(name)) {
                    gpu_columns[pos] =
                        make_empty_column_for_arrow_type(arrow_type, params_.stream);
                }
                if (arrow_type->id() == arrow::Type::STRING ||
                    arrow_type->id() == arrow::Type::LARGE_STRING) {
                    int64_t zero = 0;
                    void* pinned_offsets = nullptr;
                    cudaMallocHost(&pinned_offsets, sizeof(int64_t));
                    std::memcpy(pinned_offsets, &zero, sizeof(int64_t));
                    PinnedStringColumnData str_cache(
                        pinned_offsets, nullptr, 0, sizeof(int64_t), 0);
                    impl_->pinned_string_column_cache.emplace(
                        cache_key, std::move(str_cache));
                } else {
                    PinnedColumnData pinned_cache(
                        nullptr, 0,
                        gamax::data_type{arrow_type_to_gamax(arrow_type)}, 0);
                    impl_->pinned_column_cache.emplace(
                        cache_key, std::move(pinned_cache));
                }
            }
            cudaStreamSynchronize(params_.stream);
            return gamax::make_table(std::move(gpu_columns));
        };

        std::vector<std::shared_ptr<arrow::Table>> projected_parts;
        projected_parts.reserve(handle.files.size());
        for (const auto& file : handle.files) {
            std::shared_ptr<arrow::Table> part_projected;
            if (params_.row_group_sharded() &&
                read_policy == SourceReadPolicy::Sharded &&
                !handle.file_partitioned) {
                std::vector<int> row_groups =
                    assigned_row_groups(*file.reader, params_, read_policy);
                if (row_groups.empty()) {
                    continue;
                }
                auto projected_status = file.reader->ReadRowGroups(
                    row_groups, col_indices, &part_projected);
                if (!projected_status.ok()) {
                    throw std::runtime_error(
                        "Parquet ReadRowGroups (projected) failed for table '" +
                        table_name + "' file '" + file.path + "': " +
                        projected_status.ToString());
                }
            } else {
                auto projected_status =
                    file.reader->ReadTable(col_indices, &part_projected);
                if (!projected_status.ok()) {
                    throw std::runtime_error(
                        "Parquet ReadTable (projected) failed for table '" +
                        table_name + "' file '" + file.path + "': " +
                        projected_status.ToString());
                }
            }
            projected_parts.push_back(std::move(part_projected));
        }

        if (projected_parts.empty()) {
            return make_empty_result();
        }

        std::shared_ptr<arrow::Table> projected;
        if (projected_parts.size() == 1) {
            projected = projected_parts.front();
        } else {
            auto maybe_projected = arrow::ConcatenateTables(projected_parts);
            if (!maybe_projected.ok()) {
                throw std::runtime_error(
                    "Arrow ConcatenateTables failed for table '" + table_name +
                    "': " + maybe_projected.status().ToString());
            }
            projected = *maybe_projected;
        }

        if (projected->num_rows() == 0) {
            return make_empty_result();
        }

        for (size_t mi = 0; mi < ordered_missing.size(); ++mi) {
            const std::string& name = ordered_missing[mi];
            const std::string cache_key = table_name + "." + name + shard_suffix;
            auto chunked = projected->column(static_cast<int>(mi));
            auto arrow_type = projected->schema()->field(static_cast<int>(mi))->type();
            const auto& positions = missing.at(name);

            // Strings: walk chunks directly into int64-offset pinned
            // buffers. Skipping Concatenate here is what lets us read
            // STRING columns whose total bytes exceed INT32_MAX (the
            // "offset overflow, consider casting to large_string"
            // failure at SF30+).
            if (arrow_type->id() == arrow::Type::STRING ||
                arrow_type->id() == arrow::Type::LARGE_STRING) {
                auto str_cache = convert_arrow_chunked_string_to_pinned(chunked);
                for (size_t pos : positions) {
                    gpu_columns[pos] =
                        string_column_from_pinned_cache(str_cache, params_.stream);
                }
                impl_->pinned_string_column_cache.emplace(cache_key, std::move(str_cache));
                continue;
            }

            auto flat_array = flatten_chunked_array(chunked);
            auto [first_gpu, pinned_cache] = arrow_array_to_gamax_column_cached(
                flat_array, arrow_type, params_.stream);
            gpu_columns[positions[0]] = std::move(first_gpu);
            for (size_t k = 1; k < positions.size(); ++k) {
                gpu_columns[positions[k]] =
                    column_from_pinned_cache(pinned_cache, params_.stream);
            }
            impl_->pinned_column_cache.emplace(cache_key, std::move(pinned_cache));
        }
    }

    cudaStreamSynchronize(params_.stream);

    return gamax::make_table(std::move(gpu_columns));
}

} // namespace gamax::tpch
