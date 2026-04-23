#include "tpch/tpch.hpp"

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
#include <parquet/arrow/reader.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string_column.hpp"
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
    gamax::device_buffer buf(cache.byte_size, false, stream);
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

// Convert Arrow STRING array → pinned cache (widens int32 offsets to int64)
PinnedStringColumnData convert_arrow_string_to_pinned(
    const std::shared_ptr<arrow::Array>& array)
{
    auto str_array = std::static_pointer_cast<arrow::StringArray>(array);
    auto num_rows = static_cast<gamax::size_type>(str_array->length());

    // Offsets: Arrow has int32, gamax needs int64. Widen.
    size_t offsets_byte_size = static_cast<size_t>(num_rows + 1) * sizeof(int64_t);
    void* pinned_offsets = nullptr;
    cudaMallocHost(&pinned_offsets, offsets_byte_size);
    auto* dst_offsets = static_cast<int64_t*>(pinned_offsets);
    const int32_t* src_offsets = str_array->raw_value_offsets();
    for (gamax::size_type i = 0; i <= num_rows; ++i) {
        dst_offsets[i] = static_cast<int64_t>(src_offsets[i]);
    }

    // Data: direct copy of the flat char buffer
    size_t data_byte_size = static_cast<size_t>(dst_offsets[num_rows]);
    void* pinned_data = nullptr;
    if (data_byte_size > 0) {
        cudaMallocHost(&pinned_data, data_byte_size);
        std::memcpy(pinned_data, str_array->value_data()->data(), data_byte_size);
    }

    return PinnedStringColumnData(
        pinned_offsets, pinned_data, num_rows,
        offsets_byte_size, data_byte_size);
}

// Convert Arrow LARGE_STRING array → pinned cache (offsets already int64)
PinnedStringColumnData convert_arrow_large_string_to_pinned(
    const std::shared_ptr<arrow::Array>& array)
{
    auto str_array = std::static_pointer_cast<arrow::LargeStringArray>(array);
    auto num_rows = static_cast<gamax::size_type>(str_array->length());

    // Offsets: already int64, direct copy
    size_t offsets_byte_size = static_cast<size_t>(num_rows + 1) * sizeof(int64_t);
    void* pinned_offsets = nullptr;
    cudaMallocHost(&pinned_offsets, offsets_byte_size);
    std::memcpy(pinned_offsets, str_array->raw_value_offsets(), offsets_byte_size);

    // Data: direct copy
    const int64_t* src_offsets = str_array->raw_value_offsets();
    size_t data_byte_size = static_cast<size_t>(src_offsets[num_rows]);
    void* pinned_data = nullptr;
    if (data_byte_size > 0) {
        cudaMallocHost(&pinned_data, data_byte_size);
        std::memcpy(pinned_data, str_array->value_data()->data(), data_byte_size);
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
    gamax::device_buffer meta_buf(cache.offsets_byte_size, false, stream);
    cudaMemcpyAsync(meta_buf.data(), cache.pinned_offsets, cache.offsets_byte_size,
                    cudaMemcpyHostToDevice, stream);

    // Data → device data buffer
    gamax::device_buffer data_buf(
        cache.data_byte_size > 0 ? cache.data_byte_size : 1, false, stream);
    if (cache.data_byte_size > 0) {
        cudaMemcpyAsync(data_buf.data(), cache.pinned_data, cache.data_byte_size,
                        cudaMemcpyHostToDevice, stream);
    }

    return std::make_unique<gamax::string_column>(
        std::move(data_buf), std::move(meta_buf),
        cache.num_rows, gamax::string_impl_id::ARROW_STRING);
}

// ---------------------------------------------------------------------------
// Read a Parquet file into an Arrow Table
// ---------------------------------------------------------------------------
std::shared_ptr<arrow::Table> read_parquet_file(const std::string& file_path) {
    auto maybe_infile = arrow::io::ReadableFile::Open(file_path);
    if (!maybe_infile.ok()) {
        throw std::runtime_error(
            "Failed to open Parquet file '" + file_path + "': " +
            maybe_infile.status().ToString());
    }
    auto infile = *maybe_infile;

    auto maybe_reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
    if (!maybe_reader.ok()) {
        throw std::runtime_error(
            "Failed to create Parquet reader for '" + file_path + "': " +
            maybe_reader.status().ToString());
    }
    auto& reader = *maybe_reader;

    std::shared_ptr<arrow::Table> table;
    auto status = reader->ReadTable(&table);
    if (!status.ok()) {
        throw std::runtime_error(
            "Failed to read Parquet file '" + file_path + "': " +
            status.ToString());
    }

    return table;
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
// TpchTableReader::Impl — holds the Arrow Table cache + pinned column cache
// ---------------------------------------------------------------------------
struct TpchTableReader::Impl {
    std::unordered_map<std::string, std::shared_ptr<arrow::Table>> table_cache;
    std::unordered_map<std::string, PinnedColumnData> pinned_column_cache;
    std::unordered_map<std::string, PinnedStringColumnData> pinned_string_column_cache;
};

// ---------------------------------------------------------------------------
// TpchTableReader constructors / destructor / move
// ---------------------------------------------------------------------------
TpchTableReader::TpchTableReader(const tpch_params& params)
    : params_(params)
    , impl_(std::make_unique<Impl>()) {}

TpchTableReader::~TpchTableReader() = default;

TpchTableReader::TpchTableReader(TpchTableReader&&) noexcept = default;
TpchTableReader& TpchTableReader::operator=(TpchTableReader&&) noexcept = default;

// ---------------------------------------------------------------------------
// TpchTableReader::load
// ---------------------------------------------------------------------------
std::unique_ptr<gamax::table> TpchTableReader::load(
    const std::string& table_name,
    std::vector<std::string> column_names)
{
    std::string data_dir = get_tpch_data_dir(params_.scale_factor);

    // Ensure Arrow table is cached (needed for cold-path column conversion)
    auto table_cache_it = impl_->table_cache.find(table_name);
    if (table_cache_it == impl_->table_cache.end()) {
        std::string file_path = data_dir + "/" + table_name + ".parquet";
        std::cout << "[TpchTableReader] Loading Parquet file: " << file_path << std::endl;
        auto arrow_table = read_parquet_file(file_path);
        std::cout << "[TpchTableReader] Loaded table '" << table_name
                  << "': " << arrow_table->num_rows() << " rows, "
                  << arrow_table->num_columns() << " columns" << std::endl;
        table_cache_it = impl_->table_cache.emplace(table_name, std::move(arrow_table)).first;
    }

    const auto& arrow_table = table_cache_it->second;

    std::vector<std::unique_ptr<gamax::column>> gpu_columns;
    gpu_columns.reserve(column_names.size());

    for (const auto& col_name : column_names) {
        std::string cache_key = table_name + "." + col_name;

        // Hot path: check scalar column cache first, then string column cache
        auto pinned_it = impl_->pinned_column_cache.find(cache_key);
        if (pinned_it != impl_->pinned_column_cache.end()) {
            gpu_columns.push_back(column_from_pinned_cache(pinned_it->second, params_.stream));
            continue;
        }
        auto str_pinned_it = impl_->pinned_string_column_cache.find(cache_key);
        if (str_pinned_it != impl_->pinned_string_column_cache.end()) {
            gpu_columns.push_back(string_column_from_pinned_cache(str_pinned_it->second, params_.stream));
            continue;
        }

        // Cold path: convert from Arrow and populate pinned cache
        int col_idx = arrow_table->schema()->GetFieldIndex(col_name);
        if (col_idx < 0) {
            std::string available;
            for (int i = 0; i < arrow_table->schema()->num_fields(); ++i) {
                if (i > 0) available += ", ";
                available += arrow_table->schema()->field(i)->name();
            }
            throw std::runtime_error(
                "Column '" + col_name + "' not found in table '" + table_name + "'. "
                "Available columns: " + available);
        }

        auto arrow_col = arrow_table->column(col_idx);
        auto arrow_type = arrow_table->schema()->field(col_idx)->type();
        auto flat_array = flatten_chunked_array(arrow_col);

        // String columns use a separate cache and produce string_column objects
        if (arrow_type->id() == arrow::Type::STRING) {
            auto str_cache = convert_arrow_string_to_pinned(flat_array);
            auto gpu_col = string_column_from_pinned_cache(str_cache, params_.stream);
            impl_->pinned_string_column_cache.emplace(cache_key, std::move(str_cache));
            gpu_columns.push_back(std::move(gpu_col));
            continue;
        }
        if (arrow_type->id() == arrow::Type::LARGE_STRING) {
            auto str_cache = convert_arrow_large_string_to_pinned(flat_array);
            auto gpu_col = string_column_from_pinned_cache(str_cache, params_.stream);
            impl_->pinned_string_column_cache.emplace(cache_key, std::move(str_cache));
            gpu_columns.push_back(std::move(gpu_col));
            continue;
        }

        auto [gpu_col, pinned_cache] = arrow_array_to_gamax_column_cached(
            flat_array, arrow_type, params_.stream);

        impl_->pinned_column_cache.emplace(cache_key, std::move(pinned_cache));
        gpu_columns.push_back(std::move(gpu_col));
    }

    cudaStreamSynchronize(params_.stream);

    return gamax::make_table(std::move(gpu_columns));
}

} // namespace gamax::tpch
