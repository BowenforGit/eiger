#include "tpch/verify.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/data_type.hpp"

namespace gamax::tpch {
namespace {

// Load the Parquet file at `path` as an Arrow table. Mirrors the helper
// already present in `tpch_reader.cpp`'s anonymous namespace; kept as
// a small local copy so this module doesn't need to open a public API
// in the reader.
std::shared_ptr<arrow::Table> read_parquet(const std::string& path) {
    auto maybe_in = arrow::io::ReadableFile::Open(path);
    if (!maybe_in.ok()) {
        throw std::runtime_error(
            "verify: open('" + path + "'): " + maybe_in.status().ToString());
    }
    auto maybe_reader = parquet::arrow::OpenFile(
        *maybe_in, arrow::default_memory_pool());
    if (!maybe_reader.ok()) {
        throw std::runtime_error(
            "verify: parquet reader('" + path + "'): " +
            maybe_reader.status().ToString());
    }
    std::shared_ptr<arrow::Table> tbl;
    auto st = (*maybe_reader)->ReadTable(&tbl);
    if (!st.ok()) {
        throw std::runtime_error(
            "verify: read('" + path + "'): " + st.ToString());
    }
    return tbl;
}

// Concatenate a ChunkedArray down to a single Array — simplifies
// Value(i) access for all downstream comparison logic. Duplicates the
// helper in `tpch_reader.cpp` so `verify` has no coupling to that TU's
// anonymous namespace.
std::shared_ptr<arrow::Array> flatten(
    const std::shared_ptr<arrow::ChunkedArray>& ca)
{
    if (ca->num_chunks() == 1) return ca->chunk(0);
    auto r = arrow::Concatenate(ca->chunks());
    if (!r.ok()) {
        throw std::runtime_error(
            "verify: concatenate chunks: " + r.status().ToString());
    }
    return *r;
}

// Map Arrow type ID → expected gamax type ID. Returns `NOT_SUPPORTED`
// for anything outside the set the goldens emit (int32/int64/float64/
// string). Callers inspect and report a clearer error than a raw
// static_cast would give.
gamax::type_id arrow_to_gamax_type(arrow::Type::type a) {
    switch (a) {
        case arrow::Type::INT32:   return gamax::type_id::INT32;
        case arrow::Type::INT64:   return gamax::type_id::INT64;
        case arrow::Type::DOUBLE:  return gamax::type_id::FLOAT64;
        case arrow::Type::FLOAT:   return gamax::type_id::FLOAT32;
        case arrow::Type::BOOL:    return gamax::type_id::BOOL8;
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
            return gamax::type_id::STRING;
        default:                   return gamax::type_id::EMPTY;
    }
}

const char* type_name(gamax::type_id t) {
    switch (t) {
        case gamax::type_id::INT8:    return "INT8";
        case gamax::type_id::INT16:   return "INT16";
        case gamax::type_id::INT32:   return "INT32";
        case gamax::type_id::INT64:   return "INT64";
        case gamax::type_id::UINT8:   return "UINT8";
        case gamax::type_id::UINT16:  return "UINT16";
        case gamax::type_id::UINT32:  return "UINT32";
        case gamax::type_id::UINT64:  return "UINT64";
        case gamax::type_id::FLOAT32: return "FLOAT32";
        case gamax::type_id::FLOAT64: return "FLOAT64";
        case gamax::type_id::BOOL8:   return "BOOL8";
        case gamax::type_id::STRING:  return "STRING";
        default:                      return "?";
    }
}

bool floats_close(double a, double b, double abs_tol, double rel_tol) {
    if (std::isnan(a) && std::isnan(b)) return true;
    double diff = std::abs(a - b);
    double tol = std::max(abs_tol, rel_tol * std::abs(b));
    return diff <= tol;
}

// Gather a STRING gamax column to host via the [data_ptr, meta_ptr]
// helper buffer (same technique `run_tpch_ir::print_result` uses;
// sidesteps the NVCC void*-arithmetic codegen quirk in
// `string_column_view`'s host ctor).
std::vector<std::string> gather_strings_to_host(gamax::column_view col) {
    void* host_ptrs[2] = {nullptr, nullptr};
    cudaMemcpy(host_ptrs, col.data<void>(),
               2 * sizeof(void*), cudaMemcpyDeviceToHost);
    auto n = col.size();
    std::vector<gamax::size_type> offsets(n + 1);
    cudaMemcpy(offsets.data(), host_ptrs[1],
               (n + 1) * sizeof(gamax::size_type),
               cudaMemcpyDeviceToHost);
    const auto total = offsets.back();
    std::string bytes(total, '\0');
    if (total > 0) {
        cudaMemcpy(bytes.data(), host_ptrs[0], total,
                   cudaMemcpyDeviceToHost);
    }
    std::vector<std::string> out;
    out.reserve(n);
    for (gamax::size_type i = 0; i < n; ++i) {
        out.emplace_back(bytes.data() + offsets[i],
                         offsets[i + 1] - offsets[i]);
    }
    return out;
}

std::string compare_strings(
    gamax::column_view got_col,
    const arrow::Array& expected_any,
    int col_idx,
    std::size_t max_diffs)
{
    auto got = gather_strings_to_host(got_col);
    const auto n = got_col.size();
    std::size_t n_diffs = 0;
    std::ostringstream os;
    auto get_expected = [&](gamax::size_type i) -> std::string {
        if (expected_any.type_id() == arrow::Type::LARGE_STRING) {
            const auto& a = static_cast<const arrow::LargeStringArray&>(expected_any);
            return a.GetString(i);
        }
        const auto& a = static_cast<const arrow::StringArray&>(expected_any);
        return a.GetString(i);
    };
    for (gamax::size_type i = 0; i < n; ++i) {
        auto e = get_expected(i);
        if (got[i] != e) {
            if (n_diffs < max_diffs) {
                os << "  col=" << col_idx << " row=" << i
                   << " got=\"" << got[i] << "\""
                   << " expected=\"" << e << "\"\n";
            }
            ++n_diffs;
        }
    }
    if (n_diffs == 0) return {};
    std::ostringstream hdr;
    hdr << "string mismatch on col " << col_idx << " ("
        << n_diffs << " of " << n << " rows differ)\n" << os.str();
    return hdr.str();
}

} // namespace

VerifyResult verify_result(
    gamax::table_view got,
    const std::string& golden_path,
    const VerifyOptions& opts)
{
    VerifyResult r;

    std::shared_ptr<arrow::Table> golden;
    try {
        golden = read_parquet(golden_path);
    } catch (const std::exception& e) {
        r.diagnostic = std::string("could not load golden: ") + e.what();
        return r;
    }

    if (got.num_columns() != golden->num_columns()) {
        std::ostringstream os;
        os << "column count mismatch: got=" << got.num_columns()
           << " expected=" << golden->num_columns();
        r.diagnostic = os.str();
        return r;
    }

    const auto got_rows = got.num_rows();
    const auto golden_rows = static_cast<gamax::size_type>(golden->num_rows());
    if (got_rows != golden_rows) {
        std::ostringstream os;
        os << "row count mismatch: got=" << got_rows
           << " expected=" << golden_rows;
        r.diagnostic = os.str();
        return r;
    }

    std::ostringstream errs;
    bool any_bad = false;
    for (int c = 0; c < got.num_columns(); ++c) {
        auto got_col = got[c];
        auto chunked = golden->column(c);
        auto flat = flatten(chunked);
        auto expected_type = arrow_to_gamax_type(flat->type_id());
        auto got_type = got_col.type().id();

        // Build a canonical "compare kind" from the got-side gamax
        // type. Widening pairs (INT32 got vs INT64 golden, INT vs
        // FLOAT64 golden) fall under the wider kind — we widen gamax's
        // host copy into the compared type rather than insist the
        // exact Arrow/gamax types align. This prevents noise for
        // queries like Q7/Q8/Q9 (`year(date)` → INT32 in gamax, INT64
        // in DuckDB's `EXTRACT(YEAR FROM date)`) and Q12 (`SUM(case…)`
        // over INT literals → INT64 in gamax, DECIMAL→DOUBLE in
        // DuckDB) without hiding genuine value drift.
        enum class Kind { I32, I64, F64, STR, UNSUPPORTED };
        auto kind_of = [](gamax::type_id t) {
            switch (t) {
                case gamax::type_id::INT8:
                case gamax::type_id::INT16:
                case gamax::type_id::INT32:   return Kind::I32;
                case gamax::type_id::INT64:   return Kind::I64;
                case gamax::type_id::FLOAT32:
                case gamax::type_id::FLOAT64: return Kind::F64;
                case gamax::type_id::STRING:  return Kind::STR;
                default:                      return Kind::UNSUPPORTED;
            }
        };
        Kind got_kind = kind_of(got_type);
        Kind exp_kind = kind_of(expected_type);

        // Compatibility table:
        //   got=I32  → expected in {I32, I64, F64}     (widen on the fly)
        //   got=I64  → expected in {I64, F64}          (widen on the fly)
        //   got=F64  → expected in {F64, I64, I32}     (widen golden to F64)
        //   got=STR  → expected == STR
        // Anything else is a real type clash and we flag it.
        bool compatible = false;
        if (got_kind == exp_kind) compatible = true;
        else if (got_kind == Kind::I32 &&
                 (exp_kind == Kind::I64 || exp_kind == Kind::F64))
            compatible = true;
        else if (got_kind == Kind::I64 && exp_kind == Kind::F64)
            compatible = true;
        else if (got_kind == Kind::F64 &&
                 (exp_kind == Kind::I32 || exp_kind == Kind::I64))
            compatible = true;

        if (!compatible) {
            errs << "type mismatch on col " << c
                 << ": got=" << type_name(got_type)
                 << " expected(arrow=" << flat->type()->ToString()
                 << " → " << type_name(expected_type) << ")\n";
            any_bad = true;
            continue;
        }

        // Dispatch on the PAIR (got_kind, exp_kind). STRING only
        // compares STRING→STRING; numeric pairs go through either an
        // exact integer path or the float-tolerance path, widening as
        // needed.
        std::string diag;
        if (got_kind == Kind::STR) {
            diag = compare_strings(got_col, *flat, c, opts.max_diffs);
        } else if (exp_kind == Kind::F64 || got_kind == Kind::F64) {
            // Float comparison with tolerance; widen the integer side.
            const auto n = got_col.size();
            std::vector<double> got_host(n);
            if (got_kind == Kind::I32) {
                auto h = gamax::copy_column_to_host<int32_t>(got_col);
                for (gamax::size_type i = 0; i < n; ++i)
                    got_host[i] = static_cast<double>(h[i]);
            } else if (got_kind == Kind::I64) {
                auto h = gamax::copy_column_to_host<int64_t>(got_col);
                for (gamax::size_type i = 0; i < n; ++i)
                    got_host[i] = static_cast<double>(h[i]);
            } else {
                auto h = gamax::copy_column_to_host<double>(got_col);
                for (gamax::size_type i = 0; i < n; ++i)
                    got_host[i] = h[i];
            }
            auto get_expected = [&](gamax::size_type i) -> double {
                if (exp_kind == Kind::I32) {
                    return static_cast<double>(
                        static_cast<const arrow::Int32Array&>(*flat).Value(i));
                }
                if (exp_kind == Kind::I64) {
                    return static_cast<double>(
                        static_cast<const arrow::Int64Array&>(*flat).Value(i));
                }
                return static_cast<const arrow::DoubleArray&>(*flat).Value(i);
            };
            std::size_t n_diffs = 0;
            std::ostringstream os;
            for (gamax::size_type i = 0; i < n; ++i) {
                double g = got_host[i];
                double e = get_expected(i);
                if (!floats_close(g, e, opts.abs_tol, opts.rel_tol)) {
                    if (n_diffs < opts.max_diffs) {
                        os << "  col=" << c << " row=" << i
                           << " got=" << g << " expected=" << e
                           << " (diff=" << (g - e) << ")\n";
                    }
                    ++n_diffs;
                }
            }
            if (n_diffs > 0) {
                std::ostringstream hdr;
                hdr << "float mismatch on col " << c << " ("
                    << n_diffs << " of " << n << " rows differ)\n" << os.str();
                diag = hdr.str();
            }
        } else {
            // Both sides are integer; compare in the widest kind so
            // INT32 got against INT64 golden just widens.
            auto host32 = (got_kind == Kind::I32)
                ? gamax::copy_column_to_host<int32_t>(got_col)
                : std::unique_ptr<int32_t[]>(nullptr);
            auto host64 = (got_kind == Kind::I64)
                ? gamax::copy_column_to_host<int64_t>(got_col)
                : std::unique_ptr<int64_t[]>(nullptr);
            auto got_i = [&](gamax::size_type i) -> int64_t {
                return got_kind == Kind::I32
                    ? static_cast<int64_t>(host32[i])
                    : host64[i];
            };
            auto exp_i = [&](gamax::size_type i) -> int64_t {
                if (exp_kind == Kind::I32)
                    return static_cast<int64_t>(
                        static_cast<const arrow::Int32Array&>(*flat).Value(i));
                return static_cast<const arrow::Int64Array&>(*flat).Value(i);
            };
            const auto n = got_col.size();
            std::size_t n_diffs = 0;
            std::ostringstream os;
            for (gamax::size_type i = 0; i < n; ++i) {
                int64_t g = got_i(i);
                int64_t e = exp_i(i);
                if (g != e) {
                    if (n_diffs < opts.max_diffs) {
                        os << "  col=" << c << " row=" << i
                           << " got=" << g << " expected=" << e << "\n";
                    }
                    ++n_diffs;
                }
            }
            if (n_diffs > 0) {
                std::ostringstream hdr;
                hdr << "mismatch on col " << c << " ("
                    << n_diffs << " of " << n << " rows differ)\n" << os.str();
                diag = hdr.str();
            }
        }
        if (!diag.empty()) {
            errs << diag;
            any_bad = true;
        }
    }

    if (any_bad) {
        r.diagnostic = errs.str();
        return r;
    }
    r.ok = true;
    return r;
}

} // namespace gamax::tpch
