#include "tpch/tpch.hpp"
#include "utils/debug_print.hpp"

namespace gamax::tpch {

// The real implementation lives in `utils/debug_print.cu` under
// `gamax::util::debug`. This shim is kept so existing tpch callers —
// notably `PRINT_TABLE_PTR` in `tpch/tpch.hpp` and the legacy runners
// in `tpch.cu` — continue to compile without churn.
void print_table(
    const gamax::table_view& view,
    gamax::size_type nrows,
    gamax::size_type offset)
{
    gamax::util::debug::print_table(view, nrows, offset);
}

    // int32_t days_since_epoch(const std::string& date_str) {
    //     std::tm tm = {};
    //     std::istringstream ss(date_str);
    //     ss >> std::get_time(&tm, "%Y-%m-%d");
    //     if (ss.fail()) {
    //         throw std::runtime_error("Failed to parse date: " + date_str);
    //     }
    //     std::time_t time = std::mktime(&tm);
    //     return static_cast<int32_t>(time / (24 * 60 * 60));
    // }
}