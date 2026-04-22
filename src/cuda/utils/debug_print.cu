#include "utils/debug_print.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "exec/exec_plan.hpp"
#include "exec/print.hpp"
#include "ir/plan_node.hpp"
#include "ir/print.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string_column.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "utils/utils.cuh"

namespace gamax::util::debug {
namespace {

// Per-column printer. Numeric columns go through `print_gpu_arr`;
// STRING columns are routed through a dedicated branch that copies the
// relevant offset/char ranges to host and writes quoted literals.
struct print_col_fn {
    std::ostream* os;

    template <typename T>
    void operator()(
        const gamax::column_view& col,
        gamax::size_type nrows = -1,
        gamax::size_type offset = 0) const
    {
        if constexpr (std::is_same_v<T, gamax::string_type_tag>) {
            print_string_column(col, nrows, offset, *os);
        } else {
            gamax::size_type num_to_print = (nrows == -1)
                ? col.size() - offset
                : std::min(nrows, col.size() - offset);
            const T* data = col.data<T>() + offset;
            // print_gpu_arr writes to std::cout; call it for the common
            // case, otherwise fall back to a manual host copy so the
            // caller's stream is honoured. `std::vector<bool>` is a bit
            // special (packed bits, no contiguous storage), so copy
            // into a byte buffer and reinterpret on print.
            if (os == &std::cout) {
                print_gpu_arr(data, num_to_print, 0);
            } else {
                std::vector<unsigned char> raw(num_to_print * sizeof(T));
                cudaMemcpy(raw.data(), data, num_to_print * sizeof(T),
                           cudaMemcpyDeviceToHost);
                const T* h = reinterpret_cast<const T*>(raw.data());
                for (gamax::size_type i = 0; i < num_to_print; ++i) {
                    if (i) *os << " ";
                    if constexpr (std::is_same_v<T, bool>) {
                        *os << (h[i] ? "true" : "false");
                    } else {
                        *os << h[i];
                    }
                }
                *os << "\n";
            }
        }
    }

    static void print_string_column(
        const gamax::column_view& col,
        gamax::size_type nrows,
        gamax::size_type offset,
        std::ostream& os)
    {
        gamax::string_column_view str_view(col, gamax::ARROW_STRING);
        gamax::size_type total = col.size();
        gamax::size_type num_to_print = (nrows == -1)
            ? total - offset
            : std::min(nrows, total - offset);
        if (num_to_print <= 0) {
            os << "\n";
            return;
        }

        auto* dev_offsets = static_cast<const gamax::size_type*>(str_view.meta());
        std::vector<gamax::size_type> h_offsets(num_to_print + 1);
        cudaMemcpy(h_offsets.data(), dev_offsets + offset,
                   (num_to_print + 1) * sizeof(gamax::size_type),
                   cudaMemcpyDeviceToHost);

        gamax::size_type char_start = h_offsets[0];
        gamax::size_type char_end   = h_offsets[num_to_print];
        gamax::size_type char_bytes = char_end - char_start;

        std::vector<char> h_chars(char_bytes);
        if (char_bytes > 0) {
            cudaMemcpy(h_chars.data(), str_view.data<char>() + char_start,
                       char_bytes, cudaMemcpyDeviceToHost);
        }

        for (gamax::size_type i = 0; i < num_to_print; ++i) {
            gamax::size_type start = h_offsets[i] - char_start;
            gamax::size_type len   = h_offsets[i + 1] - h_offsets[i];
            os << "\"";
            os.write(h_chars.data() + start, len);
            os << "\" ";
        }
        os << "\n";
    }
};

// Trim + lower-case a token for case-insensitive env-var handling.
std::string normalize(std::string s) {
    auto is_space = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    };
    while (!s.empty() && is_space(s.front())) s.erase(s.begin());
    while (!s.empty() && is_space(s.back()))  s.pop_back();
    for (auto& c : s) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    }
    return s;
}

} // namespace

void print_table(
    const gamax::table_view& view,
    gamax::size_type nrows,
    gamax::size_type offset,
    std::ostream& os)
{
    os << "Table with " << view.num_columns()
       << " columns and " << view.num_rows() << " rows:\n";
    for (gamax::size_type i = 0; i < view.num_columns(); ++i) {
        const auto col = view.get_column(i);
        const auto type = col.type();
        os << "Col" << i << " (type: " << to_string(type)
           << ", size: " << col.size() << ")\n";
        gamax::type_dispatcher_all(
            col.type(), print_col_fn{&os}, col, nrows, offset);
    }
}

void print_table(
    const gamax::table_view& view,
    gamax::size_type nrows,
    gamax::size_type offset)
{
    print_table(view, nrows, offset, std::cout);
}

PhysDebugFilter::PhysDebugFilter() {
    if (const char* raw = std::getenv("GAMAX_PHYS_DEBUG")) {
        std::string s(raw);
        if (!s.empty()) {
            std::stringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                tok = normalize(std::move(tok));
                if (tok.empty()) continue;
                if (tok == "all" || tok == "*") {
                    match_all_   = true;
                    any_enabled_ = true;
                } else {
                    kinds_.insert(std::move(tok));
                    any_enabled_ = true;
                }
            }
        }
    }
    if (const char* raw = std::getenv("GAMAX_PHYS_DEBUG_ROWS")) {
        try {
            max_rows_ = static_cast<gamax::size_type>(std::stoll(raw));
        } catch (...) {
            // Ignore malformed value; keep default.
        }
    }
}

bool PhysDebugFilter::should_print(const std::string& kind_name) const {
    if (!any_enabled_) return false;
    if (match_all_)    return true;
    return kinds_.count(kind_name) != 0;
}

namespace {

// Gate for the plan-dump: triggered by a non-empty `GAMAX_PHYS_DEBUG`
// (overrides GAMAX_DEBUG_PLAN) or by GAMAX_DEBUG_PLAN=1.
bool plan_debug_enabled() {
    if (const char* raw = std::getenv("GAMAX_PHYS_DEBUG")) {
        if (*raw != '\0') return true;  // non-empty ⇒ on
    }
    if (const char* raw = std::getenv("GAMAX_DEBUG_PLAN")) {
        return std::string(raw) == "1";
    }
    return false;
}

} // namespace

void print_plans_if_debug(
    const gamax::ir::PlanNode& pre_rules,
    const gamax::ir::PlanNode& post_rules,
    const gamax::exec::ExecPlan& ep,
    std::ostream& os)
{
    if (!plan_debug_enabled()) return;

    os << "\n-- IR (pre-rules) --\n"
       << gamax::ir::format_plan(pre_rules) << "\n";
    os << "\n-- IR (post-rules) --\n"
       << gamax::ir::format_plan(post_rules) << "\n";
    os << "\n-- ExecPlan --\n"
       << gamax::exec::format_exec_plan(ep) << "\n";
    os.flush();
}

void print_plans_if_debug(
    const gamax::ir::PlanNode& pre_rules,
    const gamax::ir::PlanNode& post_rules,
    const gamax::exec::ExecPlan& ep)
{
    print_plans_if_debug(pre_rules, post_rules, ep, std::cerr);
}

} // namespace gamax::util::debug
