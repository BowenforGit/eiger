#include <string>
#include <iostream>
#include <chrono>
#include <iomanip>

#include "tpch/tpch.hpp"
#include "memory/table.hpp"
#include "memory/column.hpp"
#include "string/string_column.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "utils/utils.cuh"

namespace gamax::tpch {
namespace {
    struct print_col_fn {
        template <typename T>
        void operator()(const column_view& col, gamax::size_type nrows = -1, gamax::size_type offset = 0) const {
            if constexpr (std::is_same_v<T, gamax::string_type_tag>) {
                print_string_column(col, nrows, offset);
            } else {
                gamax::size_type num_to_print = (nrows == -1) ? col.size() - offset : std::min(static_cast<gamax::size_type>(nrows), col.size() - offset);
                const T* data = col.data<T>() + offset;
                print_gpu_arr(data, num_to_print, 0);
            }
        }

        static void print_string_column(const column_view& col, gamax::size_type nrows, gamax::size_type offset) {
            gamax::string_column_view str_view(col, gamax::ARROW_STRING);
            gamax::size_type total = col.size();
            gamax::size_type num_to_print = (nrows == -1) ? total - offset : std::min(nrows, total - offset);

            // Copy offsets to host (need offset..offset+num_to_print range, i.e. num_to_print+1 entries)
            auto* dev_offsets = static_cast<const gamax::size_type*>(str_view.meta());
            std::vector<gamax::size_type> h_offsets(num_to_print + 1);
            cudaMemcpy(h_offsets.data(), dev_offsets + offset,
                       (num_to_print + 1) * sizeof(gamax::size_type), cudaMemcpyDeviceToHost);

            // Copy char data to host (only the range we need)
            gamax::size_type char_start = h_offsets[0];
            gamax::size_type char_end = h_offsets[num_to_print];
            gamax::size_type char_bytes = char_end - char_start;

            std::vector<char> h_chars(char_bytes);
            if (char_bytes > 0) {
                cudaMemcpy(h_chars.data(), str_view.data<char>() + char_start,
                           char_bytes, cudaMemcpyDeviceToHost);
            }

            // Print each string
            for (gamax::size_type i = 0; i < num_to_print; ++i) {
                gamax::size_type start = h_offsets[i] - char_start;
                gamax::size_type len = h_offsets[i + 1] - h_offsets[i];
                std::cout << "\"";
                std::cout.write(h_chars.data() + start, len);
                std::cout << "\" ";
            }
            std::cout << std::endl;
        }
    };
} // namespace

    void print_table(const gamax::table_view& view, gamax::size_type nrows, gamax::size_type offset) {
         std::cout << "Table with " << view.num_columns() << " columns and " << view.num_rows() << " rows:\n";
         for (gamax::size_type i = 0; i < view.num_columns(); ++i) {
            const auto col = view.get_column(i);
            const auto type = col.type();
            std::cout << "Col" << i << " (type: " << to_string(type) << ", size: " << col.size() << ")\n";
            // std::cout << "Col" << i << " (size: " << col.size() << ")\n";
            type_dispatcher_all(col.type(), print_col_fn{}, col, nrows, offset);
         }
        auto num_cols = view.num_columns();
        
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