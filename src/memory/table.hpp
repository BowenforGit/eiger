#pragma once

#include <memory>

#include "memory/device_buffer.hpp"
#include "memory/column.hpp"
#include "types/data_type.hpp"

namespace gamax {
class table {
public:
    table()                        = default;
    ~table()                       = default;
    table(table&&)                 = default;  ///< Move constructor
    table& operator=(table const&) = delete;
    table& operator=(table&&)      = delete;

    table(std::vector<std::unique_ptr<column>>&& columns);

    int num_columns() const noexcept {
        return _columns.size();
    }

    gamax::size_type num_rows() const noexcept {
        return num_columns() > 0 ? _columns[0]->size() : 0;
    }

    column& get_column(gamax::size_type column_index) { return *(_columns.at(column_index)); }

    void free_column(gamax::size_type column_index) {
        _columns.at(column_index).reset(nullptr);
    }

    void set_rows(gamax::size_type rows) {
        for (auto& col : _columns) {
            col->set_size(rows);
        }
    }

private:
    std::vector<std::unique_ptr<column>> _columns {};
};
}