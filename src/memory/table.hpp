#pragma once

#include <memory>

#include "memory/device_buffer.hpp"
#include "memory/column.hpp"
#include "types/data_type.hpp"

namespace gamax {

namespace experimental {
class table_device_view {
public:
    table_device_view() = delete;
    table_device_view(table_device_view const&) = default; // shallow copy
    table_device_view(table_device_view&&)      = default;
    __host__ __device__ ~table_device_view() {}
    
    __host__ table_device_view(std::vector<column_view> const& cols, cudaStream_t stream = cudaStreamDefault) {
        _num_columns = cols.size();
        allocate_mem(&_columns, false, _num_columns*sizeof(gamax::column_view), stream);

        for (gamax::size_type i = 0; i < _num_columns; ++i) {
            cudaMemcpyAsync(&_columns[i], &cols[i], sizeof(gamax::column_view), cudaMemcpyHostToDevice, stream);
        }
    }
    
    __host__ __device__ table_device_view(gamax::column_view* columns, gamax::size_type num_columns) 
        : _columns(columns)
        , _num_columns(num_columns) {}

    __host__ __device__ gamax::size_type num_columns() const noexcept {
        return _num_columns;
    }

    __device__ gamax::size_type num_rows() const noexcept {
        return num_columns() > 0 ? _columns[0].size() : 0;
    }

    __device__ gamax::column_view& get_column(gamax::size_type column_index) {
        return _columns[column_index];
    }

    __device__ gamax::column_view& operator[](gamax::size_type column_index) {
        return get_column(column_index);
    }

    __device__ gamax::column_view* columns() {
        return _columns;
    }

    __host__ void destroy() {
        release_mem(_columns);
    }

private:
    gamax::column_view* _columns {nullptr};
    gamax::size_type _num_columns {0};
};
} // namespace experimental

class table_view {
public:
    table_view() = default;
    ~table_view() = default;

    table_view(std::vector<std::unique_ptr<column>> const& cols) : _columns(cols.size()) {
        for (int i = 0; i < cols.size(); ++i) {
            _columns[i] = cols[i]->view();
        }
    }

    int num_columns() const noexcept {
        return _columns.size();
    }

    size_type num_rows() const noexcept {
        return num_columns() > 0 ? _columns[0].size() : 0;
    }

    column_view get_column(size_type column_index) const {
        return _columns.at(column_index);
    }

    column_view operator[](size_type column_index) const {
        return get_column(column_index);
    }

    auto device_view(cudaStream_t stream = cudaStreamDefault) {
        auto deleter = [](gamax::experimental::table_device_view* t) { t->destroy(); };
        return std::unique_ptr<gamax::experimental::table_device_view, decltype(deleter)>{
            new gamax::experimental::table_device_view(_columns, stream), deleter};
    }

private:
    std::vector<column_view> _columns {};
};

class table {
public:
    table()                        = default;
    ~table()                       = default;
    table(table&&)                 = default;  ///< Move constructor
    table& operator=(table&&)      = default;  ///< Move assignment operator
    table& operator=(table const&) = delete;
    // table& operator=(table&&)      = delete;

    table(std::vector<std::unique_ptr<column>>&& columns);

    int num_columns() const noexcept {
        return _columns.size();
    }

    gamax::size_type num_rows() const noexcept {
        return num_columns() > 0 ? _columns[0]->size() : 0;
    }

    column& get_column(gamax::size_type column_index) const { return *(_columns.at(column_index)); }

    void free_column(gamax::size_type column_index) {
        _columns.at(column_index).reset(nullptr);
    }

    void set_rows(gamax::size_type rows) {
        for (auto& col : _columns) {
            col->set_size(rows);
        }
    }

    void add_column(std::unique_ptr<column>&& col) {
        if (num_rows() != 0 && col->size() != num_rows()) {
            throw std::runtime_error("Column has different number of rows than the table");
        }
        _columns.push_back(std::move(col));
    }

    table_view view() const {
        return table_view(_columns);
    }

private:
    std::vector<std::unique_ptr<column>> _columns {};
};
}