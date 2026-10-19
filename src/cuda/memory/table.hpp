// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>

#include "memory/device_buffer.hpp"
#include "memory/column.hpp"
#include "types/data_type.hpp"

namespace gamax {

namespace experimental {
template <typename column_view_t>
class table_device_view_base {
public:
    using column_view = column_view_t;

    table_device_view_base() = default;
    table_device_view_base(table_device_view_base const&) = default; // shallow copy
    table_device_view_base(table_device_view_base&&)      = default;
    ~table_device_view_base() = default;
    
    __host__ __device__ table_device_view_base(column_view_t* columns, gamax::size_type num_columns) 
        : _columns(columns)
        , _num_columns(num_columns) {}

    __host__ __device__ gamax::size_type num_columns() const noexcept {
        return _num_columns;
    }

   __host__  __device__ gamax::size_type num_rows() const noexcept {
        return num_columns() > 0 ? _columns[0].size() : 0;
    }

    __host__ __device__ column_view_t& get_column(gamax::size_type column_index) {
        return _columns[column_index];
    }

    __host__ __device__ column_view_t const& get_column(gamax::size_type column_index) const {
        return _columns[column_index];
    }

    __host__ __device__ column_view_t& operator[](gamax::size_type column_index) {
        return get_column(column_index);
    }

    __host__ __device__ column_view_t const& operator[](gamax::size_type column_index) const {
        return get_column(column_index);
    }

    __host__ __device__ column_view_t* columns() {
        return _columns;
    }

private:
    column_view_t* _columns {nullptr};
    gamax::size_type _num_columns {0};
};

using table_device_view = table_device_view_base<gamax::column_view>;
using mutable_table_device_view = table_device_view_base<gamax::mutable_column_view>;
} // namespace experimental

template <typename column_view_t>
class table_device_view_owner {
public:
    using view_type = experimental::table_device_view_base<column_view_t>;

    table_device_view_owner(std::vector<column_view_t> const& columns, cudaStream_t stream)
        : storage_{columns.size() * sizeof(column_view_t), stream}
        , view_{static_cast<column_view_t*>(storage_.data()),
                static_cast<gamax::size_type>(columns.size())}
    {
        if (!columns.empty()) {
            // `columns` is pageable std::vector storage. CUDA stages pageable
            // H2D input before returning, which is safe but may synchronize the
            // host and cannot provide true copy/compute overlap.
            auto const error = cudaMemcpyAsync(
                storage_.data(), columns.data(), storage_.size(),
                cudaMemcpyHostToDevice, stream);
            if (error != cudaSuccess) {
                throw std::runtime_error(
                    std::string("table device-view metadata upload failed: ") +
                    cudaGetErrorString(error));
            }
        }
    }

    table_device_view_owner(table_device_view_owner const&) = delete;
    table_device_view_owner& operator=(table_device_view_owner const&) = delete;
    table_device_view_owner(table_device_view_owner&&) = default;
    table_device_view_owner& operator=(table_device_view_owner&&) = default;

    [[nodiscard]] view_type const& operator*() const noexcept { return view_; }
    [[nodiscard]] view_type& operator*() noexcept { return view_; }
    [[nodiscard]] view_type const* operator->() const noexcept { return &view_; }
    [[nodiscard]] view_type* operator->() noexcept { return &view_; }

private:
    device_buffer storage_;
    view_type view_;
};

template<typename column_view_t>
class table_view_base {
public:
    // Host-side collection of borrowed column views. It owns no column data;
    // all source columns must outlive asynchronous consumers of this view.
    using column_view = column_view_t;
    using device_view_t = experimental::table_device_view_base<column_view_t>;

    __host__ table_view_base() = default;
    __host__ ~table_view_base() = default;

    __host__ table_view_base(std::vector<std::unique_ptr<column>> const& cols) : _columns(cols.size()) {
        for (gamax::size_type i = 0; i < cols.size(); ++i) {
            if constexpr (std::is_const_v<std::remove_pointer_t<decltype(std::declval<column_view_t>().data())>>) {
                _columns[i] = cols[i]->view();
            } else {
                _columns[i] = cols[i]->mutable_view();
            }
        }
    }

    __host__ table_view_base(std::vector<column_view_t> const& cols) : _columns(cols) {}

    gamax::size_type num_columns() const noexcept {
        return _columns.size();
    }

    size_type num_rows() const noexcept {
        return num_columns() > 0 ? _columns[0].size() : 0;
    }

    column_view_t get_column(size_type column_index) const {
        return _columns.at(column_index);
    }

    column_view_t operator[](size_type column_index) const {
        return get_column(column_index);
    }

    auto device_view(cudaStream_t stream) {
        return table_device_view_owner<column_view_t>{_columns, stream};
    }

    auto select_columns(std::vector<size_type> const& column_indices) const {
        std::vector<column_view_t> selected_columns(column_indices.size());
        for (size_type i = 0; i < column_indices.size(); ++i) {
            selected_columns[i] = _columns[column_indices[i]];
        }
        return table_view_base<column_view_t>(selected_columns);
    }

    void add_column(column_view_t new_col) {
        if (new_col.size() != num_rows()) {
            throw std::runtime_error("New column must have the same number of rows as the table");
        }
        _columns.push_back(new_col);
    }

private:
    std::vector<column_view_t> _columns {};
};

using table_view = table_view_base<column_view>;
using mutable_table_view = table_view_base<mutable_column_view>;

class table {
public:
    table()                        = default;
    ~table()                       = default;
    table(table&&)                 = default;  ///< Move constructor
    table& operator=(table&&)      = default;  ///< Move assignment operator
    table& operator=(table const&) = delete;
    // table& operator=(table&&)      = delete;

    table(std::vector<std::unique_ptr<column>>&& columns);

    gamax::size_type num_columns() const noexcept {
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

    std::unique_ptr<column> remove_column(gamax::size_type column_index)
    {
        if (column_index < 0 || column_index >= _columns.size())
        {
            throw std::invalid_argument("column out of range");
        }

        std::unique_ptr<column> col = std::move(_columns[column_index]);
        _columns.erase(_columns.begin() + column_index);
        return col;
    }

    /**
     * @brief Release ownership of a column without removing it from the table.
     * 
     * Unlike remove_column, this method leaves a nullptr in the column slot,
     * preserving indices for subsequent operations. Useful when processing
     * expressions that may reference multiple columns by index.
     *
     * @param column_index The index of the column to release
     * @return std::unique_ptr<column> The released column
     */
    std::unique_ptr<column> release_column(gamax::size_type column_index)
    {
        if (column_index < 0 || column_index >= _columns.size())
        {
            throw std::invalid_argument("column out of range");
        }
        return std::move(_columns[column_index]);
    }

    table_view view() const {
        return table_view(_columns);
    }

    mutable_table_view mutable_view() const {
        return mutable_table_view(_columns);
    }

private:
    std::vector<std::unique_ptr<column>> _columns {};
};

std::unique_ptr<table> make_table(std::vector<std::unique_ptr<column>>&& columns);
}
