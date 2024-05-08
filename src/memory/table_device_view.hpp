#pragma once

#include <memory>
#include <vector>
#include <cuda.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/device_buffer.hpp"
#include "utils/utils.cuh"

namespace gamax {
class table_device_view {
public:
    table_device_view() = default;
    table_device_view(table_device_view const&) = default;
    table_device_view(table_device_view&&)      = default;
    __host__ __device__ ~table_device_view() {}
    
    __host__ table_device_view(table& source_table, cudaStream_t stream = cudaStreamDefault) {
        _num_columns = source_table.num_columns();
        allocate_mem(&_columns, false, _num_columns*sizeof(gamax::column), stream);

        for (gamax::size_type i = 0; i < _num_columns; ++i) {
            cudaMemcpyAsync(&_columns[i], &source_table.get_column(i), sizeof(gamax::column), cudaMemcpyHostToDevice, stream);
        }
    }
    
    __host__ __device__ table_device_view(gamax::column* columns, gamax::size_type num_columns) 
        : _columns(columns)
        , _num_columns(num_columns) {}

    __host__ __device__ gamax::size_type num_columns() const noexcept {
        return _num_columns;
    }

    __device__ gamax::size_type num_rows() const noexcept {
        return num_columns() > 0 ? _columns[0].size() : 0;
    }

    __device__ gamax::column& get_column(gamax::size_type column_index) {
        return _columns[column_index];
    }

    __device__ gamax::column& operator[](gamax::size_type column_index) {
        return get_column(column_index);
    }

    __device__ gamax::column* columns() {
        return _columns;
    }

    __host__ void destroy() {
        release_mem(_columns);
    }

    static auto create(table& source_table, cudaStream_t stream = cudaStreamDefault) {
        auto deleter = [](table_device_view* t) { t->destroy(); };
        return std::unique_ptr<table_device_view, decltype(deleter)>{
            new table_device_view(source_table, stream), deleter};
    }

private:
    gamax::column* _columns {nullptr};
    gamax::size_type _num_columns {0};
};

}