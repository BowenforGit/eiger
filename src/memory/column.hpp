#pragma once

#include <memory>

#include "memory/device_buffer.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

namespace gamax {
class column {
public:
    column() = default;
    ~column() = default;
    column& operator=(column const& other) = delete;
    column(column const& other)            = delete; // disallow copy constructor to because the column has the exclusive ownership of the data
    column& operator=(column&& other)      = delete;

    column(column&& other) noexcept;
    
    explicit column(data_type type, size_type num_rows, bool clear = false) 
        : _size(num_rows)
        , _type(type)
        , _data(num_rows * gamax::size_of(type), clear) {}
    
    explicit column(gamax::device_buffer&& buffer, size_type size, data_type type) 
    : _data(std::move(buffer))
    , _size(size)
    , _type(type) {}

    explicit column(data_type dtype,
           size_type size,
           gamax::device_buffer&& data)
        : _type{dtype}
        , _size{size}
        , _data{std::move(data)} {}

    size_type size() const {
        return _size;
    }

    data_type type() const {
        return _type;
    }

    device_buffer& buffer() {
        return _data;
    }

    void set_size (size_type size) {
        _size = size;
    }

    template <typename T = void>
    T* data() const noexcept
    {
        return static_cast<T*>(_data.data());
    }

private:
    gamax::device_buffer _data;
    gamax::size_type _size {0}; // number of elements in the column
    gamax::data_type _type;
};

template <typename T>
std::unique_ptr<column> make_column_from_host(T* host_data, size_type num_items) {
    gamax::device_buffer buffer(num_items * sizeof(T), false);
    cudaMemcpy(buffer.data(), host_data, num_items * sizeof(T), cudaMemcpyHostToDevice);
    return std::make_unique<column>(gamax::data_type{gamax::type_to_id<T>()}, num_items, std::move(buffer));
}

template <typename T>
void copy_column_to_host(column const& col, T* host_data) {
    cudaMemcpy(host_data, col.data<T>(), col.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

std::unique_ptr<column> make_column(data_type type, size_type size); 
std::unique_ptr<column> make_idx_column(size_type size); 
std::unique_ptr<column> make_sequence(size_type size, size_type start = 0, size_type step = 1); 
void free_column(std::unique_ptr<column> column);
} // namespace gamax