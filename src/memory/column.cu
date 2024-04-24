#include "column.hpp"
#include "types/type_dispatcher.hpp"
#include "utils/sequence.cuh"

#include <memory>

namespace gamax {

column::column(column&& other) noexcept
: _type{other._type},
  _size{other._size},
  _data{std::move(other._data)}
{
  other._size       = 0;
  other._type       = data_type{type_id::EMPTY};
}

std::unique_ptr<column> make_column(data_type type, size_type size, bool clear) {
    return std::make_unique<column>(type, size, clear);
}

std::unique_ptr<column> make_idx_column(size_type size, bool clear) {
    return make_column(data_type{size_type_id}, size, clear);
}

std::unique_ptr<column> make_sequence(
    size_type size, 
    size_type start, 
    size_type step) {
    
    auto col = make_idx_column(size);
    gamax::utils::device::create_sequence(col->data<gamax::size_type>(), size, start, step);
    return col;
}


void free_column(std::unique_ptr<column> col) {
    col.reset(nullptr);
}
} // namespace gamax