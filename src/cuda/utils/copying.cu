#pragma once

#include <memory>
#include <cuda.h>

#include "memory/column.hpp"
#include "types/data_type.hpp"

namespace gamax {

    std::unique_ptr<column> copy_column(
        column_view const& view,
        cudaStream_t stream)
    {
        auto col = make_column(view.type(), view.size(), false, stream);
        cudaMemcpyAsync(
            col->data(), 
            view.data(), 
            view.size() * gamax::size_of(view.type()), 
            cudaMemcpyDeviceToDevice, 
            stream);
        return col;
    }

} // namespace gamax
