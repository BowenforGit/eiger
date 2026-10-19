// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <memory>
#include <cuda.h>

#include "memory/column.hpp"
#include "types/data_type.hpp"

namespace gamax {

    std::unique_ptr<column> copy_column(
        column_view const& view,
        cudaStream_t stream)
    {
        auto col = make_column(view.type(), view.size(), stream);
        cudaMemcpyAsync(
            col->data(), 
            view.data(), 
            view.size() * gamax::size_of(view.type()), 
            cudaMemcpyDeviceToDevice, 
            stream);
        return col;
    }

} // namespace gamax
