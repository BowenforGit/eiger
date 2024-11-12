#pragma once

#include <cuda.h>

#include "utils/utils.cuh"

namespace gamax {
namespace utils {
namespace device {
    template <typename T>
    void create_sequence(T* data, const size_t size, const T start, const T step) {
        fill_sequence<<<(size + 255) / 256, 256>>>(data, start, size, step);
    }

    template <typename T>
    void create_ids(T* data, const size_t size) {
        create_sequence(data, size, 0, 1);
    }
}
}
}