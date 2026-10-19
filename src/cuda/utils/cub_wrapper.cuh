// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include "utils.cuh"
#include "memory/device_buffer.hpp"

#define CALL_CUB_FUNC(stream, func_name, ...) \
    { \
        void* d_temp_storage = nullptr; \
        size_t temp_storage_bytes = 0; \
        func_name(d_temp_storage, temp_storage_bytes, __VA_ARGS__); \
        gamax::device_buffer cub_temp_storage(temp_storage_bytes, stream); \
        d_temp_storage = cub_temp_storage.data(); \
        func_name(d_temp_storage, temp_storage_bytes, __VA_ARGS__); \
    }
