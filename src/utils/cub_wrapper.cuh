#pragma once

#include "utils.cuh"

#define CALL_CUB_FUNC(func_name, ...) \
    { \
        void* d_temp_storage = nullptr; \
        size_t temp_storage_bytes = 0; \
        func_name(d_temp_storage, temp_storage_bytes, __VA_ARGS__); \
        allocate_mem(&d_temp_storage, false, temp_storage_bytes); \
        func_name(d_temp_storage, temp_storage_bytes, __VA_ARGS__); \
        release_mem(d_temp_storage); \
    }
