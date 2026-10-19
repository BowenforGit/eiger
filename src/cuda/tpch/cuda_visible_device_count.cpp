// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <cstring>
#include <iostream>

#include <cuda_runtime_api.h>

int main(int argc, char** argv) {
    if (argc != 2 ||
        std::strcmp(argv[1], "--print-visible-device-count") != 0) {
        std::cerr << "usage: " << argv[0]
                  << " --print-visible-device-count\n";
        return 2;
    }

    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice) {
        std::cout << "0\n";
        return 0;
    }
    if (status != cudaSuccess) {
        std::cerr << "error: cudaGetDeviceCount failed: "
                  << cudaGetErrorString(status) << "\n";
        return 2;
    }
    std::cout << count << "\n";
    return 0;
}
