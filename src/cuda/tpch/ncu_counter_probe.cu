// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <cstddef>
#include <cstdio>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace {

__global__ void ncu_counter_probe(float* values, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] = static_cast<float>(index);
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "error: %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

class ProfileRange {
public:
    ProfileRange() {
        nvtxEventAttributes_t attributes {};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attributes.message.ascii = "tpch.profile";
        nvtxDomainRangePushEx(domain(), &attributes);
    }
    ~ProfileRange() { nvtxDomainRangePop(domain()); }

private:
    static nvtxDomainHandle_t domain() {
        static auto handle = nvtxDomainCreateA("gamax.tpch");
        return handle;
    }
};

}  // namespace

int main() {
    constexpr std::size_t count = 1U << 20;
    float* values = nullptr;
    if (!check(cudaMalloc(&values, count * sizeof(float)), "cudaMalloc")) return 1;
    cudaError_t launch_status;
    cudaError_t sync_status;
    {
        ProfileRange range;
        ncu_counter_probe<<<(count + 255) / 256, 256>>>(values, count);
        launch_status = cudaGetLastError();
        sync_status = cudaDeviceSynchronize();
    }
    cudaFree(values);
    if (!check(launch_status, "kernel launch")) return 1;
    if (!check(sync_status, "cudaDeviceSynchronize")) return 1;
    std::puts("ncu_counter_probe: pass");
    return 0;
}
