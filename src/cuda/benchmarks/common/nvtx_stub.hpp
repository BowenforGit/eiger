#pragma once

// Lightweight NVTX shim to allow building benchmarks without NVTX on HIP.
#ifdef GAMAX_DISABLE_NVTX
namespace nvtx3 {
inline void mark(const char*) {}
}
inline int nvtxRangePushA(const char*) { return 0; }
inline void nvtxRangePop() {}
#else
#include <nvtx3/nvtx3.hpp>
#include <nvtx3/nvToolsExt.h>
#endif
