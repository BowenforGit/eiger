# HIP Portability Follow-up

## What we checked
- Reviewed every item in `docs/hipify_risk_report.md` and verified the code locations it references.
- Confirmed hipify touches all CUDA sources plus CUDA CMakeLists; ensured fixes live in the original sources so hipified output inherits them.

## Changes made by risk item
- **Warp-size baked as 32 (partition_hash_join, merge_path)**: kernels now derive warp size from the target (32 on CUDA, 64 on HIP) and size shared buffers accordingly. Ballot masks use the platform width so 64-lane wavefronts keep all votes. See [src/cuda/operators/join/partition_hash_join.cu](src/cuda/operators/join/partition_hash_join.cu) and [src/cuda/operators/join/merge_path.cuh](src/cuda/operators/join/merge_path.cuh).
- **32-bit ballot masks on HIP**: ballot results and lane masks are handled with platform-sized types; popcount helpers switch between 32- and 64-bit variants. Covered in the same join kernels above.
- **Shared-memory sizing vs 64 KiB LDS**: join params now size shuffle storage using the platform warp size and iteratively shrink the block size until the requested dynamic shared memory fits the device limit. Limits come from runtime device queries. See [src/cuda/operators/join/partition_hash_join.cu](src/cuda/operators/join/partition_hash_join.cu).
- **device_info hard-coded per arch**: replaced static `SM_COUNT`/`MAX_SMEM_BYTES` constants with runtime queries and safe fallbacks for CUDA and HIP. See [src/cuda/utils/device_info.hpp](src/cuda/utils/device_info.hpp).
- **Groupby hash cuCollections args gated to CUDA**: always pass thread scope and storage so hipify will map them for HIP builds. Shared-memory sizing now uses runtime limits, and dynamic shared memory attributes honor HIP via portable macros. See [src/cuda/operators/groupby/groupby_hash.cu](src/cuda/operators/groupby/groupby_hash.cu).
- **cooperative_groups::memcpy_async CUDA-only**: added a HIP-safe synchronous copy path so aggregation flushes still occur when async copy is unavailable. See [src/cuda/operators/groupby/agg_helper.cuh](src/cuda/operators/groupby/agg_helper.cuh).
- **CUDA-only occupancy/helpers**: introduced HIP/CUDA portability macros for occupancy queries and function attributes, used in join and groupby code. See [src/cuda/utils/launch.cuh](src/cuda/utils/launch.cuh) and updates in the join/groupby sources.
- **Benchmarks link nvbench + unguarded NVTX**: benchmark CMake now picks the available bench main target (falls back to `nvbench::main` if `hipbench::main` is absent) and disables NVTX on ROCm. NVTX usage is routed through a stub header so HIP builds compile cleanly while CUDA keeps full NVTX. See [src/cuda/benchmarks/CMakeLists.txt](src/cuda/benchmarks/CMakeLists.txt) and [src/cuda/benchmarks/common/nvtx_stub.hpp](src/cuda/benchmarks/common/nvtx_stub.hpp).

## Notes and remaining watchpoints
- ROCm builds still exclude some CUDA-only sources/tests per CMake configuration; expanding coverage requires separate porting work beyond this pass.
- Cooperative groups coverage in string kernels was not adjusted here; none of those files build under `USE_ROCM` today.
- When adding new kernels, favor platform-aware warp sizing (`warpSize`/`kDeviceWarpSize`) and the occupancy macros in `utils/launch.cuh` to keep HIP builds aligned.
