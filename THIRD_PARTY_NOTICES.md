# Third-party notices

The repository's existing project license is MIT, with Copyright (c) 2023
Bowen Wu. The complete grant is preserved in `LICENSE`, with an additional
2026 NVIDIA copyright notice for NVIDIA contributions. These notices do not
replace third-party copyrights or relicense third-party code.

This inventory covers tracked source, explicit source attributions, CMake
dependencies, Python imports, and the recorded submodule. It is not a complete
transitive software bill of materials for any particular binary or environment.
Unresolved items below must be cleared before redistribution.

## Included or adapted source

| Location | Origin and copyright | Terms and status |
| --- | --- | --- |
| `src/cuda/utils/assert.cuh` | Existing header: Copyright (c) 2019-2023, NVIDIA CORPORATION | Existing Apache 2.0 header is unchanged. Full text: `LICENSES/Apache-2.0.txt`. Original source repository/revision is not recorded in this file. |
| Timing macros in `src/cuda/utils/utils.cuh` | [Crystal](https://github.com/anilshanbhag/crystal), Copyright (c) 2020 Anil Shanbhag | MIT. Full copyright and license: `LICENSES/Crystal-MIT.txt`, obtained from upstream revision `f2179e607eb923f2053a706c1e30aa2b6161b9db`. Exact copied revision is not recorded. |
| `num` and `for_` templates in `src/cuda/utils/utils.cuh` | [Answer 47563100](https://stackoverflow.com/a/47563100) by [Jean-Michaël Celerier](https://stackoverflow.com/users/1495627/jean-micha%c3%abl-celerier), posted November 29, 2017, edited April 18, 2018 | The Stack Exchange API identifies this answer as CC BY-SA 3.0. Full text: `LICENSES/CC-BY-SA-3.0.txt`. Adaptation includes CUDA host/device annotations. **REVIEW REQUIRED: share-alike obligations and compatibility with the proposed contribution. The project MIT license does not override these terms.** |
| `src/cuda/data_gen/generator.cuh` | Existing attribution to Cagri Balkesen, `multicore-hashjoins-0.2`, and a function derived from Rene Mueller's code. Original link: https://www.systems.ethz.ch/node/334 | **LICENSE / ORIGINAL COPYRIGHT NOTICE NOT YET VERIFIED.** Existing attributions are preserved. Obtain the original distribution and applicable license/permission, including the Mueller-derived portion, before release. No MIT or SPDX license is asserted for the inherited third-party portions. |
| `tools/build_amd.py` | Adapted from [PyTorch's AMD build script](https://github.com/pytorch/pytorch/blob/main/tools/amd_build/build_amd.py), as stated in its existing header | Full PyTorch copyright list and BSD terms are in `LICENSES/PyTorch-BSD.txt`, obtained from the contemporary `v2.5.1` release. This repository first added the adaptation on November 16, 2024. **CONFIRM the exact copied revision and any file-specific terms.** |
| `tools/hipify_torch` submodule | [ROCm/hipify_torch](https://github.com/ROCm/hipify_torch), revision `cbce9fbfd2783e932a45e3b47ca522e59b880c66`; Copyright (c) 2021-2024, Advanced Micro Devices, Inc. All rights reserved. | MIT; exact pinned root license reproduced in `LICENSES/hipify_torch-MIT.txt`. Submodule was uninitialized during preparation. If its contents are included in a distribution, retain all of its own file-level notices too. The current CMake configuration no longer supports the ROCm/hipify build path, but these files remain tracked. |

The Stack Exchange metadata was checked at
https://api.stackexchange.com/2.3/answers/47563100?site=stackoverflow&filter=withbody.
Adding these notices does not resolve the outstanding permission questions.

## Build and runtime dependencies

These dependencies are fetched or located by the build, rather than copied into
the tracked source tree. Their own license and notice files must accompany any
distribution that actually includes their source, headers, libraries, or other
redistributable components. System-discovered versions must be recorded from
the intended release environment; the links below are not evidence that a
particular installed version was checked. Static versus dynamic linkage must
also be checked for the final build where CMake leaves it configurable.

| Dependency | Repository evidence / interaction | License source |
| --- | --- | --- |
| RAPIDS CMake | Root `CMakeLists.txt`; downloaded CMake scripts at `v26.06.00` | [Apache 2.0](https://github.com/rapidsai/rapids-cmake/blob/v26.06.00/LICENSE) |
| RMM | `src/cuda/CMakeLists.txt`; headers and compiled library, `v26.06.00` | [Apache 2.0](https://github.com/rapidsai/rmm/blob/v26.06.00/LICENSE) |
| CCCL (CUB, Thrust, libcudacxx) | Header/template use; RAPIDS table pins `207502e57019cefacf6f21d3bb6045aeebef2a3e` | [Component-specific license collection](https://github.com/NVIDIA/cccl/blob/207502e57019cefacf6f21d3bb6045aeebef2a3e/LICENSE); retain individual component notices |
| cuCollections | Header/template use; RAPIDS table pins `f517bbb1277753b1852dfd388993383e401eaa38` | [Apache 2.0](https://github.com/NVIDIA/cuCollections/blob/f517bbb1277753b1852dfd388993383e401eaa38/LICENSE) |
| NVBench | Benchmark targets; directly requested revision `b0a46f44c21c344e2c338225d724eed60d912fe2` | [Apache 2.0](https://github.com/NVIDIA/nvbench/blob/b0a46f44c21c344e2c338225d724eed60d912fe2/LICENSE) |
| GoogleTest | Test targets; directly requested `v1.15.2` | [BSD 3-Clause](https://github.com/google/googletest/blob/v1.15.2/LICENSE) |
| NVTX | Header instrumentation, `nvtx3::nvtx3-cpp`; RAPIDS table requests `69c9949150ac1c310758a304082228a36d5e4758` | [License collection](https://github.com/NVIDIA/NVTX/blob/69c9949150ac1c310758a304082228a36d5e4758/LICENSE.txt) |
| NCCL | Located by `cmake/FindNCCL.cmake`; linked via `NCCL::nccl`; version not pinned | [Upstream license and notices](https://github.com/NVIDIA/nccl/blob/master/LICENSE.txt); confirm installed version |
| MPI and OpenMP runtime | System-discovered compiled libraries; implementations and versions not pinned | **RECORD IMPLEMENTATION, VERSION, AND LICENSE**; do not assume Open MPI, MPICH, LLVM, or GCC runtime terms |
| Apache Arrow / Parquet | Optional TPC-H targets explicitly use `Arrow::arrow_shared` and `Parquet::parquet_shared`; versions not pinned | [Apache Arrow license collection](https://github.com/apache/arrow/blob/main/LICENSE.txt); retain packaged NOTICE files |
| RapidJSON | TPC-H code includes headers; version not pinned | [Upstream license collection](https://github.com/Tencent/rapidjson/blob/master/license.txt); confirm supplied headers and notices |
| OpenSSL | TPC-H targets link `OpenSSL::Crypto`; SHA-256 in `src/cuda/tpch/tpch_layout.cpp`; version not pinned | [Current upstream license](https://github.com/openssl/openssl/blob/master/LICENSE.txt); **verify the installed version's terms**, which differ across major versions |
| CUDA Toolkit, driver API, NVML, Nsight tools | SDK headers/libraries and external profiling tools; versions not pinned | NVIDIA SDK/tool terms, not the project MIT license. No SDK or profiling binaries have been added to this contribution. |

The RAPIDS dependency table is
https://github.com/rapidsai/rapids-cmake/blob/v26.06.00/rapids-cmake/cpm/versions.json.
It also contains defaults that differ from the direct NVBench/GoogleTest requests.
Record actual resolved versions from the release build; this source-only review
did not configure or build the project.

## Python and external tools

Python imports use the following non-standard packages; installed versions and
their transitive dependencies are not pinned in a release lock file:

- [NumPy](https://github.com/numpy/numpy), [pandas](https://github.com/pandas-dev/pandas),
  [Matplotlib](https://github.com/matplotlib/matplotlib), and
  [seaborn](https://github.com/mwaskom/seaborn): analysis and visualization.
- [DuckDB](https://github.com/duckdb/duckdb), including its TPC-H extension:
  golden-query generation in `script/gen_tpch_golden.py`.
- [Apache Arrow / PyArrow](https://github.com/apache/arrow),
  [cuDF / pylibcudf](https://github.com/rapidsai/cudf),
  [CuPy](https://github.com/cupy/cupy), and [RMM](https://github.com/rapidsai/rmm):
  comparison benchmarks and data processing.
- HIPify: the separately documented submodule above.

`script/generate_tpch_multifile.py` executes an externally supplied `tpchgen-cli`
binary. **RECORD ITS REPOSITORY, VERSION, LICENSE, AND ANY TPC-H TERMS** before
shipping that tool or generated benchmark materials. Its implementation is not
included in tracked source. System Python modules include `hashlib`, `sqlite3`,
and `zlib`; exact runtime distribution notices depend on what is shipped.

## Files without inline copyright comments

`ncu_kernels.txt` and `ncu_metrics.section` are kept byte-for-byte unchanged to
avoid changing their input formats. NVIDIA contributions to these files are
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved, and
covered by the project `LICENSE`, subject to any applicable third-party terms.
