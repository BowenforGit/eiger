<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# Alps: a prototype for distributed GPU-accelerated query processing on NVL72

## License

The project's MIT license and copyright notices are in [LICENSE](LICENSE).
Third-party portions retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES](LICENSES).
The notices identify unresolved source provenance that must be reviewed before
redistribution. NVIDIA copyright notices apply only to NVIDIA's contributions.

## Dependencies

- CMake 3.30.4+
- Ninja
- A C++ compiler with OpenMP support
- NVIDIA CUDA Toolkit and a compatible driver
- MPI and NCCL

CMake downloads the pinned RMM, CCCL, cuCollections, NVBench, and GoogleTest
dependencies during configuration, so the first configure requires network
access or a populated CMake/CPM cache.

## Get Alps

Replace `REPOSITORY_URL` with the URL of this repository.

```sh
git clone --recursive REPOSITORY_URL alps
cd alps
```

If the repository was cloned without `--recursive`, initialize its submodules:

```sh
git submodule update --init --recursive
```

## Build Alps

Alps uses an out-of-source CMake build. Set `CMAKE_CUDA_ARCHITECTURES` for the
target GPU; `100` targets B200/GB200.

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=100 \
  -DBUILD_ALPS=ON \
  -DBUILD_TEST=OFF \
  -DBUILD_BENCHMARK=OFF

cmake --build build --target gamax_alps -j"$(nproc)"
```

For the TPC-H executables and distributed benchmark workflow, see
[`docs/DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md`](docs/DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md).
