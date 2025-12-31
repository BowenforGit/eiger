# GaMax Copilot Instructions

## Project Overview
GaMax (GPU-Accelerated Maximus) is a GPU-accelerated query engine providing database operators (joins, grouped aggregation, sorting) implemented in CUDA with AMD HIP portability via hipify.

## Architecture

### Directory Structure
- **`src/cuda/`** - All source code organized by functionality:
  - `operators/` - Database operators: `groupby/`, `join/`, `sort/`, `partition/`, `select/`
  - `memory/` - Data structures: `column.hpp`, `table.hpp`, `mem_manager.hpp`
  - `types/` - Type system: `data_type.hpp` defines `type_id` enum and `data_type` class
  - `benchmarks/` - NVBench-based benchmarks (one per operator)
  - `tests/` - GoogleTest-based tests
  - `utils/` - CUDA utilities, CUB wrappers, timing macros

### Key Abstractions
- **`gamax::column`** / **`gamax::column_view`** - GPU column with typed access via `data<T>()`
- **`gamax::table`** / **`gamax::table_view`** - Collection of columns with device views for kernel access
- **`gamax::size_type`** - Use `int64_t` for sizes (defined in `types/data_type.hpp`)
- **Memory**: Custom allocator via `MemoryManager` class; use `allocate_mem()` / `release_mem()` helpers

### Operator Pattern
Operators follow a consistent pattern (see `operators/groupby/groupby.hpp`):
1. Header (`.hpp`) declares public API returning `std::unique_ptr<gamax::table>` or `std::unique_ptr<gamax::column>`
2. Implementation (`.cu`) uses type dispatchers for handling multiple data types
3. All operators accept `cudaStream_t stream` for async execution

## Build System

### Building for NVIDIA (CUDA)
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 ..
cmake --build . -- -j
```

### Building for AMD (ROCm/HIP)
```bash
mkdir build_hip && cd build_hip
cmake -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release -DUSE_ROCM=ON ..
cmake --build . -- -j
```

### Adding New Source Files
All `.cu/.cpp` files must be added to `src/cuda/CMakeLists.txt` in the `srcs` list. CUDA-only files go in the `if(USE_NVIDIA)` block.

## Development Patterns

### Type Dispatching
Use `gamax::type_dispatcher` to handle multiple data types:
```cpp
gamax::type_dispatcher(column.type(), functor{}, args...);
```

### CUDA Kernels
- Use `__host__ __device__` annotations for functions callable from both
- Use `CALL_CUB_FUNC` macro for CUB library calls (handles temp storage pattern)
- Timing: `SETUP_TIMING()` and `TIME_FUNC(f, t)` macros in `utils/utils.cuh`

### Platform Portability (CUDA/HIP)
Use these macros for platform-specific code:
- `__NVCC__` - NVIDIA compiler (not translated by hipify)
- `__HIP_PLATFORM_AMD__` - AMD HIP compiler
- `__CUDA_ARCH__` / `__HIP_DEVICE_COMPILE__` - Device code detection

## Testing & Benchmarks

### Running Tests
```bash
cd build
ctest                           # Run all tests
./src/cuda/tests/test_sort      # Run specific test
```

### Running Benchmarks
```bash
./build/src/cuda/benchmarks/benchmark_groupby --help
./build/src/cuda/benchmarks/benchmark_sort
```

### Test Pattern
Tests use GoogleTest with shared helpers in `tests/test_helper.hpp`:
- `gamax::test::check_cols_eq()` - Compare columns
- `gamax::test::check_cols_eq_after_sort()` - Compare sorted columns

### Benchmark Pattern
Benchmarks use NVBench. See `benchmarks/groupby/groupby_nvbench.cu` for the pattern:
1. Generate test data using utilities from `data_gen/`
2. Call `state.exec()` with the operation to benchmark
3. Use `state.skip()` for invalid parameter combinations

## Common Gotchas
- Always call `release_all_mem()` at benchmark start to reset memory state
- Column views reference data without ownership; ensure underlying column lifetime
- NVTX annotations only available on CUDA (`#ifdef __NVCC__`)
- Debug builds define `DEBUG` macro and enable `-g -G` CUDA flags

## Legacy Code (Do Not Use)
- **`string/string.cu`** - Outdated, ignore this file
- **"German strings"** - Legacy string format; do not use in new code
