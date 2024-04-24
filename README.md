# GPU-Accelerated Maximus (GaMax)

GaMax is a GPU-accelerated query engine that introduces the GPU acceleration to the Maximus, a distributed modular query engine that should be able to run on heterogeneous accelerated systems. GaMax provides a rich collection of database operators, such as joins, grouped aggregation and sorting, as well as various "sub-operators" or useful routines (e.g., partitioning, dictionary encoding) that can be used implement more complex operators. 

Currently, GaMax is under active development and has supported

- Inner eq-join
- Grouped aggregation (with a single group key)
- Dictionary encoding

## Dependencies

- CMake 3.23.1+
- GNU Make

## Code Structure

All non-third-party source code are located in the `src/` directory, including the tests and benchmarks code. 

- `atomics/` - wrappers and implementations of atomic operations.
- `benchmarks` - benchmarks based on NVIDIA NVBench.
- `compress` - data compression and encoding, e.g., dictionary encoding.
- `containers` - wrappers and implementations of data containers that facilitates the implementations of some operators.
- `data_gen` - data generators used by the tests and benchmarks.
- `memory` - in-memory data formats, such as columns and tables, and memory management.
- `operators` - database operators.
- `pullops` - volcano-style wrappers of the operators.
- `tests/` - tests based on GoogleTest.
- `types/` - definition of data types and related utility functions.
- `utils` - utility functions.

## Get GaMax

```sh
git clone --recursive git@gitlab.inf.ethz.ch:wubo/vldb24-artifact.git
```

## Build GaMax

GaMax uses CMake and Make to build the source code in an out-of-source manner. Its dependencies on NVBench, GoogleTest and CuCollections are automatically resolved by CMake. To build Gamax, follow the steps below.

```sh
# in the project root directory
mkdir build && cd build
# fill in your own build type and targeting cuda architecture
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 ..
# pass the number of compilation threads to the -j flag based on available CPU cores
cmake --build . -- -j 12
```

All the build files and executables are located in the `build/` directory, which mirrors the source directory hierarchy. For example, the test and benchmark executables are located in `build/src/tests` and `build/src/benchmarks` respectively. 

## Contributing

See the [contributing guide](CONTRIBUTING.md).