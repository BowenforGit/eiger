# CUDA to HIP Transpilation Guide

## Introduction

The GPU code is initially developed for NVIDIA GPUs. 
To let it run on AMD GPUs, we need to translate the CUDA code to [HIP code](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html), which is a portable technical stack that support CPUs, and both NVIDIA and AMD GPUs.

The process of translating CUDA to HIP is called "hipify". 
There are multiple ways to hipify, as mentioned [here](https://rocmdocs.amd.com/projects/HIPIFY/en/latest/index.html).

We mainly considered two ways of hipifying. 

- `hipify-perl` and `hipify-clang` work on a per file basis. 
After running them, you still need extra work to make your build system work, e.g., cmake, source file locations.
- `hipify-torch` is initially developed to hipify the pytorch code base but can also be used as general-purpose.
The advantage is that it hipifies the whole project and can translate source code and make the build system work in one go.

We adopt the `hipify-torch` as the solution. 
In the rest of this document, we describe the most important steps of hipifying.

## Translating source code

Translation of the source code is handled by `tools/build_amd.py`. 
It translates all the source code in `src/cuda` and stores the output in `src/hip`.
The translation is not just limited to the source code. 
Files that define the build system, most notably `src/cuda/CMakeLists.txt`, 
whenever possible are also translated. 
For now, the translation of the build system files is limited to replace the `.cu` translation units to `.hip`.

### Custom mappings

`hipify-torch` provides a very extensive mapping from CUDA to HIP that covers functions, macros etc.
However, in some cases, these are still not sufficient. 
We can define our own mapping rules in `tools/custom_hipify_mapping.json`.

### NVIDIA libraries

For most popular NVIDIA libraries (e.g., Thrust, CUB), 
HIP has corresponding ones that are distributed with the ROCm library installed in the system. 
For less popular ones that this project depends on, we need to manually specify them in the CMakeLists.txt to fetch from Github. These include `hipCollection` and `hipBench`.

### Unsupported features and incompatibilities

Some CUDA features or library functions are not supported in AMD or at least not in an equivalent manner.
To port these, we use macros to steer the compiler to compile for the specific platform. 
We reply on the compiler definitions to distinguish what the target platform is and if we are compiling host code or device code.

To tell the target platform,

- `__NVCC__` is defined when `nvcc` is compiling C/C++/CUDA source files. This will **not** be translated in hipification so therefore can be used to distinguish the target platform. However, it cannot be used to distinguish the host and device code.
- `__HIP_PLATFORM_AMD__` is defined when `hip-clang` is compiling the code.
- `__CUDACC__` is defined when `nvcc` is compiling CUDA source files. Defined for both host and device code. This definition will be translated to `__HIPCC__` after hipifying; therefore you should never use this definition to distinguish the target platform.

To tell host vs. device,

- `__CUDA_ARCH__` is defined when `nvcc` is compiling the **device** code. 
This is especially useful when a code segment can run in both host and device 
and need some customization depending on running on host or device. This definition **will not and should not be** translated during hipify. 
- `__HIP_DEVICE_COMPILE__` is defined when `hip-clang` is compiling the **device** code.

For more information on the compiler macros, check out [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-identification-macro) and [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html#distinguishing-compiler-modes)


## Build system

Since we want to reuse the source code and build system for the AMD platform, 
we need to extend the build system in addition to source code translation.
Compiling to the AMD GPUs requires the `hip-clang` compiler. 
Thanks to the HIP support by CMake, finding the toolchain is automatic. 
Although the `nvcc` and `hip-clang` work similarly, there are a few key differences.

- Much fewer flags are supported by `hip-clang`.
- It takes more effort to make OpenMP work for `clang`, including manually set the path to the headers and dynamic library.

We are only testing the MI210 GPU, so to configure the build, run

```bash
mkdir -p build_hip && cd build_hip
cmake -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release -DUSE_ROCM=ON ..
```

## Compilation and Linking

As usual, run

```bash
cmake --build . -- -j
```

## Troubleshooting

### libstdc++.so.6: version `GLIBCXX_3.4.30' not found

You need to install newer version of GCC, e.g, 12.1.0. For me, the following works 
because the linker happens to use the `libstdc++` library from my miniconda environment (not sure why).

```bash
conda install -c conda-forge gcc=12.1.0
```

If you see version `3.4.30` when you run `strings <path>/libstdc++.so.6 | grep GLIBCXX`, you are good to go.
