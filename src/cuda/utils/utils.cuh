#pragma once

#include <cuda.h>
#include <cub/cub.cuh> 
#include <iostream>

#include "types/data_type.hpp"
#include "memory/mem_manager.hpp"
#include "utils/check_error.hpp"

#define COL(t,i) (t).template get_typed_ptr<(i)>()

#define CEIL(a,b) (((a)+(b)-1)/(b))

// Timing functions are from Anil Shanbhag (https://github.com/anilshanbhag/crystal)

#define SETUP_TIMING() cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

#define TIME_FUNC(f,t) { \
    cudaEventRecord(start, 0); \
    f; \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&t, start,stop); \
}

#define TIME_FUNC_ACC(f,t) { \
    cudaEventRecord(start, 0); \
    f; \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    float temp; \
    cudaEventElapsedTime(&temp, start,stop); \
    t += temp; \
}

// https://stackoverflow.com/a/47563100/9054082
template<std::size_t N>
struct num { static const constexpr auto value = N; };
template <class F, std::size_t... Is>
__device__ __host__
void for_(F func, std::index_sequence<Is...>)
{
  (func(num<Is>{}), ...);
}
template <std::size_t N, typename F>
__device__ __host__
void for_(F func)
{
  for_(func, std::make_index_sequence<N>());
}

__host__ __device__
inline gamax::size_type num_tb(const gamax::size_type N, const gamax::size_type threads_per_block=1024, const gamax::size_type items_per_thread=1) {
    return (N + threads_per_block*items_per_thread - 1) / (threads_per_block*items_per_thread);
}

// The following two utility functions only work for 1D thread layout
__device__ __forceinline__ gamax::size_type get_cuda_tid() {
    return (gamax::size_type) threadIdx.x + (gamax::size_type) blockDim.x * (gamax::size_type) blockIdx.x;
}

__device__ __forceinline__ gamax::size_type nthreads() {
    return (gamax::size_type) blockDim.x * (gamax::size_type) gridDim.x;
}

template<typename T>
void print_gpu_arr(const T* arr, gamax::size_type n, gamax::size_type offset=0) {
    T* temp = new T[n];
    cudaMemcpy(temp, arr+offset, sizeof(T)*n, cudaMemcpyDeviceToHost);
    printf("%p: ", arr);
    for(auto i = 0; i < n; i++) {
        std::cout << temp[i] << " ";
    }
    std::cout << std::endl;
    delete [] temp;
}

template<typename T>
__global__ void fill_sequence(T* arr, const T start, const size_t N, const T step = 1) {
    for(gamax::size_type t = get_cuda_tid(); t < N; t += nthreads()) {
        arr[t] = start+t*step;
    }
}