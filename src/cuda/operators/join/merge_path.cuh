#pragma once
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <vector>
#include <tuple>
#include <string>
#include <type_traits>

#include <cuda.h>
#include <cub/cub.cuh> 
#include <cub/block/block_shuffle.cuh>
#include <cuda_pipeline.h>

#include "utils/utils.cuh"
#include "atomics/atomics.cuh"
#include "utils/warp_utils.cuh"
#include "utils/device_info.hpp"

template<typename idx_t, typename cnt_t>
struct merge_range {
    idx_t a_begin;
    idx_t a_end;
    idx_t b_begin;
    idx_t b_end;

    __device__ merge_range(const idx_t w, const idx_t x, const idx_t y, const idx_t z) 
    : a_begin(w)
    , a_end(x) 
    , b_begin(y)
    , b_end(z) {}

    __device__ merge_range(const int cta, const cnt_t spacing, const cnt_t a_count, const cnt_t b_count, const idx_t* partitions) {
        a_begin = partitions[cta];
        a_end   = partitions[cta+1];
        auto diag0 = cta*spacing;
        auto diag1 = min(a_count+b_count, diag0+spacing);
        b_begin = diag0-a_begin;
        b_end   = diag1-a_end;
    }
    
    __device__ __forceinline__ cnt_t a_count() { return a_end - a_begin; }
    __device__ __forceinline__ cnt_t b_count() { return b_end - b_begin; }
    __device__ __forceinline__ cnt_t total() { return a_count() + b_count(); }

    __device__ __forceinline__ bool a_valid() { return a_end > a_begin; }
    __device__ __forceinline__ bool b_valid() { return b_end > b_begin; }


    __device__ __forceinline__ merge_range align_to_zero() { 
        return merge_range(static_cast<idx_t>(0), a_count(), a_count(), total());
    }

    __device__ __forceinline__ merge_range partition(cnt_t mp0, cnt_t diag0, cnt_t mp1, cnt_t diag1) {
        return merge_range(a_begin + mp0, 
                           a_begin + mp1, 
                           b_begin + diag0 - mp0, 
                           b_begin + diag1 - mp1);
    }
};

template<typename key_t>
__device__ __forceinline__ gamax::size_type merge_path(key_t* a_keys, gamax::size_type a_count, key_t* b_keys, gamax::size_type b_count, gamax::size_type diag) {
    gamax::size_type begin = max(static_cast<gamax::size_type>(0), diag - b_count);
    gamax::size_type end = min(diag, a_count);

    while(begin < end) {
        gamax::size_type mid = begin + (end-begin) / 2;
        key_t a_key = a_keys[mid];
        key_t b_key = b_keys[diag - 1 - mid];
        bool pred = (a_key <= b_key);

        if(pred) begin = mid + 1;
        else end = mid;
    }
    return begin;
}

template<typename key_t>
__global__ void merge_path_partition(key_t* a, const gamax::size_type a_count, 
                                     key_t* b, const gamax::size_type b_count,
                                     gamax::size_type spacing, 
                                     gamax::size_type* partition, const gamax::size_type n) {
    gamax::size_type p = get_cuda_tid();
    if(p >= n) return;
    partition[p] = merge_path(a, a_count, b, b_count, min(spacing*p, a_count+b_count));
}

template<int NT = 512, 
         int VT = 4, 
         typename key_t,
         typename RValIt, 
         typename SValIt, 
         typename ROutIt,
         typename SOutIt>
__global__ void merge_keys_mp_improv(key_t* l_sorted_keys, key_t* r_sorted_keys, 
                              RValIt  l_sorted_vals, SValIt r_sorted_vals,
                              const gamax::size_type nl, const gamax::size_type nr, const gamax::size_type* partitions,
                              key_t* keys_out, ROutIt l_out, SOutIt r_out, 
                              gamax::size_type* g_counter, gamax::size_type circular_buffer_size, bool agg_only = false) {
    constexpr int HW_WARP = gamax::warp_size(); // 32 on CUDA, 64 on HIP

    auto cta = blockIdx.x;
    constexpr int NV = NT*VT;

    // obtain the merge partition belonging to this thread block
    merge_range mr_global(static_cast<gamax::size_type>(cta), static_cast<gamax::size_type>(NV), nr, nl, partitions);

    auto a = r_sorted_keys + mr_global.a_begin; // needles
    auto b = l_sorted_keys + mr_global.b_begin; // haystacks

    extern __shared__ char smem_buffer[]; 
    key_t* keys = reinterpret_cast<key_t*>(smem_buffer); // #items = NV+1

    for(int i = 0; i < VT; i++) {
        auto index = i*NT+threadIdx.x;
        if(index >= mr_global.total()) break;
        if(index < mr_global.a_count()) __pipeline_memcpy_async(&keys[index], &a[index], sizeof(key_t));
        else __pipeline_memcpy_async(&keys[index], &b[index - mr_global.a_count()], sizeof(key_t));
    }
    __pipeline_commit();

    // partition the workloads within the thread block
    auto mr_cta = mr_global.align_to_zero();
    gamax::size_type diag0 = min(mr_cta.total(), static_cast<gamax::size_type>(VT*threadIdx.x));
    gamax::size_type diag1 = min(mr_cta.total(), diag0+VT);
    gamax::size_type mp0[1];
    gamax::size_type mp1[1];
    __pipeline_wait_prior(0);
    __syncthreads();
    mp0[0] = merge_path(keys, mr_cta.a_count(), keys+mr_cta.b_begin, mr_cta.b_count(), diag0);
    mp1[0] = mr_cta.a_count();
    
    cub::BlockShuffle<gamax::size_type, NT>().Down(mp0, mp1);
    
    // Find the lower bound for each "needle"
    // if there is a match, the lower bound should be the first match
    // otherwise the lower bound is the smallest element in "haystack" that is
    // greater than the "needle".
    auto mr_thread = mr_cta.partition(mp0[0], diag0, mp1[0], diag1);

    __shared__ gamax::size_type match[VT][NT];
    __shared__ gamax::size_type cnt[VT][NT];
    
    #pragma unroll
    for(int i = 0; i < VT; i++) {
        match[i][threadIdx.x] = mr_thread.b_end;
        cnt[i][threadIdx.x] = 0;
    }
    
    const bool is_valid_range = mr_thread.a_valid();
    const auto a_first = mr_thread.a_begin;
    if(mp0[0] < mr_global.a_count()) {
        while(mr_thread.a_valid() && mr_thread.b_valid()) {
            auto a_key = keys[mr_thread.a_begin], b_key = keys[mr_thread.b_begin];
            if(a_key <= b_key) {
                if(a_key == b_key) {
                    match[mr_thread.a_begin - a_first][threadIdx.x] = mr_thread.b_begin;
                }
                ++mr_thread.a_begin;
            } else if(a_key > b_key) {
                ++mr_thread.b_begin;
            }
        }
    }

    // materialize the result
    if(mp0[0] < mr_global.a_count()) {
        #pragma unroll
        for(gamax::size_type a_index = a_first; a_index < mr_thread.a_end; a_index++) {
            gamax::size_type b_index = match[a_index-a_first][threadIdx.x] - mr_cta.a_count();
            while(b_index + mr_global.b_begin < nl) {
                auto b_index_cta = b_index + mr_cta.a_count();
                key_t bk = (b_index_cta < mr_cta.total()) ? keys[b_index_cta] : b[b_index];

                if(bk == keys[a_index]) {
                    b_index++;
                    cnt[a_index-a_first][threadIdx.x]++;
                } else break;
            }
        }
    }
    
    int i = 0, j = 0;
    int shuffle_ptr = 0;
    const int wid = threadIdx.x / HW_WARP;
    const int lid = threadIdx.x % HW_WARP; 
    using ballot_t = decltype(__activemask());
    const ballot_t threadmask = gamax::warp::upper_lane_mask<ballot_t>(lid, HW_WARP);
    
    constexpr int SHUFFLE_SIZE = HW_WARP;
    __shared__ key_t keys_shfl[NT/HW_WARP][SHUFFLE_SIZE];
    __shared__ std::remove_pointer_t<ROutIt> r_shfl[NT/HW_WARP][SHUFFLE_SIZE];
    __shared__ std::remove_pointer_t<SOutIt> l_shfl[NT/HW_WARP][SHUFFLE_SIZE];
    
    const bool valid_range = (mp0[0] < mr_global.a_count());
    int wr_intention;
    int ptr;


    while(true) {
        while(i < VT && j >= cnt[i][threadIdx.x]) {
            i++; 
            j = 0;
        }     
        wr_intention = (valid_range && i < VT);
        
        ballot_t mask = __ballot_sync(__activemask(), wr_intention);
        if(mask == 0) break; // no one has the write intention anymore, exit
        
        int wr_offset = shuffle_ptr +  gamax::warp::popc_mask(mask & threadmask);
        shuffle_ptr = shuffle_ptr + gamax::warp::popc_mask(mask);

        // the buffer will be full if the current batch is written
        while (shuffle_ptr >= SHUFFLE_SIZE) {
            if (wr_intention && (wr_offset < SHUFFLE_SIZE)) {
                gamax::size_type a_index = a_first + i + mr_global.a_begin;
                gamax::size_type b_index = match[i][threadIdx.x] - mr_cta.a_count() + mr_global.b_begin + j;
                keys_shfl[wid][wr_offset] = keys[a_first+i];
                r_shfl[wid][wr_offset] = r_sorted_vals[a_index];
                l_shfl[wid][wr_offset] = l_sorted_vals[b_index];
                wr_intention = 0;
            }

            if (lid == 0) {
                ptr = gamax::atomic::atomic_add(g_counter, static_cast<gamax::size_type>(SHUFFLE_SIZE));
            }

            ptr = __shfl_sync(__activemask(), ptr, 0);

            auto w_pos = (ptr + lid) % circular_buffer_size;

            keys_out[w_pos] = keys_shfl[wid][lid];
            r_out[w_pos] = r_shfl[wid][lid];
            l_out[w_pos] = l_shfl[wid][lid];

            wr_offset -= SHUFFLE_SIZE;
            shuffle_ptr -= SHUFFLE_SIZE;
        }

        if(wr_intention && (wr_offset >= 0)) {
            gamax::size_type a_index = a_first + i + mr_global.a_begin;
            gamax::size_type b_index = match[i][threadIdx.x] - mr_cta.a_count() + mr_global.b_begin + j;
            keys_shfl[wid][wr_offset] = keys[a_first+i];
            r_shfl[wid][wr_offset] = r_sorted_vals[a_index];
            l_shfl[wid][wr_offset] = l_sorted_vals[b_index];
            wr_intention = 0;
        }

        j++;
    }

    if (lid == 0) {
        ptr = gamax::atomic::atomic_add(g_counter, static_cast<gamax::size_type>(shuffle_ptr));
    }

    ptr = __shfl_sync(__activemask(), ptr, 0);

    if (lid < shuffle_ptr) {
        auto w_pos = (ptr + lid) % circular_buffer_size;
        r_out[w_pos] = r_shfl[wid][lid];
        l_out[w_pos] = l_shfl[wid][lid];
        keys_out[w_pos] = keys_shfl[wid][lid];
    }
}

template<typename KeyIt, 
         typename RValIt, 
         typename SValIt, 
         typename ROutIt,
         typename SOutIt, 
         int NT = 512, 
         int VT = 4>
void merge_path(KeyIt r_sorted_keys, KeyIt s_sorted_keys, 
                RValIt r_sorted_vals, SValIt s_sorted_vals, 
                const gamax::size_type nr, const gamax::size_type ns, 
                KeyIt keys_out, ROutIt r_out, SOutIt s_out, 
                gamax::size_type& n_matches, gamax::size_type circular_buffer_size) {
    gamax::size_type n = (nr + ns + NT*VT-1)/(NT*VT) + 1;
    gamax::size_type* partitions, *g_counter;
    allocate_mem(&partitions, false, sizeof(gamax::size_type)*n);
    allocate_mem(&g_counter, true);

    merge_path_partition<<<num_tb(n, 128), 128>>>(s_sorted_keys, ns,
                                                    r_sorted_keys, nr,
                                                    NT*VT,
                                                    partitions, n);
    
    CHECK_LAST_CUDA_ERROR();
    std::size_t smem_bytes = sizeof(*r_sorted_keys) * (NT*VT+1); // keys array
    auto fn = merge_keys_mp_improv<NT, VT, typename std::remove_pointer<KeyIt>::type, RValIt, SValIt, ROutIt, SOutIt>;
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    merge_keys_mp_improv<NT, VT><<<num_tb(nr+ns, NT, VT), NT, smem_bytes>>>(r_sorted_keys, s_sorted_keys, 
                                                            r_sorted_vals, s_sorted_vals,
                                                            nr, ns,
                                                            partitions, keys_out, 
                                                            r_out, s_out, 
                                                            g_counter, circular_buffer_size);
    CHECK_LAST_CUDA_ERROR();

    gamax::size_type temp;
    cudaMemcpy(&temp, g_counter, sizeof(gamax::size_type), cudaMemcpyDefault);
    n_matches = temp;

    release_mem(partitions);
    release_mem(g_counter);
}
