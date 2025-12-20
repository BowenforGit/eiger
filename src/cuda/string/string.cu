#include <cuda.h>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cooperative_groups.h>

#include "string/string.hpp"
#include "string/string_column.hpp"
#include "string/string_impl.cuh"

namespace gamax {
namespace {
using namespace gamax::detail;
namespace cg = cooperative_groups;

// Type tag for dispatch
template<typename T> struct type_tag { using type = T; };

// Generic dispatch helper
template<typename Functor>
void dispatch_string_impl(string_impl_id impl, Functor f) {
    switch(impl) {
        case GERMAN_STRING:
            f(type_tag<typename id_to_string_impl<GERMAN_STRING>::string_col_impl>{});
            break;
        case ARROW_STRING:
            f(type_tag<typename id_to_string_impl<ARROW_STRING>::string_col_impl>{});
            break;
        default:
            throw std::runtime_error("Unsupported string implementation");
    }
}

template<typename string_col_impl_t>
class string_agent {
public:
    using string_view_t = typename string_col_impl_t::string_view_type;
    using mutable_string_view_t = typename string_col_impl_t::mutable_string_view_type;

    __device__ __forceinline__ string_agent(string_column_view& col)
        : _impl(col) {};

    template<int GroupSize, typename Group>
    __device__ __forceinline__ bool startswith(Group const& g, size_type i, string_scalar_view const& pattern) const {
        auto str = _impl.viewAt(i);
        const auto pattern_len = pattern.size();
        const auto str_len = str.size();
        if (pattern_len > str_len) return false;

        bool match = true;
        int idx = g.thread_rank();
        #pragma unroll
        for (; match && idx < str.prefix_len() && idx < pattern_len; idx += g.size()) {
            match = (str.charAt(idx, from_header_tag{}) == pattern[idx]);
        }
        #pragma unroll
        for (; match && str_len < str.max_inlined_len() && idx < pattern_len; idx += g.size()) {
            match = (str.charAt(idx, from_header_tag{}) == pattern[idx]);
        }
        #pragma unroll
        for (; match && idx < pattern_len; idx += g.size()) {
            match = (str.charAt(idx, from_buffer_tag{}) == pattern[idx]);
        }
        return g.all(match);
    }

    __device__ __forceinline__ bool startswith(size_type i, string_scalar_view const& pattern) const {
        auto str = _impl.viewAt(i);
        const auto pattern_len = pattern.size();
        const auto str_len = str.size();
        if (pattern_len > str_len) return false;

        bool match = true;
        int idx = 0;
        #pragma unroll
        for (; match && idx < str.prefix_len() && idx < pattern_len; idx += 1) {
            match = (str.charAt(idx, from_header_tag{}) == pattern[idx]);
        }
        #pragma unroll
        for (; match && str_len < str.max_inlined_len() && idx < pattern_len; idx += 1) {
            match = (str.charAt(idx, from_header_tag{}) == pattern[idx]);
        }
        #pragma unroll
        for (; match && idx < pattern_len; idx += 1) {
            match = (str.charAt(idx, from_buffer_tag{}) == pattern[idx]);
        }
        return match;
        // return str.compare(pattern);
    }

    template<int GroupSize, typename Group>
    __device__ __forceinline__ bool endswith(Group const& g, size_type i, string_scalar_view const& pattern) const {
        auto str = _impl.viewAt(i);
        const auto pattern_len = pattern.size();
        const auto str_len = str.size();
        if (pattern_len > str_len) return false;

        bool match = true;
        const auto offset = str_len - pattern_len;
        int idx = g.thread_rank();
        int stridx = offset + idx;
        
        #pragma unroll
        for (; match && stridx < str.prefix_len() && idx < pattern_len; idx += g.size(), stridx += g.size()) {
            match = (str.charAt(stridx, from_header_tag{}) == pattern[idx]);
        }
        #pragma unroll
        for (; match && str_len < str.max_inlined_len() && idx < pattern_len; idx += g.size(), stridx += g.size()) {
            match = (str.charAt(stridx, from_header_tag{}) == pattern[idx]);
        }
        #pragma unroll
        for (; match && idx < pattern_len; idx += g.size(), stridx += g.size()) {
            match = (str.charAt(stridx, from_buffer_tag{}) == pattern[idx]);
        }
        return g.all(match);
    }

    __device__ __forceinline__ bool endswith(size_type i, string_scalar_view const& pattern) const {
        auto str = _impl.viewAt(i);
        const auto pattern_len = pattern.size();
        const auto str_len = str.size();
        if (pattern_len > str_len) return false;

        bool match = true;
        const auto offset = str_len - pattern_len;
        int idx = 0;
        int stridx = offset;
        
        #pragma unroll
        for (; match && stridx < str.prefix_len() && idx < pattern_len; idx += 1, stridx += 1) {
            match = (str.charAt(stridx, from_header_tag{}) == pattern[idx]);
        }
        #pragma unroll
        for (; match && str_len < str.max_inlined_len() && idx < pattern_len; idx += 1, stridx += 1) {
            match = (str.charAt(stridx, from_header_tag{}) == pattern[idx]);
        }
        #pragma unroll
        for (; match && idx < pattern_len; idx += 1, stridx += 1) {
            match = (str.charAt(stridx, from_buffer_tag{}) == pattern[idx]);
        }
        return match;
    }

    __device__ __forceinline__ bool contains(size_type i, string_scalar_view const& pattern) const {
        auto str = _impl.viewAt(i);
        const auto pattern_len = pattern.size();
        const auto str_len = str.size();
        if (pattern_len > str_len) return false;
        for (size_type k = 0; k <= str_len - pattern_len; ++k) {
            bool match = true;
            
            // if(str_len > str.max_inlined_len() && k >= str.prefix_len()) {
            //     #pragma unroll
            //     for (size_type j = 0; j < pattern_len; ++j) {
            //         if (str.charAt(k + j, from_buffer_tag{}) != pattern[j]) {
            //             match = false;
            //             break;
            //         }
            //     }
            // } else {
                #pragma unroll
                for (size_type j = 0; j < pattern_len; ++j) {
                    if (str[k + j] != pattern[j]) {
                        match = false;
                        break;
                    }
                }
            // }
            if (match) return true;
        }
        return false;
    }

    // KMP version
    __device__ __forceinline__ bool contains(size_type idx, string_scalar_view const& pattern, int* pf) const {
        auto str = _impl.viewAt(idx);
        const auto pattern_len = pattern.size();
        const auto str_len = str.size();
        if (pattern_len > str_len) return false;
        int i = 0, j = 0;
        while (i < str_len) {
            if (str[i] == pattern[j]) {
                i++;
                j++;
                if (j == pattern_len) {
                    return true;
                }
            } else {
                j = pf[j];
                if (j < 0) {
                    i++;
                    j++;
                }
            }
        }
        return false;
    }

private:
    string_col_impl_t _impl;
};

template<typename string_col_impl_t, int GroupSize>
__global__ void startswith_kernel(
    string_column_view col,
    string_scalar_view prefix,
    mutable_column_view out)
{
    if constexpr (GroupSize == 1) {
        string_agent<string_col_impl_t> agent(col);

        for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
            idx < col.size(); 
            idx += blockDim.x * gridDim.x) 
        {
            out.data<bool>()[idx]= agent.startswith(idx, prefix);
        }
    } else {
        __shared__ char smem[256];
        string_scalar_view prefix_smem(smem, prefix.size());
        
        // Load prefix into shared memory
        #pragma unroll
        for (size_type i = threadIdx.x; i < prefix.size(); i += blockDim.x) {
            smem[i] = prefix.data()[i];
        }
        __syncthreads();

        auto block = cg::this_thread_block();
        auto tile = cg::tiled_partition<GroupSize>(block);

        string_agent<string_col_impl_t> agent(col);
        
        // Calculate global thread index
        size_type global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Calculate which string this group is processing
        size_type string_idx = global_tid / GroupSize;

        // Stride by total number of groups in the grid
        size_type total_groups = (gridDim.x * blockDim.x) / GroupSize;

        for(; string_idx < col.size(); string_idx += total_groups) {
            bool result = agent.startswith<GroupSize>(tile, string_idx, prefix);
            tile.sync();
            if (tile.thread_rank() == 0) {
                out.data<bool>()[string_idx] = result;
            }
        }
    }
}

template<typename string_col_impl_t, int GroupSize>
__global__ void endswith_kernel(
    string_column_view col,
    string_scalar_view suffix,
    mutable_column_view out)
{
    if constexpr (GroupSize == 1) {
        string_agent<string_col_impl_t> agent(col);

        for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
            idx < col.size(); 
            idx += blockDim.x * gridDim.x) 
        {
            out.data<bool>()[idx]= agent.endswith(idx, suffix);
        }
    } else {
        __shared__ char smem[256];
        string_scalar_view suffix_smem(smem, suffix.size());
        
        // Load suffix into shared memory
        for (size_type i = threadIdx.x; i < suffix.size(); i += blockDim.x) {
            smem[i] = suffix.data()[i];
        }
        __syncthreads();

        auto block = cg::this_thread_block();
        auto tile = cg::tiled_partition<GroupSize>(block);

        string_agent<string_col_impl_t> agent(col);
        
        // Calculate global thread index
        size_type global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Calculate which string this group is processing
        size_type string_idx = global_tid / GroupSize;

        // Stride by total number of groups in the grid
        size_type total_groups = (gridDim.x * blockDim.x) / GroupSize;

        for(; string_idx < col.size(); string_idx += total_groups) {
            bool result = agent.endswith<GroupSize>(tile, string_idx, suffix_smem);
            tile.sync();
            if (tile.thread_rank() == 0) {
                out.data<bool>()[string_idx] = result;
            }
        }
    }
}

template<typename string_col_impl_t, int GroupSize>
__global__ void contains_kernel(
    string_column_view col,
    string_scalar_view substring,
    mutable_column_view out)
{
    __shared__ char smem[256];
    string_scalar_view substring_smem(smem, substring.size());
    
    for (size_type i = threadIdx.x; i < substring.size(); i += blockDim.x) {
        smem[i] = substring.data()[i];
    }
    __syncthreads();

    string_agent<string_col_impl_t> agent(col);
    for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < col.size(); 
        idx += blockDim.x * gridDim.x) 
    {
        out.data<bool>()[idx] = agent.contains(idx, substring_smem);
    }
}

template<typename string_col_impl_t, int GroupSize>
__global__ void contains_kmp_kernel(
    string_column_view col,
    string_scalar_view substring,
    mutable_column_view out)
{
    __shared__ char smem[256];
    string_scalar_view substring_smem(smem, substring.size());
    for (size_type i = threadIdx.x; i < substring.size(); i += blockDim.x) {
        smem[i] = substring.data()[i];
    }
    __syncthreads();

    __shared__ int pf[256];
    // Preprocess the pattern to create the partial match table
    if (threadIdx.x == 0) {
        int m = substring_smem.size();
        pf[0] = -1;
        int j = -1;
        for (int i = 1; i < m; ++i) {
            while (j >= 0 && substring_smem[i] != substring_smem[j + 1]) {
                j = pf[j];
            }
            if (substring_smem[i] == substring_smem[j + 1]) {
                j++;
            }
            pf[i] = j;
        }
    }
    __syncthreads();
    

    string_agent<string_col_impl_t> agent(col);
    for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < col.size(); 
        idx += blockDim.x * gridDim.x) 
    {
        out.data<bool>()[idx] = agent.contains(idx, substring_smem, pf);
    }
}

// Generic Host Helper using launcher policy
template<typename Launcher>
column_ptr perform_string_predicate(
    gamax::string_column_view in,
    gamax::string_scalar_view param,
    cudaStream_t stream,
    string_op_options options)
{
    auto out = gamax::make_column(
        gamax::data_type{gamax::type_id::BOOL8},
        in.size());
    mutable_column_view out_view = out->mutable_view();

    dispatch_string_impl(in.impl(), [&](auto tag) {
        using string_col_impl = typename decltype(tag)::type;
        constexpr int block_size = 256;
        const int grid_size = (in.size() + block_size - 1) / block_size;
        switch(options.cg_size) {
            case 1:
                Launcher::template launch<string_col_impl, 1>(
                    in, param, out_view, stream, options, grid_size, block_size);
                break;
            case 2:
                Launcher::template launch<string_col_impl, 2>(
                    in, param, out_view, stream, options, grid_size, block_size);
                break;
            case 4:
                Launcher::template launch<string_col_impl, 4>(
                    in, param, out_view, stream, options, grid_size, block_size);
                break;
            case 8:
                Launcher::template launch<string_col_impl, 8>(
                    in, param, out_view, stream, options, grid_size, block_size);
                break;
            case 16:
                Launcher::template launch<string_col_impl, 16>(
                    in, param, out_view, stream, options, grid_size, block_size);
                break;
            case 32:
                Launcher::template launch<string_col_impl, 32>(
                    in, param, out_view, stream, options, grid_size, block_size);
                break;
            default:
                throw std::runtime_error("Unsupported cooperative group size");
        }
    });

    cudaStreamSynchronize(stream);
    return out;
}

// Launcher policies
struct StartsWithLauncher {
    template<typename Impl, int CG = 1>
    static void launch(
        string_column_view in,
        string_scalar_view param,
        mutable_column_view out,
        cudaStream_t stream,
        string_op_options options,
        int grid_size,
        int block_size)
    {
        startswith_kernel<Impl, CG><<<grid_size, block_size, 0, stream>>>(in, param, out);
    }
};

struct EndsWithLauncher {
    template<typename Impl, int CG = 1>
    static void launch(
        string_column_view in,
        string_scalar_view param,
        mutable_column_view out,
        cudaStream_t stream,
        string_op_options options,
        int grid_size,
        int block_size)
    {
        endswith_kernel<Impl, CG><<<grid_size, block_size, 0, stream>>>(in, param, out);
    }
};

struct ContainsLauncher {
    template<typename Impl, int CG = 1>
    static void launch(
        string_column_view in,
        string_scalar_view param,
        mutable_column_view out,
        cudaStream_t stream,
        string_op_options options,
        int grid_size,
        int block_size)
    {
        if(options.use_kmp) {
            contains_kmp_kernel<Impl, CG><<<grid_size, block_size, 0, stream>>>(in, param, out);
        } else {
            contains_kernel<Impl, CG><<<grid_size, block_size, 0, stream>>>(in, param, out);
        }
    }
};

} // namespace

column_ptr startswith(
    gamax::string_column_view in,
    gamax::string_scalar_view prefix,
    cudaStream_t stream,
    string_op_options options) 
{
    return perform_string_predicate<StartsWithLauncher>(in, prefix, stream, options);
}

column_ptr endswith(
    gamax::string_column_view in,
    gamax::string_scalar_view suffix,
    cudaStream_t stream,
    string_op_options options)
{
    return perform_string_predicate<EndsWithLauncher>(in, suffix, stream, options);
}

column_ptr contains(
    gamax::string_column_view in,
    gamax::string_scalar_view substring,
    cudaStream_t stream,
    string_op_options options)
{
    return perform_string_predicate<ContainsLauncher>(in, substring, stream, options);
}
} // namespace gamax