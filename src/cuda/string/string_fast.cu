#include <cuda.h>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cooperative_groups.h>

#include "string/string.hpp"
#include "string/string_column.hpp"
#include "string/string_impl.cuh"
#include "types/vec_type.hpp"
#include "utils/utils.cuh"

namespace gamax {
namespace {
using namespace gamax::detail;
namespace cg = cooperative_groups;

constexpr int kMaxPatternLen = 256;
constexpr int kMaxNumPatterns = 8;

using string_col_impl_t = typename id_to_string_impl<ARROW_STRING>::string_col_impl;

template<int kAlignAt>
struct pattern_helper {
    __device__ __forceinline__ pattern_helper(
        string_scalar_view pattern,
        char buffer[][kMaxPatternLen]) 
        : _buffer(reinterpret_cast<char*>(buffer)), _size(pattern.size()) 
    {
        const char* pattern_ptr = pattern.data();
        
        for(int i = threadIdx.x; i < _size; i += blockDim.x) {
            for(int k = 0; i >= k && k < kAlignAt; ++k) {
                buffer[k][i - k] = pattern_ptr[i];
            }
        }
        __syncthreads();
    }

    __device__ __forceinline__ const char* get_ptr_at_kth_char_aligned(int kth) const {
        const auto aligned_offset = (kth % kAlignAt);
        return _buffer + aligned_offset * kMaxPatternLen + (kth - aligned_offset);
    }

    __device__ __forceinline__ const char& operator[](int index) const {
        return _buffer[index];
    }

    __device__ __forceinline__ int size() const {
        return _size;
    }

    __device__ __forceinline__ bool compare(const char* other) const {
        const char* target = other;
        int idx = 0;
        for(; reinterpret_cast<uintptr_t>(target) % kAlignAt != 0 && idx < _size; ++idx, ++target) {
            if(_buffer[idx] != *target) return false;
        }

        const auto remaining_size = _size - idx;
        const char* aligned_ptr = get_ptr_at_kth_char_aligned(idx);
        
        return compare_aligned(aligned_ptr, target, remaining_size);
    }

    template<typename Group>
    __device__ __forceinline__ bool compare(Group const& g, const char* other) const {
        const char* target = other;
        
        const int unaligned_prologue_size = (reinterpret_cast<uintptr_t>(target) % kAlignAt == 0) ? 
                                            0 : (kAlignAt - (reinterpret_cast<uintptr_t>(target) % kAlignAt));
        
        bool match = true;

        int idx = g.thread_rank();
        for(; idx < unaligned_prologue_size && idx < _size; idx += g.size()) {
            if(_buffer[idx] != target[idx]) {
                match = false;
                break;
            };
        }
        if(!g.all(match)) return false;

        const auto remaining_size = _size - unaligned_prologue_size;
        const char* aligned_ptr = get_ptr_at_kth_char_aligned(unaligned_prologue_size);
        const char* target_aligned_ptr = target + unaligned_prologue_size;
        
        return compare_aligned(g, aligned_ptr, target_aligned_ptr, remaining_size);
    }

private:
    char* _buffer;
    int _size;

    __device__ __forceinline__ bool compare_aligned(
        const char* this_ptr, 
        const char* other_ptr, 
        int size) const 
    {
        using T = widest_int_for_alignment_t<kAlignAt>;
        const T* this_ptr_wide = reinterpret_cast<const T*>(this_ptr);
        const T* other_ptr_wide = reinterpret_cast<const T*>(other_ptr);
        
        const int n_chunks = size / kAlignAt;
        for(int i = 0; i < n_chunks; ++i) {
            if(this_ptr_wide[i] != other_ptr_wide[i]) {
                return false;
            }
        }
        
        int remaining = size % kAlignAt;
        const char* this_ptr_char = reinterpret_cast<const char*>(this_ptr_wide + n_chunks);
        const char* other_ptr_char = reinterpret_cast<const char*>(other_ptr_wide + n_chunks);
        for(int i = 0; i < remaining; ++i) {
            if(this_ptr_char[i] != other_ptr_char[i]) {
                return false;
            }
        }
        
        return true;
    }

    template<typename Group>
    __device__ __forceinline__ bool compare_aligned(
        Group const& g,
        const char* this_ptr, 
        const char* other_ptr, 
        int size) const 
    {
        using T = widest_int_for_alignment_t<kAlignAt>;
        const T* this_ptr_wide = reinterpret_cast<const T*>(this_ptr);
        const T* other_ptr_wide = reinterpret_cast<const T*>(other_ptr);
        
        bool match = true;
        const int n_chunks = size / kAlignAt;
        for(int i = g.thread_rank(); i < n_chunks; i += g.size()) {
            if(this_ptr_wide[i] != other_ptr_wide[i]) {
                match = false;
                break;
            }
        }
        if(!g.all(match)) return false;
        
        int remaining = size % kAlignAt;
        if(remaining == 0) return true;

        const char* this_ptr_char = reinterpret_cast<const char*>(this_ptr_wide + n_chunks);
        const char* other_ptr_char = reinterpret_cast<const char*>(other_ptr_wide + n_chunks);
        for(int i = g.thread_rank(); i < remaining; i += g.size()) {
            if(this_ptr_char[i] != other_ptr_char[i]) {
                match = false;
                break;
            }
        }

        if(!g.all(match)) return false;        
        return true;
    }
};

// Helper for handling multiple patterns in shared memory
// Uses a simplified approach without alignment optimization for clarity
template<int kAlignAt>
struct multi_pattern_helper {
    __device__ __forceinline__ multi_pattern_helper(
        const string_scalar_view* patterns,
        int num_patterns,
        char buffer[][kMaxPatternLen],
        int* sizes)
        : _buffer(reinterpret_cast<char*>(buffer))
        , _sizes(sizes)
        , _num_patterns(num_patterns)
    {
        // Copy pattern sizes to shared memory
        if(threadIdx.x < num_patterns) {
            _sizes[threadIdx.x] = patterns[threadIdx.x].size();
        }
        __syncthreads();
        
        // Copy all patterns to shared memory
        // Each pattern is stored at buffer[pattern_idx * kMaxPatternLen]
        for(int p = 0; p < num_patterns; ++p) {
            const char* pattern_ptr = patterns[p].data();
            const int pattern_size = patterns[p].size();
            for(int i = threadIdx.x; i < pattern_size; i += blockDim.x) {
                buffer[p][i] = pattern_ptr[i];
            }
        }
        __syncthreads();
    }

    __device__ __forceinline__ int num_patterns() const {
        return _num_patterns;
    }

    __device__ __forceinline__ int pattern_size(int pattern_idx) const {
        return _sizes[pattern_idx];
    }

    __device__ __forceinline__ const char* pattern_data(int pattern_idx) const {
        return _buffer + pattern_idx * kMaxPatternLen;
    }

    // Compare pattern at pattern_idx with string at str_ptr
    __device__ __forceinline__ bool compare(int pattern_idx, const char* str_ptr) const {
        const char* pattern_ptr = pattern_data(pattern_idx);
        const int size = _sizes[pattern_idx];
        
        for(int i = 0; i < size; ++i) {
            if(pattern_ptr[i] != str_ptr[i]) {
                return false;
            }
        }
        return true;
    }

private:
    char* _buffer;
    int* _sizes;
    int _num_patterns;
};

template<int GroupSize, int kAlignAt>
__global__ void equal_kernel(
    string_column_view col,
    string_scalar_view pattern,
    mutable_column_view out)
{
    __shared__ char pattern_buffer[kAlignAt][kMaxPatternLen];
    pattern_helper<kAlignAt> ph(pattern, pattern_buffer);

    string_col_impl_t impl(col);
    
    bool* out_data = out.data<bool>();
    
    if constexpr (GroupSize == 1) {
        for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
            idx < col.size(); 
            idx += blockDim.x * gridDim.x) 
        {
            auto str = impl.viewAt(idx);
            out_data[idx] = (pattern.size() == str.size()) && ph.compare(str.data());
        }
    } else {
        auto g = cg::tiled_partition<GroupSize>(cg::this_thread_block());

        for(size_type idx = get_cuda_tid() / GroupSize; 
            idx < col.size(); 
            idx += nthreads() / GroupSize) 
        {
            auto str = impl.viewAt(idx);
            bool match = (pattern.size() == str.size()) && ph.compare(g, str.data());
            g.sync();
            if(g.thread_rank() == 0) {
                out_data[idx] = match;
            }
        }
    }
}

template<int GroupSize, int kAlignAt>
__global__ void startswith_kernel(
    string_column_view col,
    string_scalar_view pattern,
    mutable_column_view out)
{
    __shared__ char pattern_buffer[kAlignAt][kMaxPatternLen];
    pattern_helper<kAlignAt> ph(pattern, pattern_buffer);

    string_col_impl_t impl(col);
    
    bool* out_data = out.data<bool>();
    
    if constexpr (GroupSize == 1) {
        for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
            idx < col.size(); 
            idx += blockDim.x * gridDim.x) 
        {
            auto str = impl.viewAt(idx);
            out_data[idx] = (pattern.size() <= str.size()) && ph.compare(str.data());
        }
    } else {
        auto g = cg::tiled_partition<GroupSize>(cg::this_thread_block());

        for(size_type idx = get_cuda_tid() / GroupSize; 
            idx < col.size(); 
            idx += nthreads() / GroupSize) 
        {
            auto str = impl.viewAt(idx);
            bool match = (pattern.size() <= str.size()) && ph.compare(g, str.data());
            g.sync();
            if(g.thread_rank() == 0) {
                out_data[idx] = match;
            }
        }
    }
}

template<int GroupSize, int kAlignAt>
__global__ void endswith_kernel(
    string_column_view col,
    string_scalar_view pattern,
    mutable_column_view out)
{
    __shared__ char pattern_buffer[kAlignAt][kMaxPatternLen];
    pattern_helper<kAlignAt> ph(pattern, pattern_buffer);

    string_col_impl_t impl(col);
    
    bool* out_data = out.data<bool>();
    
    if constexpr (GroupSize == 1) {
        for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
            idx < col.size(); 
            idx += blockDim.x * gridDim.x) 
        {
            auto str = impl.viewAt(idx);
            out_data[idx] = (pattern.size() <= str.size()) && ph.compare(str.data() + (str.size() - pattern.size()));
        }
    } else {
        auto g = cg::tiled_partition<GroupSize>(cg::this_thread_block());

        for(size_type idx = get_cuda_tid() / GroupSize; 
            idx < col.size(); 
            idx += nthreads() / GroupSize) 
        {
            auto str = impl.viewAt(idx);
            bool match = (pattern.size() <= str.size()) && ph.compare(g, str.data() + (str.size() - pattern.size()));
            g.sync();
            if(g.thread_rank() == 0) {
                out_data[idx] = match;
            }
        }
    }
}

template<int kAlignAt=4>
__global__ void contains_kernel(
    string_column_view col,
    string_scalar_view substring,
    mutable_column_view out)
{
    __shared__ char pattern_buffer[kAlignAt][kMaxPatternLen];
    pattern_helper<kAlignAt> ph(substring, pattern_buffer);

    string_col_impl_t impl(col);
    
    bool* out_data = out.data<bool>();
    const auto pattern_len = substring.size();

    for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < col.size(); 
        idx += blockDim.x * gridDim.x) 
    {
        out_data[idx] = false;
        
        auto str = impl.viewAt(idx);
        const char* str_ptr = str.data();
        const auto str_len = str.size();
        
        if (pattern_len > str_len) {
            continue;
        }
        
        for (int k = 0; k <= str_len - pattern_len; ++k) {
            if(ph.compare(str_ptr + k)) {
                out_data[idx] = true;
                break;
            }
        }
    }
}

__global__ void contains_kmp_kernel(
    string_column_view col,
    string_scalar_view substring,
    mutable_column_view out)
{
    __shared__ char smem[kMaxPatternLen];
    string_scalar_view substring_smem(smem, substring.size());
    for (size_type i = threadIdx.x; i < substring.size(); i += blockDim.x) {
        smem[i] = substring.data()[i];
    }
    __syncthreads();

    __shared__ int pf[kMaxPatternLen];
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

    string_col_impl_t impl(col);
    const auto pattern_len = substring_smem.size();
    bool* out_data = out.data<bool>();

    for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < col.size(); 
        idx += blockDim.x * gridDim.x) 
    {
        out_data[idx] = false;
        auto str = impl.viewAt(idx);
        const auto str_len = str.size();
        if (pattern_len > str_len) continue;
        int i = 0, j = 0;
        while (i < str_len) {
            if (str[i] == substring_smem[j]) {
                i++;
                j++;
                if (j == pattern_len) {
                    out_data[idx] = true;
                    break;
                }
            } else {
                j = pf[j];
                if (j < 0) {
                    i++;
                    j++;
                }
            }
        }
    }
}

// TODO: Optimize with alignment-aware pattern_helper
template<int kAlignAt=4>
__global__ void contains_multiple_kernel(
    string_column_view col,
    const string_scalar_view* patterns,
    int num_patterns,
    mutable_column_view out)
{
    __shared__ char pattern_buffer[kMaxNumPatterns][kMaxPatternLen];
    __shared__ int pattern_sizes[kMaxNumPatterns];
    
    multi_pattern_helper<kAlignAt> mph(patterns, num_patterns, pattern_buffer, pattern_sizes);

    string_col_impl_t impl(col);
    
    bool* out_data = out.data<bool>();

    for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < col.size(); 
        idx += blockDim.x * gridDim.x) 
    {
        auto str = impl.viewAt(idx);
        const char* str_ptr = str.data();
        const auto str_len = str.size();
        
        bool all_found = true;
        int search_start = 0;  // Where to start searching for the next pattern
        
        // Find each pattern in order
        for(int p = 0; p < num_patterns && all_found; ++p) {
            const int pattern_len = mph.pattern_size(p);
            bool found = false;
            
            // Search for pattern p starting from search_start
            for(int k = search_start; k <= str_len - pattern_len; ++k) {
                if(mph.compare(p, str_ptr + k)) {
                    found = true;
                    // Next pattern search starts after this match (no overlap)
                    search_start = k + pattern_len;
                    break;
                }
            }
            
            if(!found) {
                all_found = false;
            }
        }
        
        out_data[idx] = all_found;
    }
}

template<typename Launcher, typename... Args>
void dispatch_string_ops(
    string_op_options options, 
    Args&&... args)
{
    auto dispatch_access_width = [&](auto cg_size_tag) {
        constexpr auto cg_size = decltype(cg_size_tag)::value;

        switch(options.access_width) {
            case 1:
                Launcher::template launch<cg_size, 1>(std::forward<Args>(args)...);
                break;
            case 2:
                Launcher::template launch<cg_size, 2>(std::forward<Args>(args)...);
                break;
            case 4:
                Launcher::template launch<cg_size, 4>(std::forward<Args>(args)...);
                break;
            case 8:
                Launcher::template launch<cg_size, 8>(std::forward<Args>(args)...);
                break;
            // case 16:
            //     Launcher::template launch<cg_size, 16>(std::forward<Args>(args)...);
            //     break;
            // case 32:
            //     Launcher::template launch<cg_size, 32>(std::forward<Args>(args)...);
            //     break;
            default:
                throw std::runtime_error("Unsupported access width");
        }
    };
    
    switch(options.cg_size) {
        case 1:
            dispatch_access_width(std::integral_constant<int, 1>{});
            break;
        case 2:
            dispatch_access_width(std::integral_constant<int, 2>{});
            break;
        case 4:
            dispatch_access_width(std::integral_constant<int, 4>{});
            break;
        case 8:
            dispatch_access_width(std::integral_constant<int, 8>{});
            break;
        case 16:
            dispatch_access_width(std::integral_constant<int, 16>{});
            break;
        case 32:
            dispatch_access_width(std::integral_constant<int, 32>{});
            break;
        default:
            throw std::runtime_error("Unsupported cooperative group size");
    }
}

struct EqualLauncher {
    template<int CG, int kAlignAt>
    static void launch(
        string_column_view in,
        string_scalar_view param,
        mutable_column_view out,
        cudaStream_t stream)
    {
        constexpr int block_size = 256;
        const int grid_size = (in.size() + block_size - 1) / block_size;
        equal_kernel<CG, kAlignAt><<<grid_size, block_size, 0, stream>>>(in, param, out);
        cudaStreamSynchronize(stream);
    }
};

struct StartsWithLauncher {
    template<int CG, int kAlignAt>
    static void launch(
        string_column_view in,
        string_scalar_view param,
        mutable_column_view out,
        cudaStream_t stream)
    {
        constexpr int block_size = 256;
        const int grid_size = (in.size() + block_size - 1) / block_size;
        startswith_kernel<CG, kAlignAt><<<grid_size, block_size, 0, stream>>>(in, param, out);
        cudaStreamSynchronize(stream);
    }
};

struct EndsWithLauncher {
    template<int CG, int kAlignAt>
    static void launch(
        string_column_view in,
        string_scalar_view param,
        mutable_column_view out,
        cudaStream_t stream)
    {
        constexpr int block_size = 256;
        const int grid_size = (in.size() + block_size - 1) / block_size;
        endswith_kernel<CG, kAlignAt><<<grid_size, block_size, 0, stream>>>(in, param, out);
        cudaStreamSynchronize(stream);
    }
};

struct ContainsLauncher {
    template<int /*CG, ignored*/, int kAlignAt>
    static void launch(
        string_column_view in,
        string_scalar_view param,
        mutable_column_view out,
        bool use_kmp,
        cudaStream_t stream)
    {
        constexpr int block_size = 256;
        const int grid_size = (in.size() + block_size - 1) / block_size;

        if(use_kmp) {
            contains_kmp_kernel<<<grid_size, block_size, 0, stream>>>(in, param, out);
        } else {
            contains_kernel<kAlignAt><<<grid_size, block_size, 0, stream>>>(in, param, out);
        }
        cudaStreamSynchronize(stream);
    }
};

struct ContainsMultipleLauncher {
    template<int /*CG, ignored*/, int kAlignAt>
    static void launch(
        string_column_view in,
        const string_scalar_view* patterns,
        int num_patterns,
        mutable_column_view out,
        cudaStream_t stream)
    {
        constexpr int block_size = 256;
        const int grid_size = (in.size() + block_size - 1) / block_size;
        contains_multiple_kernel<kAlignAt><<<grid_size, block_size, 0, stream>>>(in, patterns, num_patterns, out);
        cudaStreamSynchronize(stream);
    }
};

} // namespace

column_ptr equal(
    gamax::string_column_view in,
    gamax::string_scalar_view string,
    cudaStream_t stream,
    string_op_options options)
{
    auto out = gamax::make_column(
        gamax::data_type{gamax::type_id::BOOL8},
        in.size());
    
    dispatch_string_ops<EqualLauncher>(options, in, string, out->mutable_view(), stream);
    
    return out;
}

column_ptr startswith(
    gamax::string_column_view in,
    gamax::string_scalar_view prefix,
    cudaStream_t stream,
    string_op_options options) 
{
    auto out = gamax::make_column(
        gamax::data_type{gamax::type_id::BOOL8},
        in.size());
    
    dispatch_string_ops<StartsWithLauncher>(options, in, prefix, out->mutable_view(), stream);
    
    return out;
}

column_ptr endswith(
    gamax::string_column_view in,
    gamax::string_scalar_view suffix,
    cudaStream_t stream,
    string_op_options options)
{
    auto out = gamax::make_column(
        gamax::data_type{gamax::type_id::BOOL8},
        in.size());

    dispatch_string_ops<EndsWithLauncher>(options, in, suffix, out->mutable_view(), stream);
    
    return out;
}

column_ptr contains(
    gamax::string_column_view in,
    gamax::string_scalar_view substring,
    cudaStream_t stream,
    string_op_options options)
{
    auto out = gamax::make_column(
        gamax::data_type{gamax::type_id::BOOL8},
        in.size());
    
    dispatch_string_ops<ContainsLauncher>(options, in, substring, out->mutable_view(), options.use_kmp, stream);
    
    return out;
}

column_ptr contains_multiple(
    gamax::string_column_view in,
    const std::vector<gamax::string_scalar_view>& patterns,
    cudaStream_t stream,
    string_op_options options)
{
    if(patterns.empty()) {
        // If no patterns, all strings trivially "contain" them
        auto out = gamax::make_column(
            gamax::data_type{gamax::type_id::BOOL8},
            in.size());
        cudaMemsetAsync(out->mutable_view().data<bool>(), 1, in.size() * sizeof(bool), stream);
        cudaStreamSynchronize(stream);
        return out;
    }
    
    if(patterns.size() > kMaxNumPatterns) {
        throw std::runtime_error("Too many patterns. Maximum supported: " + std::to_string(kMaxNumPatterns));
    }
    
    auto out = gamax::make_column(
        gamax::data_type{gamax::type_id::BOOL8},
        in.size());
    
    // Copy pattern views to device
    gamax::device_buffer d_patterns(patterns.size() * sizeof(string_scalar_view));
    cudaMemcpyAsync(d_patterns.data(), patterns.data(), patterns.size() * sizeof(string_scalar_view), 
                    cudaMemcpyHostToDevice, stream);
    
    dispatch_string_ops<ContainsMultipleLauncher>(
        options, in, reinterpret_cast<const gamax::string_scalar_view*>(d_patterns.data()), static_cast<int>(patterns.size()), out->mutable_view(), stream);
    
    cudaStreamSynchronize(stream);
    
    return out;
}
} // namespace gamax