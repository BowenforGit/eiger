#include <cuda.h>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

#include "string/string.hpp"
#include "string/string_column.hpp"
#include "string/string_impl.cuh"

namespace gamax {
namespace {
using namespace gamax::detail;

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

    __device__ __forceinline__ bool startswith(size_type i, string_scalar_view const& prefix) const {
        auto str = _impl.viewAt(i);
        const auto prefix_len = prefix.size();
        const auto str_len = str.size();
        if (prefix_len > str_len) return false;
        for (size_type i = 0; i < prefix_len; ++i) {
            if (str[i] != prefix[i]) return false;
        }
        return true;
    }

    __device__ __forceinline__ bool endswith(size_type i, string_scalar_view const& suffix) const {
        auto str = _impl.viewAt(i);
        const auto suffix_len = suffix.size();
        const auto str_len = str.size();
        if (suffix_len > str_len) return false;
        for (size_type i = 0; i < suffix_len; ++i) {
            if (str[str_len - suffix_len + i] != suffix[i]) return false;
        }
        return true;
    }

    __device__ __forceinline__ bool contains(size_type i, string_scalar_view const& substring) const {
        auto str = _impl.viewAt(i);
        const auto substring_len = substring.size();
        const auto str_len = str.size();
        if (substring_len > str_len) return false;
        for (size_type i = 0; i <= str_len - substring_len; ++i) {
            bool match = true;
            for (size_type j = 0; j < substring_len; ++j) {
                if (str[i + j] != substring[j]) {
                    match = false;
                    break;
                }
            }
            if (match) return true;
        }
        return false;
    }

private:
    string_col_impl_t _impl;
};

template<typename string_col_impl_t>
__global__ void startswith_kernel(
    string_column_view col,
    string_scalar_view prefix,
    mutable_column_view out)
{
    string_agent<string_col_impl_t> agent(col);
    for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < col.size(); 
        idx += blockDim.x * gridDim.x) 
    {
        out.data<bool>()[idx] = agent.startswith(idx, prefix);
    }
}

template<typename string_col_impl_t>
__global__ void endswith_kernel(
    string_column_view col,
    string_scalar_view suffix,
    mutable_column_view out)
{
    string_agent<string_col_impl_t> agent(col);
    for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < col.size(); 
        idx += blockDim.x * gridDim.x) 
    {
        out.data<bool>()[idx] = agent.endswith(idx, suffix);
    }
}

template<typename string_col_impl_t>
__global__ void contains_kernel(
    string_column_view col,
    string_scalar_view substring,
    mutable_column_view out)
{
    string_agent<string_col_impl_t> agent(col);
    for(size_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < col.size(); 
        idx += blockDim.x * gridDim.x) 
    {
        out.data<bool>()[idx] = agent.contains(idx, substring);
    }
}

// Generic Host Helper using launcher policy
template<typename Launcher>
column_ptr perform_string_predicate(
    gamax::string_column_view in,
    gamax::string_scalar_view param,
    cudaStream_t stream)
{
    auto out = gamax::make_column(
        gamax::data_type{gamax::type_id::BOOL8},
        in.size());
    mutable_column_view out_view = out->mutable_view();

    dispatch_string_impl(in.impl(), [&](auto tag) {
        using string_col_impl = typename decltype(tag)::type;
        constexpr int block_size = 256;
        const int grid_size = (in.size() + block_size - 1) / block_size;
        Launcher::template launch<string_col_impl>(
            in, param, out_view, stream, grid_size, block_size);
    });

    cudaStreamSynchronize(stream);
    return out;
}

// Launcher policies
struct StartsWithLauncher {
    template<typename Impl>
    static void launch(
        string_column_view in,
        string_scalar_view param,
        mutable_column_view out,
        cudaStream_t stream,
        int grid_size,
        int block_size)
    {
        startswith_kernel<Impl><<<grid_size, block_size, 0, stream>>>(in, param, out);
    }
};

struct EndsWithLauncher {
    template<typename Impl>
    static void launch(
        string_column_view in,
        string_scalar_view param,
        mutable_column_view out,
        cudaStream_t stream,
        int grid_size,
        int block_size)
    {
        endswith_kernel<Impl><<<grid_size, block_size, 0, stream>>>(in, param, out);
    }
};

struct ContainsLauncher {
    template<typename Impl>
    static void launch(
        string_column_view in,
        string_scalar_view param,
        mutable_column_view out,
        cudaStream_t stream,
        int grid_size,
        int block_size)
    {
        contains_kernel<Impl><<<grid_size, block_size, 0, stream>>>(in, param, out);
    }
};

} // namespace

column_ptr startswith(
    gamax::string_column_view in,
    gamax::string_scalar_view prefix,
    cudaStream_t stream) 
{
    return perform_string_predicate<StartsWithLauncher>(in, prefix, stream);
}

column_ptr endswith(
    gamax::string_column_view in,
    gamax::string_scalar_view suffix,
    cudaStream_t stream)
{
    return perform_string_predicate<EndsWithLauncher>(in, suffix, stream);
}

column_ptr contains(
    gamax::string_column_view in,
    gamax::string_scalar_view substring,
    cudaStream_t stream)
{
    return perform_string_predicate<ContainsLauncher>(in, substring, stream);
}
} // namespace gamax