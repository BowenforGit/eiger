#include <cuda.h>
#include <vector>
#include <string>
#include <string_view>
#include <memory>
#include <stdexcept>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "string/string_column.hpp"
#include "string/string_impl.cuh"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"

namespace gamax {
namespace {
template<typename german_string_impl_t, typename other_string_impl_t>
__global__ void create_german_string_kernel(
    string_column_view in, // arrow string format
    column_view out_offsets, // german string offsets
    mutable_string_column_view out // german string format
)
{
    using length_t = typename german_string_impl_t::LengthType;
    using buffer_index_t = typename german_string_impl_t::BufferIndexType;

    auto arrow_string_impl = other_string_impl_t(in);
    auto german_string_impl = german_string_impl_t(out);

    for(auto i = get_cuda_tid(); i < in.size(); i += nthreads()) {
        auto str_in = arrow_string_impl.viewAt(i);
        auto str_out = german_string_impl.mutableViewAt(i);
        auto& header = str_out.header();

        header.set_length(static_cast<length_t>(str_in.size()));
        header.set_offset(out_offsets.data<buffer_index_t>()[i]);

        #pragma unroll
        for(int32_t k = 0; k < str_in.size(); k++) {
            str_out[k] = str_in[k];
        }
    } 
}

template<
    typename german_string_impl_t, 
    typename input_length_t = typename german_string_impl_t::LengthType>
column_ptr get_german_string_offsets(column_view strlen, cudaStream_t stream) 
{
    using buffer_index_t = typename german_string_impl_t::BufferIndexType;
    using str_length_t = typename german_string_impl_t::LengthType;

    constexpr auto inline_limit = german_string_impl_t::max_inlined_len();
    constexpr auto prefix_len = german_string_impl_t::prefix_len();
    
    auto overflow_strlen = gamax::make_column<str_length_t>(strlen.size(), false, stream);
    thrust::transform(
        thrust::cuda::par(*get_thrust_allocator()).on(stream),
        strlen.data<input_length_t>(),
        strlen.data<input_length_t>() + strlen.size(),
        overflow_strlen->template data<str_length_t>(),
        [] __device__ (input_length_t l) {
            auto res = (l <= inline_limit) ? 0 : (l - prefix_len);
            return static_cast<str_length_t>(res);
        });
        
    auto offsets = gamax::make_column<buffer_index_t>(strlen.size());
    thrust::exclusive_scan(
        thrust::cuda::par(*get_thrust_allocator()).on(stream),
        overflow_strlen->template data<str_length_t>(),
        overflow_strlen->template data<str_length_t>() + strlen.size(),
        offsets->template data<buffer_index_t>());

    return offsets;
}

template<
    typename arrow_string_impl_t, 
    typename input_length_t = typename arrow_string_impl_t::LengthType>
column_ptr get_arrow_string_offsets(column_view strlen, cudaStream_t stream) 
{
    using buffer_index_t = typename arrow_string_impl_t::BufferIndexType;
    using str_length_t = typename arrow_string_impl_t::LengthType;
    
    auto offsets = gamax::make_column<buffer_index_t>(strlen.size() + 1, true, stream);
    thrust::inclusive_scan(
        thrust::cuda::par(*get_thrust_allocator()).on(stream),
        strlen.data<str_length_t>(),
        strlen.data<str_length_t>() + strlen.size(),
        offsets->template data<buffer_index_t>() + 1);

    return offsets;
}

template<
    typename arrow_string_impl_t, 
    typename input_length_t = typename arrow_string_impl_t::LengthType>
std::unique_ptr<string_column> make_arrow_string_column(
    const char* h_data,
    const input_length_t* h_strlen,
    const size_type num_strings,
    cudaStream_t stream)
{
    using buffer_index_t = typename arrow_string_impl_t::BufferIndexType;
    using str_length_t = typename arrow_string_impl_t::LengthType;

    auto d_strlen = gamax::make_column_from_host(h_strlen, num_strings, stream);
    auto offsets = get_arrow_string_offsets<arrow_string_impl_t, input_length_t>(d_strlen->view(), stream);

    buffer_index_t total_bytes = 0;
    cudaMemcpyAsync(
        &total_bytes, 
        offsets->template data<buffer_index_t>() + num_strings, 
        sizeof(buffer_index_t), 
        cudaMemcpyDeviceToHost, 
        stream);
    cudaStreamSynchronize(stream);

    gamax::device_buffer d_data(total_bytes, false, stream);
    cudaMemcpyAsync(
        d_data.data(), 
        h_data, 
        total_bytes, 
        cudaMemcpyHostToDevice, 
        stream);
    
    cudaStreamSynchronize(stream);
    return std::make_unique<string_column>(
        std::move(d_data),
        std::move(offsets->buffer()),
        num_strings,
        string_impl_id::ARROW_STRING);
}

template<
    typename german_string_impl_t, 
    typename other_string_impl_t,
    typename input_length_t = typename german_string_impl_t::LengthType>
std::unique_ptr<string_column> make_german_string_column(
    const input_length_t* h_strlen,
    string_column_view in, // another string format
    cudaStream_t stream)
{
    using buffer_index_t = typename german_string_impl_t::BufferIndexType;
    using str_length_t = typename german_string_impl_t::LengthType;

    auto d_strlen = gamax::make_column_from_host(h_strlen, in.size(), stream);
    auto offsets = get_german_string_offsets<german_string_impl_t, input_length_t>(d_strlen->view(), stream);

    auto meta_buffer = gamax::device_buffer(
        in.size() * sizeof(typename german_string_impl_t::header_entry_type), 
        false, 
        stream);
    auto data_buffer = gamax::device_buffer(
        in.bytes(), 
        false, 
        stream);
    auto out = std::make_unique<string_column>(
        std::move(data_buffer),
        std::move(meta_buffer),
        in.size(),
        string_impl_id::GERMAN_STRING);
    
    create_german_string_kernel<german_string_impl_t, other_string_impl_t>
        <<<num_tb(in.size(), 256), 256, 0, stream>>>(
            in,
            offsets->view(),
            out->mutable_view_s());
    
    cudaStreamSynchronize(stream);

    return out;
}
} // namespace

std::unique_ptr<string_scalar> make_string_scalar(
    std::string_view str, 
    cudaStream_t stream)
{
    return std::make_unique<string_scalar>(str, stream);
}

column_ptr make_string_column(
    const char* flat_data,
    const int32_t* lengths,
    size_type num_strings,
    cudaStream_t stream,
    string_impl_id impl) 
{
    using arrow_string_impl_t = typename gamax::detail::id_to_string_impl<string_impl_id::ARROW_STRING>::string_col_impl;
    auto arrow = make_arrow_string_column<arrow_string_impl_t>(
        flat_data,
        lengths,
        num_strings,
        stream);
    
    if(impl == string_impl_id::ARROW_STRING) {
        return arrow;
    } else if(impl == string_impl_id::GERMAN_STRING) {
        using german_string_impl_t = typename gamax::detail::id_to_string_impl<string_impl_id::GERMAN_STRING>::string_col_impl;
        return make_german_string_column<german_string_impl_t, arrow_string_impl_t>(
            lengths,
            arrow->view_s(),
            stream);
    } else {
        throw std::runtime_error("Unsupported string implementation");
    }
}
} // namespace gamax