#include <cub/cub.cuh>

#include "operators/partition/partition.hpp"
#include "operators/sort/sort.hpp"
#include "utils/cub_wrapper.cuh"
#include "types/type_dispatcher.hpp"
#include "types/data_type.hpp"
#include "memory/column.hpp"

namespace gamax::partition {
std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
partition_column_pair(
    gamax::column& key_col, 
    gamax::column& val_col, 
    int begin_bit,
    int end_bit,
    void* d_temp_storage, 
    size_t temp_storage_bytes, 
    cudaStream_t stream)
{
    return partition_column_pair(key_col.view(), val_col.view(), begin_bit, end_bit, d_temp_storage, temp_storage_bytes, stream);
}

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
partition_column_pair(
    gamax::column_view key_col, 
    gamax::column_view val_col, 
    int begin_bit,
    int end_bit,
    void* d_temp_storage, 
    size_t temp_storage_bytes, 
    cudaStream_t stream)
{
    return gamax::sort::sort_column_pair(
        key_col, 
        val_col, 
        begin_bit, 
        end_bit, 
        gamax::sort_order::ASCENDING, 
        d_temp_storage, 
        temp_storage_bytes, 
        stream);
}

std::unique_ptr<gamax::column>
partition_column(
    gamax::column& col,
    int begin_bit,
    int end_bit,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    cudaStream_t stream)
{
    return partition_column(col.view(), begin_bit, end_bit, d_temp_storage, temp_storage_bytes, stream);
}

std::unique_ptr<gamax::column>
partition_column(
    gamax::column_view col,
    int begin_bit,
    int end_bit,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    cudaStream_t stream)
{
    return gamax::sort::sort_column(
        col, 
        begin_bit, 
        end_bit, 
        gamax::sort_order::ASCENDING, 
        d_temp_storage, 
        temp_storage_bytes, 
        stream);
}

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
arg_partition(
    gamax::column& col,
    int begin_bit,
    int end_bit,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    cudaStream_t stream) 
{
    return arg_partition(col.view(), begin_bit, end_bit, d_temp_storage, temp_storage_bytes, stream);
}

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
arg_partition(
    gamax::column_view col,
    int begin_bit,
    int end_bit,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    cudaStream_t stream) 
{
    auto idx_col = gamax::make_sequence(col.size());
    return partition_column_pair(col, idx_col->view(), begin_bit, end_bit, d_temp_storage, temp_storage_bytes, stream);
}

namespace {
template<typename T>
struct RadixExtractor
{   
    int begin_bit;
    int mask;
    __host__ __device__ RadixExtractor(int begin_bit, int end_bit) : begin_bit(begin_bit), mask((1 << (end_bit - begin_bit)) - 1) {}
    __host__ __device__ __forceinline__
    T operator()(const T &a) const {
        return (a >> begin_bit) & mask;
    }
};

struct compute_partition_offsets_impl {
    template<typename T>
    std::unique_ptr<gamax::column> operator()(
        gamax::column_view partitioned_key_col,
        int begin_bit,
        int end_bit,
        cudaStream_t stream)
    {
        const auto n_partitions = (1 << (end_bit - begin_bit));
        const auto num_items = partitioned_key_col.size();

        auto d_counts_out_ptr = gamax::make_idx_column(n_partitions);
        auto offsets = gamax::make_idx_column(n_partitions+1);
       
        RadixExtractor<T> conversion_op(begin_bit, end_bit);
        cub::TransformInputIterator<T, RadixExtractor<T>, const T*> itr(partitioned_key_col.data<T>(), conversion_op);
        
        CALL_CUB_FUNC(cub::DeviceHistogram::HistogramEven, itr, d_counts_out_ptr->data<gamax::size_type>(), n_partitions+1, 0, n_partitions, num_items);
        CALL_CUB_FUNC(cub::DeviceScan::InclusiveSum, d_counts_out_ptr->data<gamax::size_type>(), offsets->data<gamax::size_type>(), n_partitions);

        return offsets;
    }
};
    
} // unnamed namespace

std::unique_ptr<gamax::column>
compute_partition_offsets(
    gamax::column& partitioned_key_col,
    int begin_bit,
    int end_bit,
    cudaStream_t stream)
{
    return compute_partition_offsets(partitioned_key_col.view(), begin_bit, end_bit, stream);
}

std::unique_ptr<gamax::column>
compute_partition_offsets(
    gamax::column_view partitioned_key_col,
    int begin_bit,
    int end_bit,
    cudaStream_t stream)
{
    return gamax::type_dispatcher(partitioned_key_col.type(), compute_partition_offsets_impl{}, partitioned_key_col, begin_bit, end_bit, stream);
}
} // namespace gamax::partition