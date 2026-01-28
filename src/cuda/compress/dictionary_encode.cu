#include "compress/dictionary_encode.hpp"

#include <memory>
#include <stdexcept>
#include <cmath>
#include <cuco/static_set.cuh>
#include <thrust/functional.h>
#include <cuda.h>

#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "types/sentinel.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "utils/utils.cuh"
#include "utils/transform.hpp"
#include "operators/sort/sort.hpp"
#include "atomics/atomics.cuh"

namespace gamax::compress {
namespace {
    struct is_integral_check {
        template<typename T>
        bool operator()() const {
            return std::is_integral_v<T>;
        }
    };

    void check_dictionary_encode_solvable(const gamax::column_view column, const gamax::data_type encode_type) {
        // Skip strings for now
        if(column.type().id() == gamax::type_id::STRING) {
            throw std::runtime_error("Cannot dictionary encode column of type " + gamax::to_string(column.type()));
        }
        // Encoded type must be integral
        if(!type_dispatcher(encode_type, is_integral_check{})) {
            throw std::runtime_error("Encoded type must be integral, got " + gamax::to_string(encode_type));
        }
    }

    void check_dictionary_encode_solvable(const gamax::table_view table, const gamax::data_type encode_type) {
        throw std::runtime_error("Cannot dictionary encode table");
    }

    // Equality functor that compares row indices by their underlying key values
    template<typename T>
    struct key_equal {
        const T* data;

        __host__ __device__ key_equal(const T* data_) : data(data_) {}

        __device__ bool operator()(gamax::size_type idx_a, gamax::size_type idx_b) const {
            return data[idx_a] == data[idx_b];
        }
    };

    // Hash functor that hashes the key value at the given row index
    template<typename T>
    struct key_hasher {
        const T* data;
        cuco::default_hash_function<T> hash_fn{};

        __host__ __device__ key_hasher(const T* data_) : data(data_) {}

        __device__ auto operator()(gamax::size_type idx) const {
            return hash_fn(data[idx]);
        }
    };

    template<typename Set, typename EncodedT>
    __global__ void dictionary_encode_set_kernel(
        Set set_ref, 
        const gamax::size_type num_items, 
        EncodedT* __restrict__ sparse_encoded_ids,
        gamax::size_type* __restrict__ group_ids_loc,
        gamax::size_type* __restrict__ dict_size) 
    {
        int64_t const loop_stride = gridDim.x * blockDim.x;
        int64_t idx               = blockDim.x * blockIdx.x + threadIdx.x;

        while (idx < num_items) {
            auto [iter, is_new_key] = set_ref.insert_and_find(idx);
            
            if (is_new_key) {
                // This is a new unique key - assign a new encoded ID
                EncodedT encoded_value = static_cast<EncodedT>(
                    gamax::atomic::atomic_add(dict_size, static_cast<gamax::size_type>(1)));
                sparse_encoded_ids[idx] = encoded_value;
            }
            
            // Store the representative index for this row
            group_ids_loc[idx] = *iter;
            
            idx += loop_stride;
        }
    }

    // Kernel for order-preserving encoding: only builds hash set and group_ids_loc
    template<typename Set>
    __global__ void dictionary_encode_set_no_encode_kernel(
        Set set_ref, 
        const gamax::size_type num_items, 
        gamax::size_type* __restrict__ group_ids_loc,
        gamax::size_type* __restrict__ dict_size) 
    {
        int64_t const loop_stride = gridDim.x * blockDim.x;
        int64_t idx               = blockDim.x * blockIdx.x + threadIdx.x;

        while (idx < num_items) {
            auto [iter, is_new_key] = set_ref.insert_and_find(idx);
            
            if (is_new_key) {
                gamax::atomic::atomic_add(dict_size, static_cast<gamax::size_type>(1));
            }
            
            // Store the representative index for this row
            group_ids_loc[idx] = *iter;
            
            idx += loop_stride;
        }
    }

    // Kernel to scatter ranks into sparse_encoded_ids based on sorted order
    // For each rank i: sparse_encoded_ids[retrieved_row_ids[argsort_indices[i]]] = i
    template<typename EncodedT>
    __global__ void scatter_ranks_kernel(
        const gamax::size_type dict_size,
        const gamax::size_type* __restrict__ retrieved_row_ids,
        const gamax::size_type* __restrict__ argsort_indices,
        EncodedT* __restrict__ sparse_encoded_ids)
    {
        int64_t const loop_stride = gridDim.x * blockDim.x;
        int64_t rank              = blockDim.x * blockIdx.x + threadIdx.x;

        while (rank < dict_size) {
            gamax::size_type sorted_pos = argsort_indices[rank];
            gamax::size_type rep_row_id = retrieved_row_ids[sorted_pos];
            sparse_encoded_ids[rep_row_id] = static_cast<EncodedT>(rank);
            
            rank += loop_stride;
        }
    }

    struct dictionary_encode_impl {
        // Non-integral encoded type - not supported
        template<typename T, typename EncodedT, 
                 std::enable_if_t<!std::is_integral_v<EncodedT>, int> = 0>
        std::pair<std::unique_ptr<gamax::column>, gamax::size_type> operator()(
            const gamax::column_view column, 
            const gamax::size_type dictionary_size_hint,
            const double load_factor,
            cudaStream_t stream) 
        {
            throw std::runtime_error("Encoded type must be integral.");
        }

        // Main implementation - works with any input type T, as long as EncodedT is integral
        template<typename T, typename EncodedT,
                 std::enable_if_t<std::is_integral_v<EncodedT>, int> = 0>
        std::pair<std::unique_ptr<gamax::column>, gamax::size_type> operator()(
            const gamax::column_view column, 
            const gamax::size_type dictionary_size_hint,
            const double load_factor,
            cudaStream_t stream) 
        {
            auto num_items = column.size();
            auto estimated_dict_size = dictionary_size_hint == 0 ? num_items : dictionary_size_hint;
            auto set_capacity = static_cast<gamax::size_type>(std::ceil(estimated_dict_size / load_factor));
            
            // Use row index sentinel (always negative for signed size_type, so safe)
            gamax::size_type constexpr empty_key_sentinel = gamax::sentinel<gamax::size_type>();
            
            // Get allocator for cuco
            auto cuco_allocator = *get_thrust_allocator<gamax::size_type>(stream);
            
            // Create custom equality and hash functors
            auto data_ptr = column.data<T>();
            auto equality = key_equal<T>{data_ptr};
            auto hasher = key_hasher<T>{data_ptr};
            
            // Create the static_set with row indices
            auto set = cuco::static_set{
                set_capacity,
                cuco::empty_key{empty_key_sentinel},
                equality,
                cuco::linear_probing<1, key_hasher<T>>{hasher},
                cuco::thread_scope_device,
                cuco::storage<1>{},
                cuco_allocator,
                stream
            };
            auto set_ref = set.ref(cuco::insert_and_find);
            
            // Sparse encoded IDs array - stores encoded value at representative index
            auto sparse_encoded_ids = gamax::make_column<EncodedT>(num_items, false);
            
            // Group IDs location - stores the representative index for each row
            auto group_ids_loc = gamax::make_idx_column(num_items, false);
            
            // Counter for dictionary size
            auto d_dict_size = gamax::make_idx_column(1, true);

            // First kernel: build the hash set and assign encoded IDs to representatives
            dictionary_encode_set_kernel<<<num_tb(num_items, 256), 256, 0, stream>>>(
                set_ref, 
                num_items, 
                sparse_encoded_ids->template data<EncodedT>(),
                group_ids_loc->template data<gamax::size_type>(),
                d_dict_size->template data<gamax::size_type>());
            
            // Gather: finalize encoding by looking up representative's encoded value
            // encoded_col[i] = sparse_encoded_ids[group_ids_loc[i]]
            auto encoded_col = gamax::util::device::gather(
                group_ids_loc->view(),
                sparse_encoded_ids->view(),
                stream);
            
            gamax::size_type dict_size;
            cudaMemcpyAsync(&dict_size, d_dict_size->template data<gamax::size_type>(), 
                           sizeof(gamax::size_type), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            return std::make_pair(std::move(encoded_col), dict_size);
        }
    };

    // Order-preserving dictionary encoding implementation
    struct dictionary_encode_with_order_impl {
        // Non-integral encoded type - not supported
        template<typename T, typename EncodedT, 
                 std::enable_if_t<!std::is_integral_v<EncodedT>, int> = 0>
        std::pair<std::unique_ptr<gamax::column>, gamax::size_type> operator()(
            const gamax::column_view column, 
            const gamax::size_type dictionary_size_hint,
            const double load_factor,
            cudaStream_t stream) 
        {
            throw std::runtime_error("Encoded type must be integral.");
        }

        // Main implementation - order-preserving encoding
        template<typename T, typename EncodedT,
                 std::enable_if_t<std::is_integral_v<EncodedT>, int> = 0>
        std::pair<std::unique_ptr<gamax::column>, gamax::size_type> operator()(
            const gamax::column_view column, 
            const gamax::size_type dictionary_size_hint,
            const double load_factor,
            cudaStream_t stream) 
        {
            auto num_items = column.size();
            auto estimated_dict_size = dictionary_size_hint == 0 ? num_items : dictionary_size_hint;
            auto set_capacity = static_cast<gamax::size_type>(std::ceil(estimated_dict_size / load_factor));
            
            // Use row index sentinel (always negative for signed size_type, so safe)
            gamax::size_type constexpr empty_key_sentinel = gamax::sentinel<gamax::size_type>();
            
            // Get allocator for cuco
            auto cuco_allocator = *get_thrust_allocator<gamax::size_type>(stream);
            
            // Create custom equality and hash functors
            auto data_ptr = column.data<T>();
            auto equality = key_equal<T>{data_ptr};
            auto hasher = key_hasher<T>{data_ptr};
            
            // Create the static_set with row indices
            auto set = cuco::static_set{
                set_capacity,
                cuco::empty_key{empty_key_sentinel},
                equality,
                cuco::linear_probing<1, key_hasher<T>>{hasher},
                cuco::thread_scope_device,
                cuco::storage<1>{},
                cuco_allocator,
                stream
            };
            auto set_ref = set.ref(cuco::insert_and_find);
            
            // Group IDs location - stores the representative index for each row
            auto group_ids_loc = gamax::make_idx_column(num_items, false);
            
            // Counter for dictionary size
            auto d_dict_size = gamax::make_idx_column(1, true);

            // Step 1: Build the hash set and populate group_ids_loc (no encoding yet)
            dictionary_encode_set_no_encode_kernel<<<num_tb(num_items, 256), 256, 0, stream>>>(
                set_ref, 
                num_items, 
                group_ids_loc->template data<gamax::size_type>(),
                d_dict_size->template data<gamax::size_type>());
            
            // Get dictionary size
            gamax::size_type dict_size;
            cudaMemcpyAsync(&dict_size, d_dict_size->template data<gamax::size_type>(), 
                           sizeof(gamax::size_type), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // Step 2: Retrieve all unique representative row IDs from the set
            auto retrieved_row_ids = gamax::make_idx_column(dict_size, false);
            set.retrieve_all(retrieved_row_ids->template data<gamax::size_type>(), 
                            cuda::stream_ref{stream});
            
            // Step 3: Gather the input values at representative row IDs
            auto unique_values = gamax::util::device::gather(
                retrieved_row_ids->view(),
                column,
                stream);
            
            // Step 4: Argsort the unique values to get sorted indices
            auto [sorted_values, argsort_indices] = gamax::sort::argsort(
                unique_values->view(),
                0, -1,
                gamax::sort_order::ASCENDING,
                nullptr, 0,
                stream);
            
            // Step 5: Scatter ranks into sparse_encoded_ids based on sorted order
            auto sparse_encoded_ids = gamax::make_column<EncodedT>(num_items, false);
            scatter_ranks_kernel<<<num_tb(dict_size, 256), 256, 0, stream>>>(
                dict_size,
                retrieved_row_ids->template data<gamax::size_type>(),
                argsort_indices->template data<gamax::size_type>(),
                sparse_encoded_ids->template data<EncodedT>());
            
            // Step 6: Gather final encoded column
            auto encoded_col = gamax::util::device::gather(
                group_ids_loc->view(),
                sparse_encoded_ids->view(),
                stream);
            
            cudaStreamSynchronize(stream);

            return std::make_pair(std::move(encoded_col), dict_size);
        }
    };
}

std::pair<std::unique_ptr<gamax::column>, gamax::size_type> dictionary_encode(
    const gamax::column_view column, 
    const data_type encode_type,
    const gamax::size_type dictionary_size_hint,
    const double load_factor,
    cudaStream_t stream)
{
    check_dictionary_encode_solvable(column, encode_type);

    return double_type_dispatcher(
        column.type(),
        encode_type,
        dictionary_encode_impl{}, 
        column, 
        dictionary_size_hint, 
        load_factor, 
        stream);
}

std::pair<std::unique_ptr<gamax::column>, gamax::size_type> dictionary_encode_with_order(
    const gamax::column_view column, 
    const data_type encode_type,
    const gamax::size_type dictionary_size_hint,
    const double load_factor,
    cudaStream_t stream)
{
    check_dictionary_encode_solvable(column, encode_type);

    return double_type_dispatcher(
        column.type(),
        encode_type,
        dictionary_encode_with_order_impl{}, 
        column, 
        dictionary_size_hint, 
        load_factor, 
        stream);
}

std::pair<std::unique_ptr<gamax::column>, gamax::size_type> dictionary_encode(
    const gamax::table_view table, 
    const data_type encode_type,
    const gamax::size_type dictionary_size_hint,
    const double load_factor,
    cudaStream_t stream) 
{
    check_dictionary_encode_solvable(table, encode_type);
    return std::make_pair(nullptr, 0);
}
}