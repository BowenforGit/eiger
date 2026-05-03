#include "inner_join.hpp"

#include <memory>
#include <cuda.h>

#include "join.hpp"
#include "inner_join.hpp"
#include "materialize.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "operators/partition/partition.hpp"
#include "thrust/iterator/counting_iterator.h"
#include "utils/utils.cuh"
#include "atomics/atomics.cuh"
#include "utils/device_info.hpp"
#include "utils/warp_utils.cuh"

namespace gamax::join{
namespace detail {
namespace {
    using triple_col_t = std::tuple<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>>;
    using tid_type = gamax::size_type;

    struct partition_hash_join_params {
        explicit partition_hash_join_params(
            gamax::table_view const& left, 
            gamax::table_view const& right, 
            gamax::join::join_hint const& hint)
        
        {
            // TODO: set these values based on the workloads
            begin_bit = 0;
            end_bit = 16;
            // STRING payloads can't go through the narrow / early-mat
            // path: it calls partition_column_pair on (key, payload),
            // and STRING values can't be radix-partitioned alongside
            // the key. Defer those to the wide GFTR / GFUR paths (or
            // the GFTR-string wrapper one layer up).
            early_materialization = (left.num_columns() == 2 && right.num_columns() == 2
                                     && left[1].type() == right[1].type()
                                     && left[1].type().id() != gamax::type_id::STRING);
            n_partitions = 1 << (end_bit - begin_bit);
            max_build_bucket_size = 1 << 12;
            max_probe_bucket_size = 1 << 13;
            hash_table_size = 1 << 11;
            const int device_warp_size = gamax::warp_size();
            shuffle_space_size = device_warp_size;
            max_output_size = gamax::join::detail::get_ieq_join_output_alloc_size(left, right, hint);

            join_kernel_tb_size = 512;

            auto compute_smem_bytes = [&](int tb_size) {
                const auto warp_chunks = tb_size / device_warp_size;
                if(early_materialization) {
                    return (max_build_bucket_size + 512) * (gamax::size_of(left[0].type()) + gamax::size_of(left[1].type()) + sizeof(int16_t))
                        + hash_table_size * sizeof(int32_t)
                        + shuffle_space_size * warp_chunks * (gamax::size_of(left[0].type()) + gamax::size_of(left[1].type()) + gamax::size_of(right[1].type()));
                } else {
                    return (max_build_bucket_size + 512) * (gamax::size_of(left[0].type()) + sizeof(gamax::size_type) + sizeof(int16_t))
                        + hash_table_size * sizeof(int32_t)
                        + shuffle_space_size * warp_chunks * (gamax::size_of(left[0].type()) + sizeof(gamax::size_type) * 2);
                }
            };

            // HIP LDS is 64 KiB; shrink the block size until shared memory fits.
            const std::size_t max_smem = static_cast<std::size_t>(gamax::MAX_SMEM_BYTES);
            join_kernel_smem_bytes = compute_smem_bytes(join_kernel_tb_size);
            while(join_kernel_smem_bytes > max_smem && join_kernel_tb_size > device_warp_size) {
                join_kernel_tb_size /= 2;
                join_kernel_smem_bytes = compute_smem_bytes(join_kernel_tb_size);
            }
        }
        
        int begin_bit;
        int end_bit;
        bool early_materialization;
        int join_kernel_tb_size;
        std::size_t join_kernel_smem_bytes;
        gamax::size_type n_partitions;
        gamax::size_type max_build_bucket_size;
        gamax::size_type max_probe_bucket_size;
        gamax::size_type hash_table_size;
        gamax::size_type shuffle_space_size;
        gamax::size_type max_output_size;
    };

    template<typename KeyT, typename ValT>
    struct shuffle_space {
        ValT* val_S_elem {nullptr};
        ValT* val_R_elem {nullptr};
        KeyT* key_elem {nullptr};

        __device__ shuffle_space create_warp_shuffle_space(int wid, int shuffle_space_size) {
            shuffle_space warp_shuffle;
            int offset = shuffle_space_size * wid;
            warp_shuffle.val_S_elem = this->val_S_elem + offset;
            warp_shuffle.val_R_elem = this->val_R_elem + offset;
            warp_shuffle.key_elem = this->key_elem + offset;
            return warp_shuffle;
        }
    };

    // partitioned keys, partitioned values, partition offsets
    triple_col_t partition_transform(
        gamax::column_view key,
        gamax::column_view val,
        int begin_bit,
        int end_bit,
        cudaStream_t stream)
    {
        auto partitioned = gamax::partition::partition_column_pair(key, val, begin_bit, end_bit, nullptr, 0, stream);
        auto offset = gamax::partition::compute_partition_offsets(partitioned.first->view(), begin_bit, end_bit, stream);
        return std::make_tuple(std::move(partitioned.first), std::move(partitioned.second), std::move(offset));
    } 

    __global__ void generate_work_units(
        const gamax::size_type* __restrict__  r_offsets,
        const gamax::size_type* __restrict__  s_offsets,
        uint64_t*  __restrict__  r_work,
        uint64_t*  __restrict__  s_work,
        gamax::size_type*        total_work, // make sure this is initialized to 0
        gamax::size_type*        max_r_len_out, // maximum build-side bucket size observed
        const gamax::size_type   n_partitions,
        const gamax::size_type   threshold)
    {
        for(int p = get_cuda_tid(); p < n_partitions; p += nthreads()) {
            uint64_t r_start = (p >= 1 ? r_offsets[p-1] : 0);
            uint64_t r_end = r_offsets[p];
            uint64_t s_start = (p >= 1 ? s_offsets[p-1] : 0);
            uint64_t s_end = s_offsets[p];
            auto r_len = r_end - r_start;
            auto s_len = s_end - s_start;

            // Track the largest build-side bucket so the host can
            // pre-flight against `max_build_bucket_size` — the kernel
            // writes past `elem`/`payload`/`next` shared-memory
            // arrays if any partition exceeds that cap.
            gamax::atomic::atomic_max(
                max_r_len_out, static_cast<gamax::size_type>(r_len));

            auto n_units = (s_len + threshold - 1)/threshold;
            auto pos = gamax::atomic::atomic_add(total_work, static_cast<gamax::size_type>(n_units));

            #pragma unroll
            for(int i = 0; i < n_units; i++, pos++, s_len -= threshold, s_start += threshold) {
                r_work[pos] = (r_start << 32) + r_len;
                s_work[pos] = (s_start << 32) + min(s_len, (uint64_t)threshold);
            }
        }
    }

    // left work, right work, total work units. Also precondition-
    // checks the build-side: if any partition has more rows than
    // max_build_bucket_size, the kernel would write past the
    // `elem`/`payload`/`next` shared-memory arrays — throw before
    // launch with a clear error.
    triple_col_t create_work_units(
        gamax::column_view const& left_offs,
        gamax::column_view const& right_offs,
        partition_hash_join_params const& params,
        cudaStream_t stream)
    {
        auto n_partitions = params.n_partitions;

        auto left_work = gamax::make_column<uint64_t>(n_partitions*2);
        auto right_work = gamax::make_column<uint64_t>(n_partitions*2);
        auto total_work = gamax::make_column<gamax::size_type>(1, true);
        auto max_r_len  = gamax::make_column<gamax::size_type>(1, true);

        generate_work_units<<<num_tb(n_partitions,512),512, 0, stream>>>(
            left_offs.data<gamax::size_type>(),
            right_offs.data<gamax::size_type>(),
            left_work->data<uint64_t>(),
            right_work->data<uint64_t>(),
            total_work->data<gamax::size_type>(),
            max_r_len->data<gamax::size_type>(),
            n_partitions,
            params.max_probe_bucket_size);
        CHECK_LAST_CUDA_ERROR();
        cudaStreamSynchronize(stream);

        gamax::size_type host_max_r_len = 0;
        cudaMemcpyAsync(&host_max_r_len, max_r_len->data<gamax::size_type>(),
                        sizeof(gamax::size_type),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (host_max_r_len > params.max_build_bucket_size) {
            throw std::runtime_error(
                "PARTITION hash join: build-side bucket overflow "
                "(largest partition has " +
                std::to_string(host_max_r_len) +
                " rows, exceeds max_build_bucket_size=" +
                std::to_string(params.max_build_bucket_size) +
                "); use HASH or SORT for skewed key distributions");
        }

        return std::make_tuple(std::move(left_work), std::move(right_work), std::move(total_work));
    }

    template<typename KeyT, typename ValT, typename ValIt = thrust::counting_iterator<ValT>>
    __global__ void join_copartitions_arr(
        const KeyT* R, 
        ValIt Pr, 
        const KeyT* S, 
        ValIt Ps, 
        const uint64_t*  R_offsets, // offsets of partitions in R (inclusive prefix sum)
        const uint64_t*  S_offsets, // offsets of partitions in S (inclusive prefix sum)
        const gamax::size_type*  n_work_units, // number of work units 
        partition_hash_join_params params,
        gamax::size_type*        results,
        KeyT*       keys_out, 
        ValT*       r_output, 
        ValT*       s_output, 
        const int   circular_buffer_size) 
    {
        constexpr int VT = 4;

        const int hash_table_size = params.hash_table_size;
        const int log_parts = params.end_bit - params.begin_bit; // number of partitions in log2
        const auto max_build_bucket_size = params.max_build_bucket_size;
        const int shuffle_space_size = params.shuffle_space_size;
        const int tid = threadIdx.x;
        const int block = blockIdx.x;
        const int pwidth = gridDim.x;
        const auto buckets_cnt = *n_work_units;
        constexpr int warp_size = gamax::warp_size();
        const int lid = tid % warp_size;
        const int wid = tid / warp_size;
        const int nwarps = blockDim.x / warp_size;
        using ballot_t = decltype(__activemask());
        const ballot_t threadmask = gamax::warp::upper_lane_mask<ballot_t>(lid, warp_size);

        extern __shared__ int16_t temp[];

        KeyT* elem = (KeyT*)temp;
        ValT* payload = (ValT*)&elem[max_build_bucket_size+512];
        int16_t* next = (int16_t*)&payload[max_build_bucket_size+512];
        int32_t* head = (int32_t*)&next[max_build_bucket_size+512];
        
        shuffle_space<KeyT, ValT> block_shuffle;
        block_shuffle.val_S_elem = (ValT*)&head[hash_table_size];
        block_shuffle.val_R_elem = (ValT*)&block_shuffle.val_S_elem[shuffle_space_size*nwarps];
        block_shuffle.key_elem = (KeyT*)&block_shuffle.val_R_elem[shuffle_space_size*nwarps];
        
        auto warp_shuffle = block_shuffle.create_warp_shuffle_space(wid, shuffle_space_size);

        int ptr, shuffle_ptr = 0;

        // one thread block per partition
        // Always let R be the build table, S be the probe table
        for(uint32_t bucket_r = block; bucket_r < buckets_cnt; bucket_r += pwidth) {
            auto start = (R_offsets[bucket_r] >> 32);
            auto len_R = R_offsets[bucket_r] & 0xFFFFFFFF;

            for (int i = tid; i < hash_table_size; i += blockDim.x)
                head[i] = -1;
            __syncthreads();

            for(int base_r = 0; base_r < len_R; base_r += VT*blockDim.x) {
                KeyT data_R[VT];
                ValT data_Pr[VT];
                
                #pragma unroll
                for(int i = 0; i < VT; i++) {
                    data_R[i] = R[start + base_r + VT*threadIdx.x + i];
                    data_Pr[i] = Pr[start + base_r + VT*threadIdx.x + i];
                }
                
                int l_cnt_R = len_R - base_r - VT * threadIdx.x;                  

                #pragma unroll
                for (int k = 0; k < VT; k++) {
                    if (k < l_cnt_R) {
                        auto val = data_R[k];
                        elem[base_r + k*blockDim.x + tid] = val;
                        payload[base_r + k*blockDim.x + tid] = data_Pr[k];
                        int hval = (val >> log_parts) & (hash_table_size - 1);

                        int32_t last = atomicExch(&head[hval], base_r + k*blockDim.x + tid);
                        next[base_r + k*blockDim.x + tid] = last;
                    }
                }
            }
            __syncthreads();

            // probe
            start = (S_offsets[bucket_r] >> 32);
            auto len_S = S_offsets[bucket_r] & 0xFFFFFFFF;
            for(int offset_s = 0; offset_s < len_S; offset_s += VT*blockDim.x) {
                KeyT data_S[VT];
                ValT data_Ps[VT];
                
                #pragma unroll
                for(int i = 0; i < VT; i++) {
                    data_S[i] = S[start + offset_s + VT*threadIdx.x + i];
                    data_Ps[i] = Ps[start + offset_s + VT*threadIdx.x + i];
                }
                
                int l_cnt_S = len_S - offset_s - VT * threadIdx.x;

                #pragma unroll
                for (int k = 0; k < VT; k++) {
                    auto val = data_S[k];
                    auto pval = data_Ps[k];
                    int32_t hval =  (val >> log_parts) & (hash_table_size - 1);
                    ValT pay;

                    int32_t pos = (k < l_cnt_S)? head[hval] : -1;

                    /*check at warp level whether someone is still following chain => this way we can shuffle without risk*/
                    int pred = (pos >= 0);

                    while (__any_sync(__activemask(), pred)) {
                        int wr_intention = 0;

                        /*we have a match, fetch the data to be written*/
                        if (pred) {
                            if(elem[pos] == val) {
                                pay = payload[pos];
                                wr_intention = 1;
                            }

                            pos = next[pos];
                            pred = (pos >= 0);
                        }

                        /*find out who had a match in this execution step*/
                        ballot_t mask = __ballot_sync(__activemask(), wr_intention);

                        /*our software managed buffer will overflow, flush it*/
                        int wr_offset = shuffle_ptr +  gamax::warp::popc_mask(mask & threadmask);
                        shuffle_ptr = shuffle_ptr + gamax::warp::popc_mask(mask);
                        
                        /*while it overflows, flush
                        we flush 16 keys and then the 16 corresponding payloads consecutively, of course other formats might be friendlier*/
                        while (shuffle_ptr >= shuffle_space_size) {
                            if (wr_intention && (wr_offset < shuffle_space_size)) {
                                warp_shuffle.val_R_elem[wr_offset] = pay;
                                warp_shuffle.val_S_elem[wr_offset] = pval;
                                warp_shuffle.key_elem[wr_offset] = val;
                                wr_intention = 0;
                            }

                            if (lid == 0) {
                                ptr = gamax::atomic::atomic_add(results, static_cast<gamax::size_type>(shuffle_space_size));
                            }

                            ptr = __shfl_sync(__activemask(), ptr, 0);

                            auto w_pos = (ptr + lid) % circular_buffer_size;

                            if(lid < shuffle_space_size) {
                                r_output[w_pos] = warp_shuffle.val_R_elem[lid];
                                s_output[w_pos] = warp_shuffle.val_S_elem[lid];
                                keys_out[w_pos] = warp_shuffle.key_elem[lid];
                            }

                            wr_offset -= shuffle_space_size;
                            shuffle_ptr -= shuffle_space_size;
                        }

                        /*now the fit, write them in buffer*/
                        if (wr_intention && (wr_offset >= 0)) {
                            warp_shuffle.val_R_elem[wr_offset] = pay; // R
                            warp_shuffle.val_S_elem[wr_offset] = pval; // S
                            warp_shuffle.key_elem[wr_offset] = val; // key
                            wr_intention = 0;
                        }
                    }                   
                }
            }

            if(bucket_r + pwidth >= buckets_cnt) break;
            __syncthreads();
        }

        if (lid == 0) {
            ptr = gamax::atomic::atomic_add(results, static_cast<gamax::size_type>(shuffle_ptr));
        }

        ptr = __shfl_sync(__activemask(), ptr, 0);

        if (lid < shuffle_ptr) {
            auto w_pos = (ptr + lid) % circular_buffer_size;
            if(lid < shuffle_space_size) {
                r_output[w_pos] = warp_shuffle.val_R_elem[lid];
                s_output[w_pos] = warp_shuffle.val_S_elem[lid];
                keys_out[w_pos] = warp_shuffle.key_elem[lid];
            }
        }
    }

    struct join_copartitions_impl {
        template<typename KeyT, typename ValT,
                 typename std::enable_if_t<!std::is_integral<KeyT>::value, int> = 0>
        void operator()(
            gamax::column_view left_key,
            gamax::column_view left_val,
            gamax::column_view left_offs,
            gamax::column_view right_key,
            gamax::column_view right_val,
            gamax::column_view right_offs,
            partition_hash_join_params const& params,
            gamax::mutable_column_view res_key,
            gamax::mutable_column_view res_left,
            gamax::mutable_column_view res_right,
            gamax::size_type& n_matches,
            cudaStream_t stream)
        {
            throw std::runtime_error("join_copartitions_impl does not support non-integral key types");
        }

        template<typename KeyT, typename ValT,
                 typename std::enable_if_t<std::is_integral<KeyT>::value, int> = 0>
        void operator()(
            gamax::column_view left_key,
            gamax::column_view left_val,
            gamax::column_view left_offs,
            gamax::column_view right_key,
            gamax::column_view right_val,
            gamax::column_view right_offs,
            partition_hash_join_params const& params,
            gamax::mutable_column_view res_key,
            gamax::mutable_column_view res_left,
            gamax::mutable_column_view res_right,
            gamax::size_type& n_matches,
            cudaStream_t stream)
        {
            auto [left_work, right_work, total_work] = create_work_units(left_offs, right_offs, params, stream);

            auto d_n_matches = gamax::make_column<gamax::size_type>(1, true);

            auto join_fn = join_copartitions_arr<KeyT, ValT, const ValT*>;
            CHECK_CUDA_ERROR(cudaFuncSetAttribute(join_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, params.join_kernel_smem_bytes));
            

            join_fn<<<params.n_partitions, params.join_kernel_tb_size, params.join_kernel_smem_bytes, stream>>>(
                left_key.data<KeyT>(), left_val.data<ValT>(), 
                right_key.data<KeyT>(), right_val.data<ValT>(), 
                left_work->data<uint64_t>(), right_work->data<uint64_t>(), 
                total_work->data<gamax::size_type>(), 
                params, 
                d_n_matches->data<gamax::size_type>(), 
                res_key.data<KeyT>(), res_left.data<ValT>(), res_right.data<ValT>(), 
                res_key.size());
            CHECK_LAST_CUDA_ERROR();
            cudaMemcpyAsync(&n_matches, d_n_matches->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        template<typename KeyT,
                 typename std::enable_if_t<!std::is_integral<KeyT>::value, int> = 0>
        void operator()(
            gamax::column_view left_key,
            gamax::column_view left_offs,
            gamax::column_view right_key,
            gamax::column_view right_offs,
            partition_hash_join_params const& params,
            gamax::mutable_column_view res_key,
            gamax::mutable_column_view res_left,
            gamax::mutable_column_view res_right,
            gamax::size_type& n_matches,
            cudaStream_t stream)
        {
            throw std::runtime_error("join_copartitions_impl does not support non-integral key types");
        }

        template<typename KeyT,
                 typename std::enable_if_t<std::is_integral<KeyT>::value, int> = 0>
        void operator()(
            gamax::column_view left_key,
            gamax::column_view left_offs,
            gamax::column_view right_key,
            gamax::column_view right_offs,
            partition_hash_join_params const& params,
            gamax::mutable_column_view res_key,
            gamax::mutable_column_view res_left,
            gamax::mutable_column_view res_right,
            gamax::size_type& n_matches,
            cudaStream_t stream)
        {
            auto [left_work, right_work, total_work] = create_work_units(left_offs, right_offs, params, stream);
            auto d_n_matches = gamax::make_column<gamax::size_type>(1, true);

            thrust::counting_iterator<tid_type> left_itr(0);
            thrust::counting_iterator<tid_type> right_itr(0);

            auto join_fn = join_copartitions_arr<KeyT, tid_type>;
            CHECK_CUDA_ERROR(cudaFuncSetAttribute(join_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, params.join_kernel_smem_bytes));
            
            join_fn<<<params.n_partitions, params.join_kernel_tb_size, params.join_kernel_smem_bytes, stream>>>(
                left_key.data<KeyT>(), left_itr, 
                right_key.data<KeyT>(), right_itr, 
                left_work->data<uint64_t>(), right_work->data<uint64_t>(), 
                total_work->data<gamax::size_type>(), 
                params, 
                d_n_matches->data<gamax::size_type>(), 
                res_key.data<KeyT>(), res_left.data<tid_type>(), res_right.data<tid_type>(), 
                res_key.size());
            CHECK_LAST_CUDA_ERROR();
            cudaMemcpyAsync(&n_matches, d_n_matches->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    };

    triple_col_t join_copartitions(
        gamax::column_view const& left_key,
        gamax::column_view const& left_val,
        gamax::column_view const& left_offs,
        gamax::column_view const& right_key,
        gamax::column_view const& right_val,
        gamax::column_view const& right_offs,
        partition_hash_join_params const& params,
        cudaStream_t stream)
    {
        auto key_type = left_key.type();
        auto val_type = left_val.type();

        auto res_key = gamax::make_column(key_type, params.max_output_size);
        auto res_left = gamax::make_column(val_type, params.max_output_size);
        auto res_right = gamax::make_column(val_type, params.max_output_size);

        gamax::size_type n_matches = 0;

        gamax::double_type_dispatcher(
            key_type, val_type, join_copartitions_impl{}, 
            left_key, left_val, left_offs, 
            right_key, right_val, right_offs, 
            params, 
            res_key->mutable_view(), 
            res_left->mutable_view(), res_right->mutable_view(), 
            n_matches, 
            stream
        );
        
        res_key->set_size(n_matches);
        res_left->set_size(n_matches);
        res_right->set_size(n_matches);

        return std::make_tuple(std::move(res_key), std::move(res_left), std::move(res_right));
    }

    triple_col_t join_copartitions(
        gamax::column_view const& left_key,
        gamax::column_view const& left_offs,
        gamax::column_view const& right_key,
        gamax::column_view const& right_offs,
        partition_hash_join_params const& params,
        cudaStream_t stream)
    {
        auto key_type = left_key.type();

        auto res_key = gamax::make_column(key_type, params.max_output_size);
        auto res_left = gamax::make_idx_column(params.max_output_size);
        auto res_right = gamax::make_idx_column(params.max_output_size);

        gamax::size_type n_matches = 0;

        gamax::type_dispatcher(
            key_type, join_copartitions_impl{}, 
            left_key, left_offs, 
            right_key, right_offs, 
            params, 
            res_key->mutable_view(), 
            res_left->mutable_view(), res_right->mutable_view(), 
            n_matches, 
            stream
        );
        
        res_key->set_size(n_matches);
        res_left->set_size(n_matches);
        res_right->set_size(n_matches);

        return std::make_tuple(std::move(res_key), std::move(res_left), std::move(res_right));
    }

    std::unique_ptr<gamax::table> ieq_partition_hash_join_narrow(
        gamax::table_view left, 
        gamax::table_view right,
        partition_hash_join_params const& params,
        cudaStream_t stream)
    {   
        // transformation
        auto [par_left_key, par_left_val, par_left_offs] = partition_transform(left[0], left[1], params.begin_bit, params.end_bit, stream);
        auto [par_right_key, par_right_val, par_right_offs] = partition_transform(right[0], right[1], params.begin_bit, params.end_bit, stream);
        
        // join co-partitions
        auto [res_key, res_left, res_right] = detail::join_copartitions(
            par_left_key->view(), par_left_val->view(), par_left_offs->view(),
            par_right_key->view(), par_right_val->view(), par_right_offs->view(),
            params, stream);
        
        // materialize
        auto res_table = std::make_unique<gamax::table>();
        res_table->add_column(std::move(res_key));
        res_table->add_column(std::move(res_left));
        res_table->add_column(std::move(res_right));
        
        return res_table;
    }
} // namespace

    std::unique_ptr<gamax::table> ieq_partition_hash_join_gftr(
        gamax::table_view left,
        gamax::table_view right,
        gamax::join::join_hint const& hint,
        cudaStream_t stream)
    {
        gamax::join::detail::partition_hash_join_params params(left, right, hint);

        if(params.early_materialization) {
            return ieq_partition_hash_join_narrow(left, right, params, stream);
        }

        // transformation
        auto [par_left_key, par_left_val, par_left_offs] = partition_transform(left[0], left[1], params.begin_bit, params.end_bit, stream);
        auto [par_right_key, par_right_val, par_right_offs] = partition_transform(right[0], right[1], params.begin_bit, params.end_bit, stream);

        // join co-partitions
        auto [res_key, left_tid, right_tid] = detail::join_copartitions(
            par_left_key->view(), par_left_offs->view(),
            par_right_key->view(), par_right_offs->view(),
            params, stream);
        
        // materialize
        auto res_table = std::make_unique<gamax::table>();
        res_table->add_column(std::move(res_key));
        res_table->add_column(simple_materialize(par_left_val->view(), left_tid->view(), stream));
        par_left_val.reset(nullptr);
        for(auto i = 2; i < left.num_columns(); i++) {
            std::tie(par_left_key, par_left_val) = gamax::partition::partition_column_pair(left[0], left[i], params.begin_bit, params.end_bit, nullptr, 0, stream);
            res_table->add_column(simple_materialize(par_left_val->view(), left_tid->view(), stream));
        }
        left_tid.reset(nullptr);
        
        res_table->add_column(simple_materialize(par_right_val->view(), right_tid->view(), stream));
        for(auto i = 2; i < right.num_columns(); i++) {
            std::tie(par_right_key, par_right_val) = gamax::partition::partition_column_pair(right[0], right[i], params.begin_bit, params.end_bit, nullptr, 0, stream);
            res_table->add_column(simple_materialize(par_right_val->view(), right_tid->view(), stream));
        }
        
        return res_table;
    }

    std::unique_ptr<gamax::table> ieq_partition_hash_join_gfur(
        gamax::table_view left,
        gamax::table_view right,
        gamax::join::join_hint const& hint,
        cudaStream_t stream)
    {
        gamax::join::detail::partition_hash_join_params params(left, right, hint);

        if(params.early_materialization) {
            return ieq_partition_hash_join_narrow(left, right, params, stream);
        }

        // transformation
        auto left_idx_seq = gamax::make_sequence(left.num_rows());
        auto [par_left_key, par_left_idx_seq, par_left_offs] = partition_transform(left[0], left_idx_seq->view(), params.begin_bit, params.end_bit, stream);
        left_idx_seq.reset(nullptr);

        auto right_idx_seq = gamax::make_sequence(right.num_rows());
        auto [par_right_key, par_right_idx_seq, par_right_offs] = partition_transform(right[0], right_idx_seq->view(), params.begin_bit, params.end_bit, stream);
        right_idx_seq.reset(nullptr);

        // join co-partitions
        auto [res_key, left_tid, right_tid] = detail::join_copartitions(
            par_left_key->view(), par_left_idx_seq->view(), par_left_offs->view(),
            par_right_key->view(), par_right_idx_seq->view(), par_right_offs->view(),
            params, stream);
        
        par_left_key.reset(nullptr);
        par_right_key.reset(nullptr);
        par_left_offs.reset(nullptr);
        par_left_idx_seq.reset(nullptr);
        par_right_idx_seq.reset(nullptr);
        par_right_offs.reset(nullptr);

        // materialize
        auto res_table = std::make_unique<gamax::table>();
        res_table->add_column(std::move(res_key));
        for(auto i = 1; i < left.num_columns(); i++) {
            res_table->add_column(simple_materialize(left[i], left_tid->view(), stream));
        }
        left_tid.reset(nullptr);
        
        for(auto i = 1; i < right.num_columns(); i++) {
            res_table->add_column(simple_materialize(right[i], right_tid->view(), stream));
        }
        
        return res_table;
    }
} // namespace detail
} // namespace gamax::join

