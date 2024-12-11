#include "table_gen.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <limits>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/generate.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "data_gen/generator.cuh"
#include "utils/transform.hpp"
#include "types/type_dispatcher.hpp"

namespace gamax::datagen::device {
namespace {
    struct check_valid_start_helper {
        template<typename T>
        void operator()(
            const uint64_t start)
        {
            if(start > std::numeric_limits<T>::max()) {
                throw std::runtime_error("Start value exceeds the maximum value of the data type");
            }
        }
    };

    void check_valid_start(
        const gamax::data_type type,
        const uint64_t start)
    {
        type_dispatcher(type, check_valid_start_helper{}, start);
    }

    struct circular_fill_impl {
        template<typename T>
        void operator()(
            gamax::column_view src,
            gamax::mutable_column_view res,
            const gamax::size_type n,
            cudaStream_t stream)
        {
            const auto num_values = src.size();
            thrust::for_each_n(
                thrust::cuda::par(*get_thrust_allocator()).on(stream), 
                thrust::counting_iterator<gamax::size_type>(0),
                n, 
                [src, res, num_values] __device__ (gamax::size_type i) {
                    res.data<T>()[i] = src.data<T>()[i % num_values];
            });
        }
    };

    std::unique_ptr<gamax::column> generate_from_values(
    gamax::column_view values, // set of values to sample from
    const gamax::size_type n,
    const bool shuffle = true, 
    const int seed = 42,
    cudaStream_t stream = cudaStreamDefault) 
    {
        const auto num_values = values.size();

        auto ret = std::make_unique<gamax::column>(values.type(), n);
        type_dispatcher(values.type(), circular_fill_impl{}, values, ret->mutable_view(), n, stream);
        cudaStreamSynchronize(stream);

        if(shuffle) {
            return gamax::util::device::shuffle(ret->view(), seed, stream);
        } else {
            return ret;
        }
    }
    
    struct apply_match_ratio_impl {
        template<typename T>
        void operator()(
            gamax::mutable_column_view src,
            const double match_ratio)
        {
            const auto nkeys = src.size();
            if(match_ratio < 1.0) {
                const gamax::size_type inv_ratio = match_ratio != 0 ? 
                                                    (1.0 / match_ratio) 
                                                    : nkeys+1;
                thrust::for_each_n(
                    thrust::cuda::par(*get_thrust_allocator()), 
                    thrust::counting_iterator<gamax::size_type>(0),
                    nkeys, 
                    [src, inv_ratio] __device__ (gamax::size_type i) {
                        constexpr T leftmost_one = static_cast<T>(1) << (sizeof(T) * 8 - 1);
                        if(i % inv_ratio != 0) {
                            src.data<T>()[i] += leftmost_one;
                        }
                    });
                
                std::cout << "Match ratio is adjusted to " << match_ratio << std::endl;
            }
        }
    };

    void apply_match_ratio(
        gamax::mutable_column_view src,
        const double match_ratio)
    {
        type_dispatcher(src.type(), apply_match_ratio_impl{}, src, match_ratio);
    }

    struct pk_uniform_id_impl {
        template<typename T>
        std::unique_ptr<gamax::table> operator()(
            const gamax::size_type card,
            const gamax::size_type ncols,
            const int seed,
            const uint64_t start) const
        {
            auto src = gamax::make_sequence<T>(card, static_cast<T>(start));
            return generate_table(src->view(), card, ncols, seed);
        }
    };

    // struct pk_uniform_sparse_impl {
    //     template<typename T>
    //     std::unique_ptr<gamax::table> operator()(
    //         const gamax::size_type card,
    //         const gamax::size_type ncols,
    //         const int seed) const
    //     {
    //         constexpr T max_elem = std::numeric_limits<T>::max() / 2 - 1; // generate values from [0, max_elem]
    //         const auto n_unique_keys = card;

    //         const auto range_len = max_elem / n_unique_keys;
    //         if(range_len < 1) {
    //             throw std::runtime_error("Number of unique keys exceeds the maximum value available");
    //         }
            
    //         auto idx_helper = gamax::datagen::device::generate(
    //             gamax::datagen::device::uniform_int_gen<T>(0, range_len-1), 
    //             n_unique_keys);
            
    //         auto src = gamax::make_column<T>(n_unique_keys);
    //         thrust::for_each_n(
    //             thrust::device, 
    //             thrust::counting_iterator<gamax::size_type>(0),
    //             n_unique_keys, 
    //             [=] __device__ (gamax::size_type i) {
    //                 src->data<T>()[i] = idx_helper->data<gamax::size_type>()[i] + i * range_len; // TODO: this seems wrong
    //             });
            
    //         return generate_table(src->view(), card, ncols, seed);
    //     }
    // };

    struct fk_uniform_id_impl {
        template<typename T>
        std::unique_ptr<gamax::table> operator()(
            const gamax::size_type n_unique_keys,
            const gamax::size_type card,
            const gamax::size_type ncols,
            const double match_ratio,
            const int seed,
            const uint64_t start) const
        {
            auto src = gamax::make_sequence<T>(n_unique_keys, static_cast<T>(start));
            return fk_uniform(src->view(), card, ncols, match_ratio, seed);
        }
    };

    struct fk_zipf_id_impl {
        template<typename T>
        std::unique_ptr<gamax::table> operator()(
            const gamax::size_type n_unique_keys,
            const gamax::size_type card,
            const gamax::size_type ncols,
            const double zipf_factor,
            const int seed,
            const uint64_t start) const
        {
            auto seq = gamax::make_sequence<T>(n_unique_keys, static_cast<T>(start));
            return fk_zipf(seq->view(), card, ncols, zipf_factor, seed);
        }
    };
} // namespace

    std::unique_ptr<gamax::table>
    generate_table(
        gamax::column_view src,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const int seed)
    {
        auto keys = generate_from_values(src, card, true, seed);

        auto t = std::make_unique<gamax::table>();
        for(gamax::size_type i = 0; i < ncols; i++) {
            t->add_column(keys->deep_copy());
        }

        cudaDeviceSynchronize();
        return t;
    }

    std::unique_ptr<gamax::table>
    pk_uniform_id(
        const gamax::data_type type,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const int seed,
        const uint64_t start)
    {
        check_valid_start(type, start);
        return type_dispatcher(type, pk_uniform_id_impl{}, card, ncols, seed, start);
    }

    // std::unique_ptr<gamax::table>
    // pk_uniform_sparse(
    //     const gamax::data_type type,
    //     const gamax::size_type card,
    //     const gamax::size_type ncols,
    //     const int seed)
    // {
    //     return type_dispatcher(type, pk_uniform_sparse_impl{}, card, ncols, seed);
    // }

    std::unique_ptr<gamax::table>
    fk_uniform_id(
        const gamax::data_type type,
        const gamax::size_type n_unique_keys,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const double match_ratio,
        const int seed,
        const uint64_t start)
    {
        check_valid_start(type, start);
        return type_dispatcher(type, fk_uniform_id_impl{}, n_unique_keys, card, ncols, match_ratio, seed, start);
    }

    std::unique_ptr<gamax::table>
    fk_zipf_id(
        const gamax::data_type type,
        const gamax::size_type n_unique_keys,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const double zipf_factor,
        const int seed,
        const uint64_t start)
    {
        check_valid_start(type, start);
        return type_dispatcher(type, fk_zipf_id_impl{}, n_unique_keys, card, ncols, zipf_factor, seed, start);
    }

    std::unique_ptr<gamax::table>
    fk_uniform(
        gamax::column_view seq,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const double match_ratio,
        const int seed)
    {
        std::cout << "[!!WARNING!!] Modifying match ratio for FK only works when the key are IDs" << std::endl;
        auto seq_copy = gamax::make_column_from_view(seq);
        apply_match_ratio(seq_copy->mutable_view(), match_ratio);
        return generate_table(seq_copy->view(), card, ncols, seed);
    }

    std::unique_ptr<gamax::table>
    fk_zipf(
        gamax::column_view seq,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const double zipf_factor,
        const int seed)
    {   
        gamax::size_type* h_idx = nullptr;
        create_integral_relation_nonunique(&h_idx, card, false, zipf_factor, seq.size(), seed);
        auto d_idx = gamax::make_column_from_host(h_idx, card);
        auto src = gamax::util::device::gather(d_idx->view(), seq);

        delete[] h_idx;

        return generate_table(src->view(), card, ncols, seed);
    }
}