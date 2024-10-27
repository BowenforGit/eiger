#include "join.hpp"

#include <memory>
#include <vector>
#include <cuda.h>
#include <type_traits>

#include "materialize.hpp"
#include "merge_path.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/sort/sort.hpp"
#include "thrust/iterator/counting_iterator.h"

namespace gamax {
namespace join {
namespace detail {
namespace {
    using triple_col_t = std::tuple<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>>;
    using tid_type = gamax::size_type;

    std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>>
    sort_transform(gamax::column_view key, gamax::column_view val, cudaStream_t stream) {
        return gamax::sort::sort_column_pair(key, val, 0, -1, gamax::sort_order::ASCENDING, nullptr, 0, stream);
    }

    bool narrow_join(const gamax::table_view& left, const gamax::table_view& right) {
        return left.num_columns() == 2 && right.num_columns() == 2 && left[1].type() == right[1].type();
    }

    struct merge_path_impl {
        template <typename T>
        void operator()(
            gamax::column_view left_key, 
            gamax::column_view right_key,
            gamax::mutable_column_view res_key,
            gamax::mutable_column_view left_tid,
            gamax::mutable_column_view right_tid,
            gamax::size_type& n_matches,
            cudaStream_t stream) 
        {            
            thrust::counting_iterator<tid_type> left_itr(0);
            thrust::counting_iterator<tid_type> right_itr(0);

            // TODO - make stream work
            // TODO - make merge_path more "modern"
            merge_path(
                const_cast<T*>(left_key.data<T>()),
                const_cast<T*>(right_key.data<T>()),
                left_itr,
                right_itr,
                left_key.size(),
                right_key.size(),
                res_key.data<T>(),
                left_tid.data<tid_type>(),
                right_tid.data<tid_type>(),
                n_matches,
                res_key.size() // used as the circular buffer size
            );

            cudaStreamSynchronize(stream);
        }

        template <typename T, typename U>
        void operator()(
            gamax::column_view left_key, 
            gamax::column_view right_key,
            gamax::column_view left_val,
            gamax::column_view right_val,
            gamax::mutable_column_view res_key,
            gamax::mutable_column_view res_left,
            gamax::mutable_column_view res_right,
            gamax::size_type& n_matches,
            cudaStream_t stream) 
        {   
            // TODO - make stream work
            // TODO - make n_matches gamax::size_type
            merge_path(
                const_cast<T*>(left_key.data<T>()),
                const_cast<T*>(right_key.data<T>()),
                left_val.data<U>(),
                right_val.data<U>(),
                left_key.size(),
                right_key.size(),
                res_key.data<T>(),
                res_left.data<U>(),
                res_right.data<U>(),
                n_matches,
                res_key.size() // used as the circular buffer size
            );

            cudaStreamSynchronize(stream);

            res_key.set_size(n_matches);
            res_left.set_size(n_matches);
            res_right.set_size(n_matches);
        }
    };

    /**
     * sorted_left_val and sorted_right_val must have the same type
     */
    triple_col_t merge_join(
        gamax::column_view sorted_left_key, 
        gamax::column_view sorted_right_key, 
        gamax::column_view sorted_left_val,
        gamax::column_view sorted_right_val,
        const gamax::size_type alloc_out_size,
        cudaStream_t stream
    ) {
        auto key_type = sorted_left_key.type();
        auto val_type = sorted_left_val.type();

        auto res_key = std::make_unique<gamax::column>(key_type, alloc_out_size);
        auto res_left = gamax::make_column(val_type, alloc_out_size);
        auto res_right = gamax::make_column(val_type, alloc_out_size);

        gamax::size_type n_matches = 0;

        double_type_dispatcher(
            key_type,
            val_type,
            merge_path_impl{},
            sorted_left_key,
            sorted_right_key,
            sorted_left_val,
            sorted_right_val,
            res_key->mutable_view(),
            res_left->mutable_view(),
            res_right->mutable_view(),
            n_matches,
            stream
        );

        res_key->set_size(n_matches);
        res_left->set_size(n_matches);
        res_right->set_size(n_matches);

        return std::make_tuple(std::move(res_key), std::move(res_left), std::move(res_right));
    }

    triple_col_t merge_join(
        gamax::column_view sorted_left_key, 
        gamax::column_view sorted_right_key, 
        const gamax::size_type alloc_out_size,
        cudaStream_t stream
    ) {
        auto key_type = sorted_left_key.type();

        auto res_key = std::make_unique<gamax::column>(key_type, alloc_out_size);
        auto left_tid = gamax::make_idx_column(alloc_out_size);
        auto right_tid = gamax::make_idx_column(alloc_out_size);

        gamax::size_type n_matches = 0;

        type_dispatcher(
            key_type,
            merge_path_impl{},
            sorted_left_key,
            sorted_right_key,
            res_key->mutable_view(),
            left_tid->mutable_view(),
            right_tid->mutable_view(),
            n_matches,
            stream
        );

        res_key->set_size(n_matches);
        left_tid->set_size(n_matches);
        right_tid->set_size(n_matches);

        return std::make_tuple(std::move(res_key), std::move(left_tid), std::move(right_tid));
    }

    std::unique_ptr<gamax::table> ieq_sort_merge_join_narrow(
        gamax::table_view left, 
        gamax::table_view right,
        gamax::join::join_hint const& hint,
        cudaStream_t stream)
    {     
        // transformation
        auto [sorted_left_key, sorted_left_val] = sort_transform(left[0], left[1], stream);
        auto [sorted_right_key, sorted_right_val] = sort_transform(right[0], right[1], stream);

        // join
        const auto alloc_out_size = get_join_output_alloc_size(left, right, hint); // number of rows in the output table to be allocated
        auto [res_key, res_left, res_right] = merge_join(
            sorted_left_key->view(), sorted_right_key->view(), 
            sorted_left_val->view(), sorted_right_val->view(),
            alloc_out_size, stream);

        auto res_table = std::make_unique<gamax::table>();
        res_table->add_column(std::move(res_key));
        res_table->add_column(std::move(res_left));
        res_table->add_column(std::move(res_right));
        
        return res_table;
    }
} // namespace
    
    std::unique_ptr<gamax::table> ieq_sort_merge_join_gftr(
        gamax::table_view left, 
        gamax::table_view right,
        gamax::join::join_hint const& hint,
        cudaStream_t stream)
    {   
        if(narrow_join(left, right)) {
            return ieq_sort_merge_join_narrow(left, right, hint, stream);
        }
        
        // transformation
        auto [sorted_left_key, sorted_left_val] = sort_transform(left[0], left[1], stream);
        auto [sorted_right_key, sorted_right_val] = sort_transform(right[0], right[1], stream);

        // join
        const auto alloc_out_size = get_join_output_alloc_size(left, right, hint); // number of rows in the output table to be allocated
        auto [res_key, left_tid, right_tid] = merge_join(sorted_left_key->view(), sorted_right_key->view(), alloc_out_size, stream);
        sorted_left_key.reset(nullptr);
        sorted_right_key.reset(nullptr);

        // materialize
        auto res_table = std::make_unique<gamax::table>();
        res_table->add_column(std::move(res_key));
        res_table->add_column(simple_materialize(sorted_left_val->view(), left_tid->view(), stream));
        sorted_left_val.reset(nullptr);
        for(auto i = 2; i < left.num_columns(); i++) {
            std::tie(sorted_left_key, sorted_left_val) = sort_transform(left[0], left[i], stream);
            res_table->add_column(simple_materialize(sorted_left_val->view(), left_tid->view(), stream));
        }
        left_tid.reset(nullptr);
        
        res_table->add_column(simple_materialize(sorted_right_val->view(), right_tid->view(), stream));
        for(auto i = 2; i < right.num_columns(); i++) {
            std::tie(sorted_right_key, sorted_right_val) = sort_transform(right[0], right[i], stream);
            res_table->add_column(simple_materialize(sorted_right_val->view(), right_tid->view(), stream));
        }
        
        return res_table;
    }

    std::unique_ptr<gamax::table> ieq_sort_merge_join_gfur(
        gamax::table_view left, 
        gamax::table_view right,
        gamax::join::join_hint const& hint,
        cudaStream_t stream)
    {     
        if(narrow_join(left, right)) {
            return ieq_sort_merge_join_narrow(left, right, hint, stream);
        }

        // transformation
        auto left_idx_seq = gamax::make_sequence(left.num_rows());
        auto [sorted_left_key, sorted_left_idx_seq] = sort_transform(left[0], left_idx_seq->view(), stream);
        left_idx_seq.reset(nullptr);

        auto right_idx_seq = gamax::make_sequence(right.num_rows());
        auto [sorted_right_key, sorted_right_idx_seq] = sort_transform(right[0], right_idx_seq->view(), stream);
        right_idx_seq.reset(nullptr);

        // join
        const auto alloc_out_size = get_join_output_alloc_size(left, right, hint); // number of rows in the output table to be allocated
        auto [res_key, left_tid, right_tid] = merge_join(
            sorted_left_key->view(), sorted_right_key->view(), 
            sorted_left_idx_seq->view(), sorted_right_idx_seq->view(), 
            alloc_out_size, stream);
        
        sorted_left_key.reset(nullptr);
        sorted_right_key.reset(nullptr);
        sorted_left_idx_seq.reset(nullptr);
        sorted_right_idx_seq.reset(nullptr);

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
} // namespace join
} // namespace gamax