#include "inner_join.hpp"

#include <memory>
#include <cuda.h>
#include <cuda/atomic>
#include <type_traits>

#include "join.hpp"
#include "materialize.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "containers/multimap.cuh"
#include "types/sentinel.hpp"

namespace gamax::join {
namespace detail {

namespace {
    using tid_type = gamax::size_type;

    struct hash_join_impl {
        template<typename Key, std::enable_if_t<sizeof(Key) == sizeof(cuda::atomic<Key>), int> = 0>
        std::unique_ptr<gamax::table> operator()(
            gamax::table_view left, 
            gamax::table_view right,
            gamax::join::join_hint const& hint,
            cudaStream_t stream)
        {
            auto max_output_size = get_ieq_join_output_alloc_size(left, right, hint);

            auto map = gamax::containers::multimap<Key>(
                left.num_rows(), 
                gamax::sentinel<Key>(),
                gamax::sentinel<tid_type>(),
                *get_thrust_allocator<char>(stream),
                stream);
            
            map.insert_with_tids(left[0], stream);

            auto [res_keys, res_left_tids, res_right_tids] = map.probe(right[0], max_output_size, stream);

            // materialize
            auto res_table = std::make_unique<gamax::table>();
            res_table->add_column(std::move(res_keys));
            for(auto i = 1; i < left.num_columns(); i++) {
                res_table->add_column(simple_materialize(left[i], res_left_tids->view(), stream));
            }
            res_left_tids.reset(nullptr);
            
            for(auto i = 1; i < right.num_columns(); i++) {
                res_table->add_column(simple_materialize(right[i], res_right_tids->view(), stream));
            }
            
            cudaStreamSynchronize(stream);
            
            return res_table;
        }   

        template<typename Key, std::enable_if_t<sizeof(Key) != sizeof(cuda::atomic<Key>), int> = 0>
        std::unique_ptr<gamax::table> operator()(
            gamax::table_view left, 
            gamax::table_view right,
            gamax::join::join_hint const& hint,
            cudaStream_t stream)
        {
            throw std::runtime_error("Hash join only supports atomic types");
        } 
    };
} // namespace

std::unique_ptr<gamax::table> ieq_hash_join(
    gamax::table_view left, 
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream)
{
    return gamax::type_dispatcher(
        left[0].type(), 
        hash_join_impl{}, 
        left, 
        right, 
        hint, 
        stream);
}
} // namespace detail
} // namespace gamax::join