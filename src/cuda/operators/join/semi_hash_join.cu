#include "semi_join.hpp"

#include <cuda.h>
#include <memory>
#include "utils/transform.hpp"
#include "types/sentinel.hpp"

#include "containers/set.cuh"

namespace gamax::join::detail {
namespace {
    struct semi_hash_join_impl {
        template<typename Key, std::enable_if_t<sizeof(Key) != sizeof(cuda::atomic<Key>) || !cuco::is_bitwise_comparable_v<Key>, int> = 0>
        std::unique_ptr<gamax::table> operator()(
            gamax::table_view left, 
            gamax::table_view right,
            gamax::join::join_hint const& hint,
            cudaStream_t stream)
        {
            throw std::runtime_error("Semi-join is not supported for non-integral keys.");
        }

        template<typename Key, std::enable_if_t<sizeof(Key) == sizeof(cuda::atomic<Key>) && cuco::is_bitwise_comparable_v<Key>, int> = 0>
        std::unique_ptr<gamax::table> operator()(
            gamax::table_view left, 
            gamax::table_view right,
            gamax::join::join_hint const& hint,
            cudaStream_t stream)
        {
            auto set = gamax::containers::set<Key>(
                right[0].size(), 
                0.25, 
                gamax::sentinel<Key>(),
                stream);
            
            set.insert(right[0], stream);
            auto mask = set.contains(left[0], stream);


            auto res = gamax::util::device::compact(left, mask->view(), stream);
            
            cudaStreamSynchronize(stream);
            
            return res;
        }   
    };
}

    std::unique_ptr<gamax::table> semi_hash_join(
        gamax::table_view left, 
        gamax::table_view right,
        gamax::join::join_hint const& hint,
        cudaStream_t stream)
    {
        return gamax::type_dispatcher(
            right[0].type(), 
            semi_hash_join_impl{}, 
            left, 
            right, 
            hint, 
            stream);
    }
} // namespace gamax::join::detail