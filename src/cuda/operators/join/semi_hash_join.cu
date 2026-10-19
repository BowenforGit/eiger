// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "semi_join.hpp"

#include <cuda.h>
#include <memory>
#include "utils/transform.hpp"
#include "types/sentinel.hpp"

#include "containers/set.cuh"

namespace gamax::join::detail {
namespace {
    template <typename Key>
    class typed_semi_hash_set final : public semi_hash_set {
    public:
        typed_semi_hash_set(
            gamax::column_view right_key,
            cudaStream_t stream)
            : set_(
                right_key.size(),
                0.25,
                gamax::sentinel<Key>(),
                stream)
        {
            set_.insert(right_key, stream);
        }

        std::unique_ptr<gamax::column> contains(
            gamax::column_view keys,
            cudaStream_t stream) override
        {
            return set_.contains(keys, stream);
        }

    private:
        gamax::containers::set<Key> set_;
    };

    struct build_semi_hash_set_impl {
        template<typename Key, std::enable_if_t<sizeof(Key) != sizeof(cuda::atomic<Key>) || !cuco::is_bitwise_comparable_v<Key>, int> = 0>
        std::unique_ptr<semi_hash_set> operator()(
            gamax::column_view,
            cudaStream_t)
        {
            throw std::runtime_error(
                "Hash semi/anti-join is not supported for non-integral keys.");
        }

        template<typename Key, std::enable_if_t<sizeof(Key) == sizeof(cuda::atomic<Key>) && cuco::is_bitwise_comparable_v<Key>, int> = 0>
        std::unique_ptr<semi_hash_set> operator()(
            gamax::column_view right_key,
            cudaStream_t stream)
        {
            return std::make_unique<typed_semi_hash_set<Key>>(
                right_key, stream);
        }
    };

    // Membership-only functor: returns the bool-per-left-row mask of
    // whether left[0][i] is present in right[0]. Used by both
    // `semi_hash_join` and `anti_hash_join`.
    struct semi_hash_mask_impl {
        template<typename Key, std::enable_if_t<sizeof(Key) != sizeof(cuda::atomic<Key>) || !cuco::is_bitwise_comparable_v<Key>, int> = 0>
        std::unique_ptr<gamax::column> operator()(
            gamax::table_view /*left*/,
            gamax::table_view /*right*/,
            cudaStream_t /*stream*/)
        {
            throw std::runtime_error("Hash semi/anti-join is not supported for non-integral keys.");
        }

        template<typename Key, std::enable_if_t<sizeof(Key) == sizeof(cuda::atomic<Key>) && cuco::is_bitwise_comparable_v<Key>, int> = 0>
        std::unique_ptr<gamax::column> operator()(
            gamax::table_view left,
            gamax::table_view right,
            cudaStream_t stream)
        {
            auto set = gamax::containers::set<Key>(
                right[0].size(),
                0.25,
                gamax::sentinel<Key>(),
                stream);
            set.insert(right[0], stream);
            return set.contains(left[0], stream);
        }
    };
}

    std::unique_ptr<semi_hash_set> build_semi_hash_set(
        gamax::column_view right_key,
        cudaStream_t stream)
    {
        return gamax::type_dispatcher(
            right_key.type(),
            build_semi_hash_set_impl{},
            right_key,
            stream);
    }

    std::unique_ptr<gamax::column> semi_hash_membership_mask(
        gamax::table_view left,
        gamax::table_view right,
        cudaStream_t stream)
    {
        return gamax::type_dispatcher(
            right[0].type(),
            semi_hash_mask_impl{},
            left,
            right,
            stream);
    }

    std::unique_ptr<gamax::table> semi_hash_join(
        gamax::table_view left,
        gamax::table_view right,
        gamax::join::join_hint const& /*hint*/,
        cudaStream_t stream)
    {
        auto mask = semi_hash_membership_mask(left, right, stream);
        auto res = gamax::util::device::compact(left, mask->view(), stream);
        cudaStreamSynchronize(stream);
        return res;
    }
} // namespace gamax::join::detail
