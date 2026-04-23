#include "anti_join.hpp"

#include <cuda.h>
#include <memory>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/join/semi_join.hpp"
#include "utils/transform.hpp"

namespace gamax::join {
namespace {

void check_anti_join_solvable(
    gamax::table_view left,
    gamax::table_view right,
    gamax::join::join_hint const& /*hint*/)
{
    if (left.num_columns() == 0 || right.num_columns() == 0) {
        throw std::runtime_error("[anti-join] Empty table");
    }
    if (left[0].type() != right[0].type()) {
        throw std::runtime_error("[anti-join] Join columns cannot have different types");
    }
}

} // namespace

namespace detail {
    std::unique_ptr<gamax::table> anti_hash_join(
        gamax::table_view left,
        gamax::table_view right,
        gamax::join::join_hint const& /*hint*/,
        cudaStream_t stream)
    {
        // Anti-join = left rows that are NOT in the right key set.
        // Reuse semi_hash's membership-mask pass, then compact `left`
        // by the negated mask.
        auto mask = gamax::join::detail::semi_hash_membership_mask(
            left, right, stream);
        auto anti_mask = gamax::util::device::bitmask_not(mask->view(), stream);
        auto res = gamax::util::device::compact(left, anti_mask->view(), stream);
        cudaStreamSynchronize(stream);
        return res;
    }
} // namespace detail

std::unique_ptr<gamax::table> anti_join(
    gamax::table_view left,
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream)
{
    check_anti_join_solvable(left, right, hint);

    switch (hint.join_algo) {
        case gamax::join::HASH:
            return detail::anti_hash_join(left, right, hint, stream);
        default:
            throw std::runtime_error("Unsupported anti-join algorithm");
    }
}

} // namespace gamax::join
