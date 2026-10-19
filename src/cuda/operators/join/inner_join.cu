// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "inner_join.hpp"

#include <cuda.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "types/data_type.hpp"
#include "operators/join/join.hpp"

namespace gamax {
namespace join {
namespace {
void check_join_solvable(
    gamax::table_view left,
    gamax::table_view right,
    gamax::join::join_hint const& hint)
{
    if(hint.join_algo == gamax::join::AUTO_ALGO || hint.mat_algo == gamax::join::AUTO_MAT) {
        throw std::runtime_error("Join algorithm and materialization algorithm do not support auto selection");
    }

    if(hint.max_out > gamax::join::MAX_JOIN_OUTPUT_SIZE) {
        throw std::runtime_error("Output size exceeds the maximum limit");
    }

    if(hint.mat_algo == gamax::join::INDEX_CLUSTER) {
        throw std::runtime_error("Index cluster materialization algorithm is not supported");
    }

    if(left[0].type() != right[0].type()) {
        throw std::runtime_error("Join columns cannot have different types");
    }

    if (left.num_columns() == 0 || right.num_columns() == 0) {
        throw std::runtime_error("Empty table");
    }

    if (left.num_columns() == 1 || right.num_columns() == 1) {
        throw std::runtime_error("Left and right table should have at least 1 payload column");
    }
}

bool any_string_payload(gamax::table_view t) {
    for (gamax::size_type i = 1; i < t.num_columns(); ++i) {
        if (t[i].type().id() == gamax::type_id::STRING) return true;
    }
    return false;
}

std::unique_ptr<gamax::table> dispatch_inner_eq_join(
    gamax::table_view left,
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream)
{
    switch (hint.join_algo) {
        case gamax::join::SORT:
            return hint.mat_algo == gamax::join::GFUR ?
                detail::ieq_sort_merge_join_gfur(left, right, hint, stream) :
                detail::ieq_sort_merge_join_gftr(left, right, hint, stream);
        case gamax::join::HASH:
            return detail::ieq_hash_join(left, right, hint, stream);
        case gamax::join::PARTITION:
            return hint.mat_algo == gamax::join::GFUR ?
                detail::ieq_partition_hash_join_gfur(left, right, hint, stream) :
                detail::ieq_partition_hash_join_gftr(left, right, hint, stream);
        default:
            throw std::runtime_error("Unsupported join algorithm");
    }
}

// Shape carried through the GFTR-string wrapper: a zero-copy
// `[key, numeric_payloads..., row_id_delegate]` view over `t`, plus
// the owned row_id sequence and the segregation of payload positions.
struct skinny_side {
    gamax::table_view view;
    std::unique_ptr<gamax::column> row_id;     // delegate column owned here
    // For each input payload position i in [1, t.num_columns()):
    //   true  → STRING, gather from original column via materialised row_id
    //   false → numeric, take from the inner-join output's L/R numeric slice
    std::vector<bool> payload_is_string;
    gamax::size_type n_numeric_payloads {0};   // count of false entries above
};

skinny_side make_skinny(gamax::table_view t, cudaStream_t stream) {
    skinny_side s;
    s.payload_is_string.reserve(t.num_columns() - 1);

    std::vector<gamax::column_view> cols;
    cols.reserve(t.num_columns() + 1);
    cols.push_back(t[0]);
    for (gamax::size_type i = 1; i < t.num_columns(); ++i) {
        const bool is_str = (t[i].type().id() == gamax::type_id::STRING);
        s.payload_is_string.push_back(is_str);
        if (!is_str) {
            cols.push_back(t[i]);
            ++s.n_numeric_payloads;
        }
    }
    s.row_id = gamax::make_sequence<gamax::size_type>(t.num_rows(), stream, 0, 1);
    cols.push_back(s.row_id->view());
    s.view = gamax::table_view(cols);
    return s;
}

std::unique_ptr<gamax::column> gather_string_column(
    gamax::column_view src,
    gamax::column_view tids,
    cudaStream_t stream)
{
    gamax::string_column_view sv(src, gamax::ARROW_STRING);
    auto g = gamax::gather_strings(sv, tids, stream);
    return std::unique_ptr<gamax::column>(std::move(g));
}

// GFTR + any STRING payload → segregate STRINGs onto a tid-delegate
// path. Numeric payloads (and the delegate) flow through GFTR
// normally; STRING payloads are gathered post-join from the original
// L / R views using the materialised delegate. Output is spliced back
// into input column order so callers see the same shape as numeric-
// only / GFUR / HASH paths.
std::unique_ptr<gamax::table> inner_eq_join_gftr_with_strings(
    gamax::table_view left,
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream)
{
    auto L_skin = make_skinny(left, stream);
    auto R_skin = make_skinny(right, stream);

    // L_skin / R_skin contain no STRINGs in payload positions, so this
    // call routes straight to the numeric GFTR dispatch.
    auto inner = dispatch_inner_eq_join(L_skin.view, R_skin.view, hint, stream);

    // Inner output layout:
    //   [0]                            shared_key
    //   [1 .. 1+L_n)                   L numeric payloads (skin order)
    //   [1+L_n]                        L row_id_delegate (materialised)
    //   [2+L_n .. 2+L_n+R_n)           R numeric payloads (skin order)
    //   [2+L_n+R_n]                    R row_id_delegate (materialised)
    const auto L_n = L_skin.n_numeric_payloads;
    const auto R_n = R_skin.n_numeric_payloads;
    const auto l_delegate_idx = 1 + L_n;
    const auto r_delegate_idx = 2 + L_n + R_n;

    // release_column leaves nullptr in the slot, so subsequent
    // releases keep their original indices.
    auto l_tids = inner->release_column(l_delegate_idx);
    auto r_tids = inner->release_column(r_delegate_idx);

    std::vector<std::unique_ptr<gamax::column>> out;
    out.reserve(1 + (left.num_columns() - 1) + (right.num_columns() - 1));
    out.push_back(inner->release_column(0));   // shared_key

    // Walk L payload positions in input order. Numeric → take next
    // slot from inner's L numeric slice; STRING → gather original.
    {
        gamax::size_type next_num_slot = 1;
        for (size_t i = 0; i < L_skin.payload_is_string.size(); ++i) {
            if (L_skin.payload_is_string[i]) {
                // Original L position is (i + 1) — payload positions
                // start after the key.
                out.push_back(gather_string_column(
                    left[static_cast<gamax::size_type>(i) + 1],
                    l_tids->view(), stream));
            } else {
                out.push_back(inner->release_column(next_num_slot++));
            }
        }
    }
    // R numeric slice starts at index (l_delegate_idx + 1).
    {
        gamax::size_type next_num_slot = l_delegate_idx + 1;
        for (size_t i = 0; i < R_skin.payload_is_string.size(); ++i) {
            if (R_skin.payload_is_string[i]) {
                out.push_back(gather_string_column(
                    right[static_cast<gamax::size_type>(i) + 1],
                    r_tids->view(), stream));
            } else {
                out.push_back(inner->release_column(next_num_slot++));
            }
        }
    }

    return std::make_unique<gamax::table>(std::move(out));
}
} // namespace

namespace detail {
    gamax::size_type get_ieq_join_output_alloc_size(
        gamax::table_view const& left,
        gamax::table_view const& right,
        gamax::join::join_hint const& hint) {

        if(hint.max_out != 0) return hint.max_out;
        // A unique side bounds the output by the OTHER side's row count (each row
        // of the non-unique side matches at most one row of the unique side). Only
        // when NEITHER side is unique is the output unbounded by cardinality —
        // then fall back to the MAX_JOIN_OUTPUT_SIZE heuristic. Do NOT cap the
        // unique-side bound at MAX: that under-allocates and corrupts memory once
        // the bound exceeds 2^30 (e.g. a PK/FK join with > 2^30 FK rows).
        constexpr gamax::size_type kUnbounded = std::numeric_limits<gamax::size_type>::max();
        const gamax::size_type bound_l = hint.unique_left  ? right.num_rows() : kUnbounded;
        const gamax::size_type bound_r = hint.unique_right ? left.num_rows()  : kUnbounded;
        const gamax::size_type bound   = std::min(bound_l, bound_r);
        return (bound == kUnbounded) ? gamax::join::MAX_JOIN_OUTPUT_SIZE : bound;
    }
} // namespace detail

std::unique_ptr<gamax::table> inner_eq_join(
    gamax::table_view left,
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream)
{
    check_join_solvable(left, right, hint);

    // GFTR variants partition / sort each (key, payload) pair, which
    // can't accept STRING payloads. Wrap with a tid-delegate trick:
    // route numerics + a row_id delegate through GFTR, then gather
    // STRINGs from the original views using the materialised delegate.
    // HASH and the GFUR variants gather payloads after the join via
    // simple_materialize — STRING-safe already, no wrapper needed.
    const bool gftr_algo =
        hint.mat_algo == gamax::join::GFTR &&
        (hint.join_algo == gamax::join::SORT ||
         hint.join_algo == gamax::join::PARTITION);
    if (gftr_algo && (any_string_payload(left) || any_string_payload(right))) {
        return inner_eq_join_gftr_with_strings(left, right, hint, stream);
    }

    return dispatch_inner_eq_join(left, right, hint, stream);
}
} // namespace join
} // namespace gamax
