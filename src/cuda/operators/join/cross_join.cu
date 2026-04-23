#include "cross_join.hpp"

#include <cuda.h>
#include <memory>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/join/materialize.hpp"
#include "types/data_type.hpp"

namespace gamax::join {

namespace {

// Build a size_type column of length `len` whose i-th entry is `i / divisor`
// (if div=true) or `i % divisor` (if div=false). Used as the tuple-id
// gather map fed to `simple_materialize`.
enum class tid_kind { quotient, remainder };

std::unique_ptr<gamax::column> build_tid_column(
    gamax::size_type len,
    gamax::size_type divisor,
    tid_kind kind,
    cudaStream_t stream)
{
    auto col = gamax::make_idx_column(len, /*clear=*/false, stream);
    auto* out = col->data<gamax::size_type>();
    auto first = thrust::make_counting_iterator<gamax::size_type>(0);
    if (kind == tid_kind::quotient) {
        thrust::transform(
            thrust::cuda::par.on(stream),
            first, first + len, out,
            [divisor] __host__ __device__ (gamax::size_type i) {
                return i / divisor;
            });
    } else {
        thrust::transform(
            thrust::cuda::par.on(stream),
            first, first + len, out,
            [divisor] __host__ __device__ (gamax::size_type i) {
                return i % divisor;
            });
    }
    return col;
}

std::unique_ptr<gamax::table> empty_cross_shape(
    gamax::table_view L, gamax::table_view R, cudaStream_t stream)
{
    std::vector<gamax::column_ptr> cols;
    cols.reserve(L.num_columns() + R.num_columns());
    auto push_zero_row = [&](gamax::column_view in) {
        // Match the existing phys::reorder "empty-input" branch: allocate
        // a 0-row column of the right type rather than gather through a
        // 0-length map (some gather kernels dislike 0-length inputs).
        cols.push_back(gamax::make_column(in.type(), 0, false, stream));
    };
    for (int i = 0; i < L.num_columns(); ++i) push_zero_row(L.get_column(i));
    for (int i = 0; i < R.num_columns(); ++i) push_zero_row(R.get_column(i));
    return std::make_unique<gamax::table>(std::move(cols));
}

} // namespace

std::unique_ptr<gamax::table> cross_join(
    gamax::table_view L,
    gamax::table_view R,
    cudaStream_t stream)
{
    const auto m = L.num_rows();
    const auto n = R.num_rows();
    if (m == 0 || n == 0) return empty_cross_shape(L, R, stream);

    const auto out_rows = m * n;

    // Two tuple-id gather maps: L side walks 0,0,...,1,1,...,m-1 (floor
    // division); R side walks 0..n-1 repeated m times (modulo). The
    // pair (L_map[i], R_map[i]) enumerates the Cartesian product in
    // "left slowest" row order.
    auto L_map = build_tid_column(out_rows, n, tid_kind::quotient,  stream);
    auto R_map = build_tid_column(out_rows, n, tid_kind::remainder, stream);

    std::vector<gamax::column_ptr> out_cols;
    out_cols.reserve(L.num_columns() + R.num_columns());
    for (int i = 0; i < L.num_columns(); ++i) {
        out_cols.push_back(gamax::join::detail::simple_materialize(
            L.get_column(i), L_map->view(), stream));
    }
    for (int i = 0; i < R.num_columns(); ++i) {
        out_cols.push_back(gamax::join::detail::simple_materialize(
            R.get_column(i), R_map->view(), stream));
    }
    return std::make_unique<gamax::table>(std::move(out_cols));
}

} // namespace gamax::join
