#include "phys/phys.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

#include "operators/groupby/groupby.hpp"
#include "operators/join/cross_join.hpp"
#include "operators/join/inner_join.hpp"
#include "operators/join/join.hpp"
#include "operators/join/anti_join.hpp"
#include "operators/join/semi_join.hpp"
#include "operators/project/project.hpp"
#include "operators/select/select.hpp"
#include "operators/sort/sort.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "utils/copying.hpp"
#include "utils/transform.hpp"

namespace gamax::phys {

namespace {

[[noreturn]] void phys_error(const std::string& what) {
    throw std::runtime_error("phys::" + what);
}

[[noreturn]] void not_implemented(const std::string& op, const std::string& why) {
    throw std::runtime_error("phys::" + op + ": not yet implemented — " + why);
}

// Build a 0-row column of the given type. STRING dispatches to
// make_string_column with num_strings=0 (the factory accepts 0).
gamax::column_ptr make_zero_row_column(
    gamax::data_type type, cudaStream_t stream)
{
    if (type.id() == gamax::type_id::STRING) {
        // Passing nullptr for flat_data/lens with 0 strings: the
        // factory shouldn't dereference. Using tiny static buffers
        // is robust against backends that check pointers.
        static const char empty_data[1] = {'\0'};
        static const int32_t empty_len[1] = {0};
        auto ptr = gamax::make_string_column(
            empty_data, empty_len, 0, stream, gamax::ARROW_STRING);
        return std::unique_ptr<gamax::column>(std::move(ptr));
    }
    return gamax::make_column(type, 0, /*clear=*/false, stream);
}

// Build a 1-row zero-initialized column. Used by scalar_aggregate on
// an empty input (SQL says "one row of NULL"; we approximate with 0).
gamax::column_ptr make_one_row_zero_column(
    gamax::data_type type, cudaStream_t stream)
{
    if (type.id() == gamax::type_id::STRING) {
        static const char empty_data[1] = {'\0'};
        static const int32_t empty_len[1] = {0};
        auto ptr = gamax::make_string_column(
            empty_data, empty_len, 1, stream, gamax::ARROW_STRING);
        return std::unique_ptr<gamax::column>(std::move(ptr));
    }
    return gamax::make_column(type, 1, /*clear=*/true, stream);
}

// Deep-copy a single column regardless of its type. For numeric columns
// falls back to `gamax::copy_column`; for STRING uses an identity
// gather over `gather_strings` so the caller owns a freshly allocated
// string column.
//
// Extracted (and hoisted) from the emitter's ad-hoc `copy_any_column`
// so the STRING-passthrough path lives with the physical layer, not
// inside IR emission.
gamax::column_ptr copy_any_column(
    gamax::column_view col, cudaStream_t stream)
{
    if (col.type().id() == gamax::type_id::STRING) {
        auto ids = gamax::make_sequence<gamax::size_type>(
            col.size(), 0, 1, stream);
        gamax::string_column_view sv(col, gamax::ARROW_STRING, stream);
        auto gathered = gamax::gather_strings(sv, ids->view(), stream);
        return std::unique_ptr<gamax::column>(std::move(gathered));
    }
    return gamax::copy_column(col, stream);
}

// Apply a BOOL8 bitmask to a selected subset of columns of `in`,
// producing a compact output table. An empty `ordinals` means "all
// columns in their original order"; otherwise only those ordinals are
// materialized (in that order, with repeats allowed).
//
// Mirrors the heterogeneous-filter pattern from `tpch_ops.cu`: build
// `compact_tids` once (by compacting an identity id sequence under the
// mask) and reuse it for every STRING column via `gather_strings`;
// compact numeric columns directly. The selective-output path shares
// the same tids — the bitmask (and thus surviving row set) is a
// property of the input, not of which columns we materialize.
gamax::phys::table_ptr apply_bitmask(
    gamax::table_view in,
    gamax::column_view mask,
    const std::vector<gamax::size_type>& ordinals,  // empty = all
    cudaStream_t stream)
{
    const bool all_cols = ordinals.empty();
    auto col_at = [&](gamax::size_type slot) -> gamax::size_type {
        return all_cols ? slot : ordinals[slot];
    };
    const auto n_out = all_cols
        ? static_cast<gamax::size_type>(in.num_columns())
        : static_cast<gamax::size_type>(ordinals.size());

    bool any_string = false;
    for (gamax::size_type i = 0; i < n_out; ++i) {
        if (in.get_column(col_at(i)).type().id() == gamax::type_id::STRING) {
            any_string = true;
            break;
        }
    }
    // Fast path: no STRING columns + all-columns output → one compact
    // over the whole table. (A selective all-numeric output would still
    // go through the per-column loop below since `compact` takes a
    // whole table view.)
    if (!any_string && all_cols) {
        return gamax::util::device::compact(in, mask, stream);
    }

    std::vector<gamax::column_ptr> out;
    out.reserve(n_out);
    gamax::column_ptr compact_tids;
    auto ensure_tids = [&]() {
        if (!compact_tids) {
            auto seq = gamax::make_sequence<gamax::size_type>(
                in.num_rows(), 0, 1, stream);
            compact_tids = gamax::util::device::compact(
                seq->view(), mask, stream);
        }
    };
    for (gamax::size_type i = 0; i < n_out; ++i) {
        auto col = in.get_column(col_at(i));
        if (col.type().id() == gamax::type_id::STRING) {
            ensure_tids();
            gamax::string_column_view sv(col, gamax::ARROW_STRING, stream);
            auto gathered = gamax::gather_strings(
                sv, compact_tids->view(), stream);
            out.push_back(std::unique_ptr<gamax::column>(std::move(gathered)));
        } else {
            out.push_back(
                gamax::util::device::compact(col, mask, stream));
        }
    }
    return std::make_unique<gamax::table>(std::move(out));
}

} // namespace

// ── Data reshaping ────────────────────────────────────────────────

table_ptr reorder(gamax::table_view in,
                  std::vector<gamax::size_type> ordinals,
                  const Context& ctx)
{
    std::vector<gamax::column_ptr> cols;
    cols.reserve(ordinals.size());
    const bool empty = (in.num_rows() == 0);
    for (auto o : ordinals) {
        if (o < 0 || o >= in.num_columns()) {
            phys_error("reorder: ordinal " + std::to_string(o) +
                       " out of range for " +
                       std::to_string(in.num_columns()) + "-column input");
        }
        // On 0-row input, build a fresh 0-row column of the right
        // type rather than touching the (possibly invalid-data)
        // source column via copy/gather — avoids hitting kernels
        // that don't handle size 0 cleanly.
        cols.push_back(empty
            ? make_zero_row_column(in.get_column(o).type(), ctx.stream)
            : copy_any_column(in.get_column(o), ctx.stream));
    }
    return std::make_unique<gamax::table>(std::move(cols));
}

// ── Computation ───────────────────────────────────────────────────

table_ptr project(gamax::table_view in,
                  std::vector<ProjectItem>& items,
                  const Context& ctx)
{
    // Per-item classification. Five kinds:
    //   * Predicate        → select_to_bitmask (BOOL8 out)
    //   * ExprTree ColRef  → copy_any_column   (STRING-safe)
    //   * Computed ExprTree→ batched project::project
    //   * Substring        → gamax::substring  (STRING out)
    //   * (empty input is a separate branch below)
    enum class Kind { Predicate, ColRef, Computed, Substring };
    const auto n = static_cast<gamax::size_type>(items.size());
    std::vector<Kind> kinds(n, Kind::Computed);
    std::vector<gamax::expression::ExprTree> computed;
    for (gamax::size_type i = 0; i < n; ++i) {
        if (std::holds_alternative<gamax::select::Predicate>(items[i])) {
            kinds[i] = Kind::Predicate;
        } else if (std::holds_alternative<SubstringItem>(items[i])) {
            kinds[i] = Kind::Substring;
        } else {
            auto& e = std::get<gamax::expression::ExprTree>(items[i]);
            if (e.is_column_ref()) {
                kinds[i] = Kind::ColRef;
            } else {
                kinds[i] = Kind::Computed;
                computed.push_back(e);
            }
        }
    }

    // Empty input → 0 rows of each item's declared output type.
    // Predicate items always produce BOOL8; Substring items always
    // produce STRING.
    if (in.num_rows() == 0) {
        std::vector<gamax::column_ptr> cols;
        cols.reserve(n);
        for (gamax::size_type i = 0; i < n; ++i) {
            if (kinds[i] == Kind::Predicate) {
                cols.push_back(make_zero_row_column(
                    gamax::data_type{gamax::type_id::BOOL8}, ctx.stream));
            } else if (kinds[i] == Kind::Substring) {
                cols.push_back(make_zero_row_column(
                    gamax::data_type{gamax::type_id::STRING}, ctx.stream));
            } else {
                auto& e = std::get<gamax::expression::ExprTree>(items[i]);
                cols.push_back(make_zero_row_column(e.type(), ctx.stream));
            }
        }
        return std::make_unique<gamax::table>(std::move(cols));
    }

    // Batch the computed ExprTrees through project::project once,
    // if any.
    gamax::project::project_options opts;  // defaults
    std::unique_ptr<gamax::table> computed_tbl;
    if (!computed.empty()) {
        computed_tbl = gamax::project::project(
            in, computed, opts, ctx.stream);
    }

    // Stitch per-item results in positional order.
    std::vector<gamax::column_ptr> out;
    out.reserve(n);
    for (gamax::size_type i = 0; i < n; ++i) {
        switch (kinds[i]) {
            case Kind::Predicate: {
                auto& pred = std::get<gamax::select::Predicate>(items[i]);
                auto bm = gamax::select::select_to_bitmask(in, pred, ctx.stream);
                out.push_back(std::move(bm));
                break;
            }
            case Kind::ColRef: {
                auto& e = std::get<gamax::expression::ExprTree>(items[i]);
                out.push_back(copy_any_column(
                    in.get_column(e.get_column_index()), ctx.stream));
                break;
            }
            case Kind::Computed: {
                // Pull the next computed column out of computed_tbl in
                // request order. remove_column(0) keeps head-taking
                // semantics deterministic.
                out.push_back(computed_tbl->remove_column(0));
                break;
            }
            case Kind::Substring: {
                auto& si = std::get<SubstringItem>(items[i]);
                gamax::string_column_view src_view(
                    in.get_column(si.src_col),
                    gamax::ARROW_STRING,
                    ctx.stream);
                auto r = gamax::substring(
                    src_view, si.sql_offset, si.length, ctx.stream);
                out.push_back(std::unique_ptr<gamax::column>(std::move(r)));
                break;
            }
        }
    }
    return std::make_unique<gamax::table>(std::move(out));
}

table_ptr filter(gamax::table_view in,
                 gamax::select::Predicate& pred,
                 const Context& ctx)
{
    return filter(in, pred, /*output_ordinals=*/{}, ctx);
}

table_ptr filter(gamax::table_view in,
                 gamax::select::Predicate& pred,
                 std::vector<gamax::size_type> output_ordinals,
                 const Context& ctx)
{
    const bool all_cols = output_ordinals.empty();
    // Bounds-check ordinals eagerly: a bad ordinal here would otherwise
    // surface only deep inside the gather path, with a confusing error.
    for (auto o : output_ordinals) {
        if (o < 0 || o >= in.num_columns()) {
            phys_error("filter: output ordinal " + std::to_string(o) +
                       " out of range for " +
                       std::to_string(in.num_columns()) + "-column input");
        }
    }

    // Empty input → 0 rows of the requested output schema.
    if (in.num_rows() == 0) {
        std::vector<gamax::column_ptr> cols;
        const auto n_out = all_cols
            ? static_cast<gamax::size_type>(in.num_columns())
            : static_cast<gamax::size_type>(output_ordinals.size());
        cols.reserve(n_out);
        for (gamax::size_type i = 0; i < n_out; ++i) {
            auto ord = all_cols ? i : output_ordinals[i];
            cols.push_back(make_zero_row_column(
                in.get_column(ord).type(), ctx.stream));
        }
        return std::make_unique<gamax::table>(std::move(cols));
    }

    // The Predicate may reference ANY ordinal of `in`, not just the
    // ones in `output_ordinals`. select_to_bitmask runs over the full
    // view; only the materialization is selective.
    auto mask = gamax::select::select_to_bitmask(in, pred, ctx.stream);
    return apply_bitmask(in, mask->view(), output_ordinals, ctx.stream);
}

// ── Joins ─────────────────────────────────────────────────────────

namespace {

// Single-key-only for now; multi-key goes through `multikey_compress`
// in a future follow-up. Throws a clear message until then.
void require_single_key(const std::string& op,
                        const std::vector<gamax::size_type>& lkeys,
                        const std::vector<gamax::size_type>& rkeys)
{
    if (lkeys.size() != rkeys.size()) {
        phys_error(op + ": lkeys (" + std::to_string(lkeys.size()) +
                   ") and rkeys (" + std::to_string(rkeys.size()) +
                   ") must have the same length");
    }
    if (lkeys.size() != 1) {
        not_implemented(op, "multi-column keys (underlying "
                            "inner_eq_join / semi_join is single-key; "
                            "key-fusion via multikey_compress is "
                            "tracked separately)");
    }
}

gamax::join::join_hint default_join_hint() {
    // `inner_eq_join` rejects AUTO_ALGO / AUTO_MAT, so pick a concrete
    // pair. HASH+GFTR matches the existing join tests and works
    // across shapes; callers that need a specific algo can plumb this
    // later via `Context` or `AggSpec`-style overrides.
    gamax::join::join_hint h;
    h.join_algo = gamax::join::HASH;
    h.mat_algo  = gamax::join::GFTR;
    return h;
}

// Gather one column regardless of type. Mirrors `copy_any_column` but
// with an arbitrary index map rather than identity.
gamax::column_ptr gather_any(
    gamax::column_view src,
    gamax::column_view gather_map,
    cudaStream_t stream)
{
    if (src.type().id() == gamax::type_id::STRING) {
        gamax::string_column_view sv(src, gamax::ARROW_STRING, stream);
        auto g = gamax::gather_strings(sv, gather_map, stream);
        return std::unique_ptr<gamax::column>(std::move(g));
    }
    return gamax::util::device::gather(gather_map, src, stream);
}

// Build a 2-column "skinny" table `[key, row_id]` for one side of a
// join. The row_id is a fresh identity sequence used later to gather
// each original (non-key) column into its output position — that's
// how this wrapper routes around the join backend's refusal to
// materialize STRING payloads (B7b).
gamax::phys::table_ptr build_key_plus_rowid(
    gamax::table_view in,
    const std::vector<gamax::size_type>& keys,
    const gamax::phys::Context& ctx)
{
    auto key_tbl = gamax::phys::reorder(in, keys, ctx);
    auto ids = gamax::make_sequence<gamax::size_type>(
        in.num_rows(), 0, 1, ctx.stream);
    key_tbl->add_column(std::move(ids));
    return key_tbl;
}

// Deep-copy one column, handling STRING via identity gather — same
// shape as `copy_any_column` above but declared here because the
// join-assembly code is in the anonymous namespace.
gamax::column_ptr deep_copy_any(gamax::column_view src, cudaStream_t stream) {
    if (src.type().id() == gamax::type_id::STRING) {
        auto ids = gamax::make_sequence<gamax::size_type>(
            src.size(), 0, 1, stream);
        gamax::string_column_view sv(src, gamax::ARROW_STRING, stream);
        auto g = gamax::gather_strings(sv, ids->view(), stream);
        return std::unique_ptr<gamax::column>(std::move(g));
    }
    return gamax::copy_column(src, stream);
}

// Assemble the final output of a (self-driven) eq-join: shared key at
// position 0, then each of L's non-key columns in their original L
// order, then each of R's non-key columns in their original R order.
// Uses the row-id columns left over from the join to gather payloads.
gamax::phys::table_ptr assemble_eq_join_output(
    gamax::table_view L,
    const std::vector<gamax::size_type>& lkeys,
    gamax::table_view R,
    const std::vector<gamax::size_type>& rkeys,
    gamax::column_view shared_key,
    gamax::column_view l_row_id,
    gamax::column_view r_row_id,
    cudaStream_t stream)
{
    std::vector<gamax::column_ptr> out;
    out.reserve(1 +
                (L.num_columns() - lkeys.size()) +
                (R.num_columns() - rkeys.size()));
    out.push_back(deep_copy_any(shared_key, stream));

    auto is_key = [](gamax::size_type i,
                     const std::vector<gamax::size_type>& ks) {
        return std::find(ks.begin(), ks.end(), i) != ks.end();
    };

    for (gamax::size_type i = 0; i < L.num_columns(); ++i) {
        if (is_key(i, lkeys)) continue;
        out.push_back(gather_any(L.get_column(i), l_row_id, stream));
    }
    for (gamax::size_type i = 0; i < R.num_columns(); ++i) {
        if (is_key(i, rkeys)) continue;
        out.push_back(gather_any(R.get_column(i), r_row_id, stream));
    }
    return std::make_unique<gamax::table>(std::move(out));
}

} // namespace

namespace {

// 0-row eq_join output: [shared_key_type, L_non_keys..., R_non_keys...].
gamax::phys::table_ptr make_empty_eq_join_output(
    gamax::table_view L, const std::vector<gamax::size_type>& lkeys,
    gamax::table_view R, const std::vector<gamax::size_type>& rkeys,
    cudaStream_t stream)
{
    std::vector<gamax::column_ptr> cols;
    cols.reserve(1 +
                 (L.num_columns() - lkeys.size()) +
                 (R.num_columns() - rkeys.size()));
    // The shared-key column's type is the L key type (equal to R's).
    cols.push_back(gamax::phys::make_zero_row_column(
        L.get_column(lkeys[0]).type(), stream));
    for (gamax::size_type i = 0; i < L.num_columns(); ++i) {
        if (std::find(lkeys.begin(), lkeys.end(), i) != lkeys.end()) continue;
        cols.push_back(gamax::phys::make_zero_row_column(
            L.get_column(i).type(), stream));
    }
    for (gamax::size_type i = 0; i < R.num_columns(); ++i) {
        if (std::find(rkeys.begin(), rkeys.end(), i) != rkeys.end()) continue;
        cols.push_back(gamax::phys::make_zero_row_column(
            R.get_column(i).type(), stream));
    }
    return std::make_unique<gamax::table>(std::move(cols));
}

// 0-row semi_join output: [shared_key_type, L_non_keys...].
gamax::phys::table_ptr make_empty_semi_join_output(
    gamax::table_view L, const std::vector<gamax::size_type>& lkeys,
    cudaStream_t stream)
{
    std::vector<gamax::column_ptr> cols;
    cols.reserve(1 + L.num_columns() - lkeys.size());
    cols.push_back(gamax::phys::make_zero_row_column(
        L.get_column(lkeys[0]).type(), stream));
    for (gamax::size_type i = 0; i < L.num_columns(); ++i) {
        if (std::find(lkeys.begin(), lkeys.end(), i) != lkeys.end()) continue;
        cols.push_back(gamax::phys::make_zero_row_column(
            L.get_column(i).type(), stream));
    }
    return std::make_unique<gamax::table>(std::move(cols));
}

} // namespace

table_ptr eq_join(gamax::table_view L, std::vector<gamax::size_type> lkeys,
                  gamax::table_view R, std::vector<gamax::size_type> rkeys,
                  const Context& ctx)
{
    require_single_key("eq_join", lkeys, rkeys);

    // Empty-input short-circuit: inner_eq_join is the primary
    // caller of thrust reduce in the backend; 0-row inputs blow up
    // at "determine reduce temp storage size".
    if (L.num_rows() == 0 || R.num_rows() == 0) {
        return make_empty_eq_join_output(L, lkeys, R, rkeys, ctx.stream);
    }

    // Join a skinny [key, row_id] on each side. The underlying
    // inner_eq_join is numeric-only at the output materialization
    // step, so we keep its inputs numeric and do the payload
    // materialization ourselves via gather — this closes B7b (STRING
    // payloads in the join input) uniformly for all column types.
    auto L_small = build_key_plus_rowid(L, lkeys, ctx);
    auto R_small = build_key_plus_rowid(R, rkeys, ctx);

    auto joined = gamax::join::inner_eq_join(
        L_small->view(), R_small->view(), default_join_hint(), ctx.stream);
    // Joined shape is [shared_key, L_row_id, R_row_id] (3 columns —
    // the single-key dedup matches the ieq_join_test.cu invariant
    // `rcols+scols-1`).
    auto shared_key = joined->view()[0];
    auto l_row_id   = joined->view()[1];
    auto r_row_id   = joined->view()[2];

    return assemble_eq_join_output(
        L, lkeys, R, rkeys, shared_key, l_row_id, r_row_id, ctx.stream);
}

table_ptr semi_join(gamax::table_view L, std::vector<gamax::size_type> lkeys,
                    gamax::table_view R, std::vector<gamax::size_type> rkeys,
                    const Context& ctx)
{
    require_single_key("semi_join", lkeys, rkeys);

    if (L.num_rows() == 0 || R.num_rows() == 0) {
        return make_empty_semi_join_output(L, lkeys, ctx.stream);
    }

    auto L_small = build_key_plus_rowid(L, lkeys, ctx);
    auto R_small = build_key_plus_rowid(R, rkeys, ctx);

    auto joined = gamax::join::semi_join(
        L_small->view(), R_small->view(), default_join_hint(), ctx.stream);
    // Semi-join output carries only the left (surviving) rows: shape
    // is [shared_key, L_row_id] (2 columns, matching the single-key
    // convention).
    auto shared_key = joined->view()[0];
    auto l_row_id   = joined->view()[1];

    // Output = [shared_key, L_non_keys_in_original_order], gathered
    // via L_row_id.
    std::vector<gamax::column_ptr> out;
    out.reserve(1 + L.num_columns() - lkeys.size());
    out.push_back(deep_copy_any(shared_key, ctx.stream));
    for (gamax::size_type i = 0; i < L.num_columns(); ++i) {
        if (std::find(lkeys.begin(), lkeys.end(), i) != lkeys.end()) continue;
        out.push_back(gather_any(L.get_column(i), l_row_id, ctx.stream));
    }
    return std::make_unique<gamax::table>(std::move(out));
}

table_ptr left_outer_join(
    gamax::table_view, std::vector<gamax::size_type>,
    gamax::table_view, std::vector<gamax::size_type>, const Context&)
{
    not_implemented("left_outer_join",
                    "no underlying left_outer_join operator yet");
}

table_ptr anti_join(gamax::table_view L, std::vector<gamax::size_type> lkeys,
                    gamax::table_view R, std::vector<gamax::size_type> rkeys,
                    const Context& ctx)
{
    require_single_key("anti_join", lkeys, rkeys);

    // Anti-join edge cases:
    //   * R empty → no key can collide → every L row survives → return L.
    //   * L empty → nothing to survive → return L-shaped empty table.
    if (L.num_rows() == 0) {
        return make_empty_semi_join_output(L, lkeys, ctx.stream);
    }
    if (R.num_rows() == 0) {
        // Shape matches semi_join's output: [shared_key, L_non_keys...].
        // Here the "shared key" is just L's key column itself (no right
        // side intersected).
        std::vector<gamax::column_ptr> out;
        out.reserve(1 + L.num_columns() - lkeys.size());
        out.push_back(deep_copy_any(L.get_column(lkeys[0]), ctx.stream));
        for (gamax::size_type i = 0; i < L.num_columns(); ++i) {
            if (std::find(lkeys.begin(), lkeys.end(), i) != lkeys.end()) continue;
            out.push_back(deep_copy_any(L.get_column(i), ctx.stream));
        }
        return std::make_unique<gamax::table>(std::move(out));
    }

    auto L_small = build_key_plus_rowid(L, lkeys, ctx);
    auto R_small = build_key_plus_rowid(R, rkeys, ctx);

    // Anti-join in the operator layer yields the left rows (with row-id)
    // that have no match in right. Output shape matches the [key, row_id]
    // layout that semi_join produces, because `gamax::join::anti_join`
    // compacts the same `L_small` table.
    auto joined = gamax::join::anti_join(
        L_small->view(), R_small->view(), default_join_hint(), ctx.stream);
    auto shared_key = joined->view()[0];
    auto l_row_id   = joined->view()[1];

    // Materialise [key, L_non_keys...] by gathering L's non-key columns
    // through l_row_id — identical materialisation shape to semi_join.
    std::vector<gamax::column_ptr> out;
    out.reserve(1 + L.num_columns() - lkeys.size());
    out.push_back(deep_copy_any(shared_key, ctx.stream));
    for (gamax::size_type i = 0; i < L.num_columns(); ++i) {
        if (std::find(lkeys.begin(), lkeys.end(), i) != lkeys.end()) continue;
        out.push_back(gather_any(L.get_column(i), l_row_id, ctx.stream));
    }
    return std::make_unique<gamax::table>(std::move(out));
}

table_ptr cross_join(gamax::table_view L, gamax::table_view R,
                     const Context& ctx)
{
    // Thin delegation to the dedicated operator, now living alongside
    // the other join types under `operators/join/`. See cross_join.cu
    // for the tuple-id-gather implementation.
    return gamax::join::cross_join(L, R, ctx.stream);
}

// ── Aggregation ───────────────────────────────────────────────────

table_ptr scalar_aggregate(gamax::table_view values,
                           std::vector<AggSpec> specs,
                           const Context& ctx)
{
    // Scalar aggregate over 0 rows — SQL would say "1 row of NULL";
    // we approximate with 1 row of zeros, typed per-AggSpec.
    if (values.num_rows() == 0) {
        std::vector<gamax::column_ptr> cols;
        cols.reserve(specs.size());
        for (const auto& s : specs) {
            auto in_type  = values.get_column(s.column).type();
            auto out_type = gamax::groupby::agg_output_data_type(s.agg, in_type);
            cols.push_back(make_one_row_zero_column(out_type, ctx.stream));
        }
        return std::make_unique<gamax::table>(std::move(cols));
    }
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.reserve(specs.size());
    for (const auto& s : specs) requests.emplace_back(s.column, s.agg);
    return gamax::groupby::aggregate::aggregate(
        values, requests, ctx.stream);
}

table_ptr group_by(gamax::table_view keys,
                   gamax::table_view values,
                   std::vector<AggSpec> specs,
                   const Context& ctx)
{
    // Grouped aggregate over 0 rows → 0 groups → 0-row output
    // whose schema is [key_types..., agg_output_types...].
    if (keys.num_rows() == 0) {
        std::vector<gamax::column_ptr> cols;
        cols.reserve(keys.num_columns() + specs.size());
        for (int i = 0; i < keys.num_columns(); ++i) {
            cols.push_back(make_zero_row_column(
                keys.get_column(i).type(), ctx.stream));
        }
        for (const auto& s : specs) {
            auto in_type  = values.get_column(s.column).type();
            auto out_type = gamax::groupby::agg_output_data_type(s.agg, in_type);
            cols.push_back(make_zero_row_column(out_type, ctx.stream));
        }
        return std::make_unique<gamax::table>(std::move(cols));
    }
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.reserve(specs.size());
    for (const auto& s : specs) requests.emplace_back(s.column, s.agg);
    gamax::groupby::groupby_hint hint;
    // SKF (smart-key-fusion) left OFF for now. The `groupby_noskf` path
    // combined with the new STRING→HASH / multi-col→HASH routing in
    // `auto_select_groupby_algo` is sufficient to cover STRING and
    // mixed keys; enabling SKF is an independent perf question.
    return gamax::groupby::groupby(
        keys, values, requests, hint, ctx.stream);
}

// ── Ordering ──────────────────────────────────────────────────────

namespace {

// Helper: 0-row copy of the input schema, used by sort/top_k/filter
// when input is empty.
gamax::phys::table_ptr zero_rows_like(
    gamax::table_view in, cudaStream_t stream)
{
    std::vector<gamax::column_ptr> cols;
    cols.reserve(in.num_columns());
    for (int i = 0; i < in.num_columns(); ++i) {
        cols.push_back(gamax::phys::make_zero_row_column(
            in.get_column(i).type(), stream));
    }
    return std::make_unique<gamax::table>(std::move(cols));
}

} // namespace

table_ptr sort(gamax::table_view in,
               std::vector<SortSpec> keys,
               const Context& ctx)
{
    if (in.num_rows() == 0) return zero_rows_like(in, ctx.stream);
    std::vector<int> k;
    std::vector<gamax::sort_order> o;
    k.reserve(keys.size());
    o.reserve(keys.size());
    for (const auto& s : keys) {
        k.push_back(static_cast<int>(s.column));
        o.push_back(s.order);
    }
    return gamax::sort::sort_table(in, k, o, ctx.stream);
}

table_ptr top_k(gamax::table_view in,
                std::vector<SortSpec> keys,
                gamax::size_type k,
                const Context& ctx)
{
    if (in.num_rows() == 0) return zero_rows_like(in, ctx.stream);
    std::vector<int> ks;
    std::vector<gamax::sort_order> os;
    ks.reserve(keys.size());
    os.reserve(keys.size());
    for (const auto& s : keys) {
        ks.push_back(static_cast<int>(s.column));
        os.push_back(s.order);
    }
    return gamax::sort::topk_table(
        in, ks, os, k, /*result_in_sorted_order=*/true, ctx.stream);
}

} // namespace gamax::phys
