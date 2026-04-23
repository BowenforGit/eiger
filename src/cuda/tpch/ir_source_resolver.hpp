#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "exec/executor.hpp"           // exec::SourceResolver
#include "ir/catalog.hpp"
#include "memory/table.hpp"
#include "tpch/tpch.hpp"               // TpchTableReader, tpch_params

namespace gamax::tpch {

// Adapter: turns an `ir::Catalog` + `TpchTableReader` into an
// `exec::SourceResolver` suitable for `exec::execute()`.
//
// The Executor's lowering model is:
//
//   Scan("lineitem", {cols...}) → SourceTable("lineitem", cols)
//
// That is, `SourceTable`'s schema is the **scanned subset** — no
// separate Reorder step — and the resolver must return a
// `table_view` whose `k`-th column is `columns[k]`. Column-pruning
// happens at the scan boundary so we never materialise unused
// columns on GPU.
//
// Each unique (table, columns) request is loaded lazily. Re-requests
// of the same key hit an in-resolver cache; requests for overlapping
// but not-identical subsets benefit from `TpchTableReader`'s per-column
// pinned cache, which avoids a second parquet read and a second H2D
// for already-seen columns.
class IrSourceResolver {
public:
    IrSourceResolver(TpchTableReader& reader, const ir::Catalog& catalog)
        : _reader(&reader), _catalog(&catalog) {}

    // Returns a view whose `k`-th column corresponds to `columns[k]`
    // in the named catalog table. Keeps the underlying `gamax::table`
    // alive for the resolver's lifetime so the returned view stays
    // valid for the entire `execute()` call.
    //
    // Call-site pattern for wiring into `exec::execute`:
    //
    //   IrSourceResolver resolver(reader, catalog);
    //   exec::SourceResolver src_fn =
    //       [&resolver](const std::string& name,
    //                   const std::vector<std::string>& cols) {
    //           return resolver(name, cols);
    //       };
    //   exec::execute(plan, src_fn, ctx);
    //
    // (Direct `static_cast<SourceResolver>(resolver)` won't compile:
    // `std::function` demands a copyable functor, and this class
    // holds a `unique_ptr` cache that makes it move-only.)
    gamax::table_view operator()(
        const std::string& table_name,
        const std::vector<std::string>& columns);

private:
    TpchTableReader* _reader;
    const ir::Catalog* _catalog;
    // Cache key is "name|col1|col2|...": unique per requested subset.
    std::unordered_map<std::string, std::unique_ptr<gamax::table>> _cache;
};

} // namespace gamax::tpch
