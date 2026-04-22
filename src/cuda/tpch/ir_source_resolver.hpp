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
//   Scan("lineitem", {cols...}) → SourceTable("lineitem")
//                               → Reorder(ordinals-into-full-catalog)
//
// That is, `SourceTable`'s schema is the **full catalog table**; the
// subsequent Reorder step picks out the requested subset in order.
// So any resolver wired to this lowering must return a `table_view`
// containing *every* catalog-declared column of the table in the
// catalog's declared order.
//
// The resolver loads each table lazily on first resolve, passing the
// full catalog column list down to `TpchTableReader::load`. Loaded
// tables are cached for the resolver's lifetime, so within one
// `execute()` call a table is loaded at most once even if multiple
// Scan nodes reference it.
//
// Column-pruning-at-the-Scan is a separate optimization: that will
// make `SourceTable`'s schema the scanned-subset, so the resolver
// only loads the columns the plan actually reads. Tracked as a
// follow-up rule; today we trade that wasted H2D for a uniform
// SourceTable layout.
class IrSourceResolver {
public:
    IrSourceResolver(TpchTableReader& reader, const ir::Catalog& catalog)
        : _reader(&reader), _catalog(&catalog) {}

    // Returns a view over the full catalog-shaped table. Keeps the
    // underlying `gamax::table` alive for as long as this resolver
    // object does (so the Executor's `table_view` stays valid for
    // the entire `execute()` call).
    //
    // Call-site pattern for wiring into `exec::execute`:
    //
    //   IrSourceResolver resolver(reader, catalog);
    //   exec::SourceResolver src_fn =
    //       [&resolver](const std::string& name) { return resolver(name); };
    //   exec::execute(plan, src_fn, ctx);
    //
    // (Direct `static_cast<SourceResolver>(resolver)` won't compile:
    // `std::function` demands a copyable functor, and this class
    // holds a `unique_ptr` cache that makes it move-only.)
    gamax::table_view operator()(const std::string& table_name);

private:
    TpchTableReader* _reader;
    const ir::Catalog* _catalog;
    std::unordered_map<std::string, std::unique_ptr<gamax::table>> _cache;
};

} // namespace gamax::tpch
