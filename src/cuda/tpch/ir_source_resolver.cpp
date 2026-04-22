#include "tpch/ir_source_resolver.hpp"

#include <stdexcept>

namespace gamax::tpch {

gamax::table_view IrSourceResolver::operator()(const std::string& table_name) {
    auto it = _cache.find(table_name);
    if (it != _cache.end()) return it->second->view();

    const auto* ct = _catalog->find(table_name);
    if (!ct) {
        throw std::runtime_error(
            "IrSourceResolver: '" + table_name + "' not in catalog");
    }

    // Load every catalog-declared column in declared order. Matches
    // the SourceTable schema that `ir::lower_scan` emits.
    std::vector<std::string> cols = ct->column_names;
    auto tbl = _reader->load(table_name, std::move(cols));

    auto ins = _cache.emplace(table_name, std::move(tbl));
    return ins.first->second->view();
}

} // namespace gamax::tpch
