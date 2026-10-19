// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "tpch/ir_source_resolver.hpp"

#include <stdexcept>

namespace gamax::tpch {

namespace {

// Form a deterministic cache key from (name, cols). Columns are
// positional (order-significant), so we do NOT sort — different
// orders get different cache entries so each caller gets its view
// in its requested order.
std::string make_key(
    const std::string& name, const std::vector<std::string>& cols)
{
    std::string k = name;
    k.push_back('|');
    for (const auto& c : cols) {
        k.append(c);
        k.push_back('|');
    }
    return k;
}

} // namespace

gamax::table_view IrSourceResolver::operator()(
    const std::string& table_name,
    const std::vector<std::string>& columns)
{
    auto key = make_key(table_name, columns);
    auto it = _cache.find(key);
    if (it != _cache.end()) return it->second->view();

    if (!_catalog->find(table_name)) {
        throw std::runtime_error(
            "IrSourceResolver: '" + table_name + "' not in catalog");
    }

    // TpchTableReader::load returns columns in the requested order and
    // reuses its pinned per-column cache across calls, so two Scans of
    // the same table with different subsets only pay the parquet read once.
    auto tbl = _reader->load(table_name, columns);

    auto ins = _cache.emplace(std::move(key), std::move(tbl));
    return ins.first->second->view();
}

} // namespace gamax::tpch
