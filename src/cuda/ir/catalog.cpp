#include "ir/catalog.hpp"

#include <stdexcept>

namespace gamax::ir {

void Catalog::add_table(CatalogTable table) {
    if (table.name.empty()) {
        throw std::runtime_error("Catalog::add_table: table name is empty");
    }
    if (table.column_names.size() != table.column_types.size()
        || table.column_names.size() != table.column_nullable.size()) {
        throw std::runtime_error(
            "Catalog::add_table: column_names / column_types / column_nullable must be the same size");
    }
    auto name = table.name;
    auto [it, inserted] = _tables.emplace(std::move(name), std::move(table));
    if (!inserted) {
        throw std::runtime_error("Catalog::add_table: duplicate table name '"
                                 + it->first + "'");
    }
}

const CatalogTable* Catalog::find(const std::string& name) const {
    auto it = _tables.find(name);
    if (it == _tables.end()) return nullptr;
    return &it->second;
}

} // namespace gamax::ir
