#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "types/data_type.hpp"

namespace gamax::ir {

// Metadata for a table known to the query planner. Maps a table name to
// its declared column names, types, and nullability. ColumnIds are NOT
// stored here; they are allocated fresh per Scan node, since two scans of
// the same table are two distinct column sources in a plan.
struct CatalogTable {
    std::string name;
    std::vector<std::string> column_names;
    std::vector<gamax::data_type> column_types;
    std::vector<bool> column_nullable;  // parallel to names / types
};

// Registry of available tables. Frontends (SQL parsers, builders) look up
// tables here when resolving scans — they need the name -> (type,
// nullable) information to materialize a Scan's output schema and to
// bind identifiers in expressions.
class Catalog {
public:
    // Registers a table. Throws on duplicate names or inconsistent
    // metadata (mismatched names/types/nullable sizes, empty name).
    void add_table(CatalogTable table);

    // Looks up a table by name. Returns nullptr if not found; callers
    // that want to error on absence do so themselves so the error
    // carries query-level context (e.g. a position in the source SQL).
    const CatalogTable* find(const std::string& name) const;

private:
    std::unordered_map<std::string, CatalogTable> _tables;
};

} // namespace gamax::ir
