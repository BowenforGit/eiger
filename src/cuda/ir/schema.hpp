#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ir/column_id.hpp"
#include "types/data_type.hpp"

namespace gamax::ir {

// A single column in a plan node's output.
//
// `id` is the load-bearing correctness handle: every reference to this
// column from an expression or operator spec uses this ColumnId. `name`
// is cosmetic (used for printers, frontend name resolution, and error
// messages) and may be duplicated within a schema — e.g. after a join of
// two tables that share a column name.
struct Field {
    ColumnId id;
    gamax::data_type type;
    bool nullable {false};
    std::string name;
};

// Ordered list of fields describing a plan node's output.
//
// The order is meaningful: when a physical operator materializes the
// output, the n-th result column corresponds to the n-th Field here.
// Inside the IR, expressions refer to columns by ColumnId only, so
// reordering a schema (via an InsertReshapeProject rule) does not
// invalidate any reference elsewhere.
class Schema {
public:
    Schema() = default;
    explicit Schema(std::vector<Field> fields) : _fields(std::move(fields)) {}

    size_t size() const noexcept { return _fields.size(); }
    bool empty() const noexcept { return _fields.empty(); }

    const Field& operator[](size_t i) const { return _fields[i]; }
    const std::vector<Field>& fields() const noexcept { return _fields; }

    // Linear search by ColumnId. Returns the position within this schema
    // where the column resides, or std::nullopt if absent.
    std::optional<size_t> index_of(ColumnId id) const {
        for (size_t i = 0; i < _fields.size(); ++i) {
            if (_fields[i].id == id) return i;
        }
        return std::nullopt;
    }

    // First field matching a name. Intended for frontend / builder use
    // (binding unresolved identifiers to ColumnIds). Correctness inside
    // the IR must not depend on name lookup.
    std::optional<size_t> index_of(const std::string& name) const {
        for (size_t i = 0; i < _fields.size(); ++i) {
            if (_fields[i].name == name) return i;
        }
        return std::nullopt;
    }

    // Throws if the column is absent. Use when the caller believes the
    // id must be present (e.g. an expression referencing a column from
    // its declared input).
    size_t require_index(ColumnId id) const {
        auto i = index_of(id);
        if (!i) {
            throw std::runtime_error(
                "Schema::require_index: column not in schema");
        }
        return *i;
    }

    void append(Field f) { _fields.push_back(std::move(f)); }

private:
    std::vector<Field> _fields;
};

} // namespace gamax::ir
