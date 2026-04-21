#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <ostream>

#include "types/data_type.hpp"  // for gamax::size_type

namespace gamax::ir {

// Stable identity for a column produced by a plan node.
//
// A ColumnId is assigned once when a column enters the plan (at a Scan,
// or at a Project for a computed expression) and preserved through every
// rewrite and through physical lowering. Expressions and operator specs
// inside the IR reference columns only via ColumnId, never by positional
// ordinal. At emit time, a resolver maps ColumnId -> ordinal against the
// concrete input schema of each physical operator.
//
// Uses gamax::size_type (signed int64) for consistency with the rest of
// the codebase. ColumnId{-1} is the "unset" sentinel; fresh ids from
// IdGenerator start at 0 and increase monotonically.
struct ColumnId {
    gamax::size_type value {-1};

    constexpr bool is_valid() const noexcept { return value >= 0; }
};

inline constexpr bool operator==(ColumnId lhs, ColumnId rhs) noexcept {
    return lhs.value == rhs.value;
}
inline constexpr bool operator!=(ColumnId lhs, ColumnId rhs) noexcept {
    return lhs.value != rhs.value;
}
inline constexpr bool operator<(ColumnId lhs, ColumnId rhs) noexcept {
    return lhs.value < rhs.value;
}

inline std::ostream& operator<<(std::ostream& os, ColumnId id) {
    return os << "#" << id.value;
}

// Allocates fresh ColumnIds. Multiple generators may coexist; by default
// the builder uses the process-global instance so that column ids are
// globally unique and plans can be composed without reconciliation. Tests
// may construct their own instance for hermeticity.
class IdGenerator {
public:
    ColumnId next() {
        return ColumnId{_counter.fetch_add(1, std::memory_order_relaxed)};
    }

    static IdGenerator& default_instance();

private:
    std::atomic<gamax::size_type> _counter {0};
};

} // namespace gamax::ir

namespace std {
template <>
struct hash<gamax::ir::ColumnId> {
    size_t operator()(gamax::ir::ColumnId id) const noexcept {
        return std::hash<gamax::size_type>{}(id.value);
    }
};
} // namespace std
