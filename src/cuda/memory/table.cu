#include <memory>

#include "table.hpp"

namespace gamax {
// Move the contents of a vector `unique_ptr<column>`
table::table(std::vector<std::unique_ptr<column>>&& columns) : _columns{std::move(columns)}
{
  if (num_columns() > 0) {
    for (auto const& c : _columns) {
        if(!c) {
            throw std::runtime_error("Null column");
        }
        if(c->size() != _columns.front()->size()) {

            throw std::runtime_error("Column size mismatch: "+std::to_string(c->size())+" vs "+std::to_string(_columns.front()->size()));
        }
    }
  }
}

std::unique_ptr<table> make_table(std::vector<std::unique_ptr<column>>&& columns) {
    return std::make_unique<table>(std::move(columns));
}
}