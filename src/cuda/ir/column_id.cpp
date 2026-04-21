#include "ir/column_id.hpp"

namespace gamax::ir {

IdGenerator& IdGenerator::default_instance() {
    static IdGenerator instance;
    return instance;
}

} // namespace gamax::ir
