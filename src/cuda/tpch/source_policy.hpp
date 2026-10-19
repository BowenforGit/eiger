// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace gamax::tpch {

enum class SourceReadPolicy {
    Sharded,
    Replicated,
};

inline const char* source_read_policy_name(SourceReadPolicy policy) {
    switch (policy) {
        case SourceReadPolicy::Sharded: return "sharded";
        case SourceReadPolicy::Replicated: return "replicated";
    }
    return "unknown";
}

// Immutable once shared with a reader/planner. Build the policy map first, then
// pass the same shared object to both components so physical placement cannot
// diverge from what was read from disk.
class SourceReadPolicies {
public:
    SourceReadPolicies& set(std::string table, SourceReadPolicy policy) {
        policies_[std::move(table)] = policy;
        return *this;
    }

    SourceReadPolicy for_table(const std::string& table) const noexcept {
        const auto found = policies_.find(table);
        return found == policies_.end()
            ? SourceReadPolicy::Sharded
            : found->second;
    }

private:
    std::unordered_map<std::string, SourceReadPolicy> policies_;
};

using SourceReadPoliciesPtr = std::shared_ptr<const SourceReadPolicies>;

inline SourceReadPoliciesPtr make_sharded_source_read_policies() {
    return std::make_shared<const SourceReadPolicies>();
}

inline SourceReadPoliciesPtr make_distributed_tpch_source_read_policies() {
    SourceReadPolicies policies;
    policies.set("region", SourceReadPolicy::Replicated)
            .set("nation", SourceReadPolicy::Replicated);
    return std::make_shared<const SourceReadPolicies>(std::move(policies));
}

} // namespace gamax::tpch
