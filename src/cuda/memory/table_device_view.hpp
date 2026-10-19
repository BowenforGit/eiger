// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Legacy include path retained while call sites migrate. Device table metadata
// is now the trivially-copyable non-owning view declared in table.hpp, with
// ownership held separately by table_device_view_owner. Owning column objects
// and rmm::device_buffer instances are never copied into device memory.
#include "memory/table.hpp"
