// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Mappings between gamax type ids and the NCCL / MPI datatype handles.
//
// These are intentionally plain inline functions (not constexpr): in some MPI
// implementations the MPI_Datatype handles are addresses of extern objects and
// are not usable in a constant-expression context.

#include <stdexcept>
#include <type_traits>

#include <mpi.h>
#include <nccl.h>

#include "types/data_type.hpp"

namespace alps::convert {

/// MPI datatype matching gamax::size_type (used for count exchange).
inline MPI_Datatype mpi_size_type() {
    if constexpr (std::is_same_v<gamax::size_type, int64_t>) {
        return MPI_INT64_T;
    } else {
        return MPI_INT32_T;
    }
}

inline ncclDataType_t to_nccl_type(gamax::type_id id) {
    switch (id) {
        case gamax::type_id::INT8:
        case gamax::type_id::BOOL8:   return ncclInt8;
        case gamax::type_id::UINT8:   return ncclUint8;
        case gamax::type_id::INT32:   return ncclInt32;
        case gamax::type_id::UINT32:  return ncclUint32;
        case gamax::type_id::INT64:   return ncclInt64;
        case gamax::type_id::UINT64:  return ncclUint64;
        case gamax::type_id::FLOAT32: return ncclFloat32;
        case gamax::type_id::FLOAT64: return ncclFloat64;
        default:
            throw std::runtime_error("alps::convert: gamax type has no NCCL datatype");
    }
}

inline ncclDataType_t to_nccl_type(gamax::data_type type) {
    return to_nccl_type(type.id());
}

} // namespace alps::convert
