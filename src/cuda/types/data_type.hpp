#pragma once

#include <limits>
#include <cstdint>

namespace gamax {
using size_type = int32_t;
constexpr size_type MAXIMUM_SIZE = std::numeric_limits<size_type>::max();

enum class type_id {
    EMPTY,                   ///< Always null with no underlying data
    // INT8,                    ///< 1 byte signed integer
    INT16,                   ///< 2 byte signed integer
    INT32,                   ///< 4 byte signed integer
    INT64,                   ///< 8 byte signed integer
    // UINT8,                   ///< 1 byte unsigned integer
    // UINT16,                  ///< 2 byte unsigned integer
    // UINT32,                  ///< 4 byte unsigned integer
    // UINT64,                  ///< 8 byte unsigned integer
    // FLOAT32,                 ///< 4 byte floating point
    // FLOAT64,                 ///< 8 byte floating point
    // BOOL8,                   ///< Boolean using one byte per value, 0 == false, else true
    // `NUM_TYPE_IDS` must be last!
    NUM_TYPE_IDS  ///< Total number of type ids
};

class data_type {
public:
    data_type()                 = default;
    ~data_type()                = default;
    data_type(data_type const&) = default;  ///< Copy constructor
    data_type(data_type&&)      = default;  ///< Move constructor

    data_type& operator=(data_type const&) = default;
    data_type& operator=(data_type&&) = default;
    explicit constexpr data_type(type_id id) : _id{id} {}

    constexpr type_id id() const noexcept { return _id; }

private:
    type_id _id {type_id::EMPTY};
};

constexpr bool operator==(data_type const& lhs, data_type const& rhs)
{
  // use std::tie in the future, breaks JITIFY currently
  return lhs.id() == rhs.id();
}

inline bool operator!=(data_type const& lhs, data_type const& rhs) { return !(lhs == rhs); }

enum class sort_order {
    ASCENDING,  ///< Sort in ascending order
    DESCENDING, ///< Sort in descending order
};
} // namespace gamax