#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "data_type.hpp"
#include "utils/assert.cuh"

namespace gamax {
template <typename T>
inline constexpr type_id type_to_id()
{
  return type_id::EMPTY;
};

template <gamax::type_id t>
struct id_to_name_impl {
  static constexpr const char* name = "EMPTY";
};

template <gamax::type_id t>
struct id_to_type_impl {
  using type = void;
};

#ifndef GAMAX_TYPE_MAPPING
#define GAMAX_TYPE_MAPPING(Type, Id)                       \
  template <>                                              \
  constexpr inline type_id type_to_id<Type>()              \
  {                                                        \
    return Id;                                             \
  }                                                        \
  template <>                                              \
  struct id_to_type_impl<Id> {                             \
    using type = Type;                                     \
  };                                                       \
  template <>                                              \
  struct id_to_name_impl<Id> {                             \
    static constexpr const char* name = #Id;               \
  };
#endif

// GAMAX_TYPE_MAPPING(int8_t, type_id::INT8)
GAMAX_TYPE_MAPPING(int16_t, type_id::INT16)
GAMAX_TYPE_MAPPING(int32_t, type_id::INT32)
GAMAX_TYPE_MAPPING(int64_t, type_id::INT64)
// GAMAX_TYPE_MAPPING(uint8_t, type_id::UINT8)
// GAMAX_TYPE_MAPPING(uint16_t, type_id::UINT16)
// GAMAX_TYPE_MAPPING(uint32_t, type_id::UINT32)
// GAMAX_TYPE_MAPPING(uint64_t, type_id::UINT64)
// GAMAX_TYPE_MAPPING(float, type_id::FLOAT32)
// GAMAX_TYPE_MAPPING(double, type_id::FLOAT64)
// GAMAX_TYPE_MAPPING(bool, type_id::BOOL8)

#define size_type_id type_to_id<gamax::size_type>()

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
template <template <gamax::type_id> typename IdTypeMap = id_to_type_impl,
          typename Functor,
          typename... Ts>
__host__ __device__ __forceinline__ constexpr decltype(auto) type_dispatcher(gamax::data_type dtype,
                                                                          Functor f,
                                                                          Ts&&... args)
{
  switch (dtype.id()) {
    // case type_id::INT8:
    //   return f.template operator()<typename IdTypeMap<type_id::INT8>::type>(
    //     std::forward<Ts>(args)...);
    case type_id::INT16:
      return f.template operator()<typename IdTypeMap<type_id::INT16>::type>(
        std::forward<Ts>(args)...);
    case type_id::INT32:
      return f.template operator()<typename IdTypeMap<type_id::INT32>::type>(
        std::forward<Ts>(args)...);
    case type_id::INT64:
      return f.template operator()<typename IdTypeMap<type_id::INT64>::type>(
        std::forward<Ts>(args)...);
    // case type_id::UINT8:
    //   return f.template operator()<typename IdTypeMap<type_id::UINT8>::type>(
    //     std::forward<Ts>(args)...);
    // case type_id::UINT16:
    //   return f.template operator()<typename IdTypeMap<type_id::UINT16>::type>(
    //     std::forward<Ts>(args)...);
    // case type_id::UINT32:
    //   return f.template operator()<typename IdTypeMap<type_id::UINT32>::type>(
    //     std::forward<Ts>(args)...);
    // case type_id::UINT64:
    //   return f.template operator()<typename IdTypeMap<type_id::UINT64>::type>(
    //     std::forward<Ts>(args)...);
    // case type_id::FLOAT32:
    //   return f.template operator()<typename IdTypeMap<type_id::FLOAT32>::type>(
    //     std::forward<Ts>(args)...);
    // case type_id::FLOAT64:
    //   return f.template operator()<typename IdTypeMap<type_id::FLOAT64>::type>(
    //     std::forward<Ts>(args)...);
    // case type_id::BOOL8:
    //   return f.template operator()<typename IdTypeMap<type_id::BOOL8>::type>(
    //     std::forward<Ts>(args)...);
    default: {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
      throw std::runtime_error("Invalid type_id.");
#else
      GAMAX_UNREACHABLE("Invalid type_id.");
#endif
    }
  }
}

namespace {
template <typename T1>
struct double_type_dispatcher_second_type {
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
    template <typename T2, typename F, typename... Ts>
    __host__ __device__ __forceinline__ decltype(auto) operator()(F&& f, Ts&&... args) const
    {
        return f.template operator()<T1, T2>(std::forward<Ts>(args)...);
    }
};

template <template <gamax::type_id> typename IdTypeMap>
struct double_type_dispatcher_first_type {
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
    template <typename T1, typename F, typename... Ts>
    __host__ __device__ __forceinline__ decltype(auto) operator()(gamax::data_type type2,
                                                                F&& f,
                                                                Ts&&... args) const
    {
        return type_dispatcher<IdTypeMap>(type2,
                                        double_type_dispatcher_second_type<T1>{},
                                        std::forward<F>(f),
                                        std::forward<Ts>(args)...);
    }
};
} // unnamed namespace

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
template <template <gamax::type_id> typename IdTypeMap = id_to_type_impl, typename F, typename... Ts>
__host__ __device__ __forceinline__ constexpr decltype(auto) double_type_dispatcher(
  gamax::data_type type1, gamax::data_type type2, F&& f, Ts&&... args)
{
  return type_dispatcher<IdTypeMap>(type1,
                                    double_type_dispatcher_first_type<IdTypeMap>{},
                                    type2,
                                    std::forward<F>(f),
                                    std::forward<Ts>(args)...);
}

struct size_of_helper {
  template <typename T>
  constexpr __host__ __device__ int operator()() const noexcept
  {
    return sizeof(T);
  }
};

struct to_string_helper {
  template <typename T>
  std::string operator()() const
  {
    return std::string(id_to_name_impl<type_to_id<T>()>::name);
  }
};

inline __host__ __device__ std::size_t size_of(data_type element_type)
{
  return type_dispatcher(element_type, size_of_helper{});
}

inline std::string to_string(data_type element_type)
{
  return type_dispatcher(element_type, to_string_helper{});
}
} // namespace gamax