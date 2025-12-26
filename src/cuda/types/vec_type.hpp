#include <cuda_runtime_api.h>
#include <cstdint>

namespace gamax {

// Primary template (undefined for unsupported combinations)
template <typename T, int N>
struct vec_type;

// Specializations for bool
template <> struct vec_type<bool, 1> { using type = uchar1; };
template <> struct vec_type<bool, 2> { using type = uchar2; };
template <> struct vec_type<bool, 3> { using type = uchar3; };
template <> struct vec_type<bool, 4> { using type = uchar4; };

// Specializations for char
template <> struct vec_type<char, 1> { using type = char1; };
template <> struct vec_type<char, 2> { using type = char2; };
template <> struct vec_type<char, 3> { using type = char3; };
template <> struct vec_type<char, 4> { using type = char4; };

// Specializations for signed char (int8_t)
template <> struct vec_type<signed char, 1> { using type = char1; };
template <> struct vec_type<signed char, 2> { using type = char2; };
template <> struct vec_type<signed char, 3> { using type = char3; };
template <> struct vec_type<signed char, 4> { using type = char4; };

// Specializations for unsigned char
template <> struct vec_type<unsigned char, 1> { using type = uchar1; };
template <> struct vec_type<unsigned char, 2> { using type = uchar2; };
template <> struct vec_type<unsigned char, 3> { using type = uchar3; };
template <> struct vec_type<unsigned char, 4> { using type = uchar4; };

// Specializations for short
template <> struct vec_type<short, 1> { using type = short1; };
template <> struct vec_type<short, 2> { using type = short2; };
template <> struct vec_type<short, 3> { using type = short3; };
template <> struct vec_type<short, 4> { using type = short4; };

// Specializations for unsigned short
template <> struct vec_type<unsigned short, 1> { using type = ushort1; };
template <> struct vec_type<unsigned short, 2> { using type = ushort2; };
template <> struct vec_type<unsigned short, 3> { using type = ushort3; };
template <> struct vec_type<unsigned short, 4> { using type = ushort4; };

// Specializations for int
template <> struct vec_type<int, 1> { using type = int1; };
template <> struct vec_type<int, 2> { using type = int2; };
template <> struct vec_type<int, 3> { using type = int3; };
template <> struct vec_type<int, 4> { using type = int4; };

// Specializations for unsigned int
template <> struct vec_type<unsigned int, 1> { using type = uint1; };
template <> struct vec_type<unsigned int, 2> { using type = uint2; };
template <> struct vec_type<unsigned int, 3> { using type = uint3; };
template <> struct vec_type<unsigned int, 4> { using type = uint4; };

// Specializations for long
template <> struct vec_type<long, 1> { using type = long1; };
template <> struct vec_type<long, 2> { using type = long2; };
template <> struct vec_type<long, 3> { using type = long3; };
template <> struct vec_type<long, 4> { using type = long4; };

// Specializations for unsigned long
template <> struct vec_type<unsigned long, 1> { using type = ulong1; };
template <> struct vec_type<unsigned long, 2> { using type = ulong2; };
template <> struct vec_type<unsigned long, 3> { using type = ulong3; };
template <> struct vec_type<unsigned long, 4> { using type = ulong4; };

// Specializations for long long
template <> struct vec_type<long long, 1> { using type = longlong1; };
template <> struct vec_type<long long, 2> { using type = longlong2; };
template <> struct vec_type<long long, 3> { using type = longlong3; };
template <> struct vec_type<long long, 4> { using type = longlong4; };

// Specializations for unsigned long long
template <> struct vec_type<unsigned long long, 1> { using type = ulonglong1; };
template <> struct vec_type<unsigned long long, 2> { using type = ulonglong2; };
template <> struct vec_type<unsigned long long, 3> { using type = ulonglong3; };
template <> struct vec_type<unsigned long long, 4> { using type = ulonglong4; };

// Specializations for float
template <> struct vec_type<float, 1> { using type = float1; };
template <> struct vec_type<float, 2> { using type = float2; };
template <> struct vec_type<float, 3> { using type = float3; };
template <> struct vec_type<float, 4> { using type = float4; };

// Specializations for double
template <> struct vec_type<double, 1> { using type = double1; };
template <> struct vec_type<double, 2> { using type = double2; };
template <> struct vec_type<double, 3> { using type = double3; };
template <> struct vec_type<double, 4> { using type = double4; };

// Helper type alias for convenience (C++14 and later)
template <typename T, int N>
using vec_type_t = typename vec_type<T, N>::type;

// Meta-programming helper that returns the widest integer type (scalar or vector)
// that can be accessed when alignment is exactly kAlignAt bytes. For unsupported
// alignments, falls back to char.
template <int kAlignAt>
struct widest_int_for_alignment {
	static_assert(kAlignAt > 0, "Alignment must be positive");
	using type = char;
};

template <> struct widest_int_for_alignment<2>  { using type = std::int16_t; };
template <> struct widest_int_for_alignment<4>  { using type = std::int32_t; };
template <> struct widest_int_for_alignment<8>  { using type = std::int64_t; };
template <> struct widest_int_for_alignment<16> { using type = vec_type_t<long long, 2>; };
template <> struct widest_int_for_alignment<32> { using type = vec_type_t<long long, 4>; };

template <int kAlignAt>
using widest_int_for_alignment_t = typename widest_int_for_alignment<kAlignAt>::type;

}