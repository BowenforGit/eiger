# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# FindNCCL.cmake — locate the NVIDIA Collective Communications Library.
#
# Exposes the imported target NCCL::nccl and the variables
#   NCCL_FOUND, NCCL_INCLUDE_DIR, NCCL_LIBRARY, NCCL_VERSION.
#
# Honors the NCCL_ROOT / NCCL_HOME hints (and the standard CMake search
# paths). On this system NCCL is installed under /usr (nccl.h in
# /usr/include, libnccl.so in /usr/lib/x86_64-linux-gnu).

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS ${NCCL_ROOT} ${NCCL_HOME} ENV NCCL_ROOT ENV NCCL_HOME
  PATH_SUFFIXES include)

find_library(NCCL_LIBRARY
  NAMES nccl
  HINTS ${NCCL_ROOT} ${NCCL_HOME} ENV NCCL_ROOT ENV NCCL_HOME
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

# Parse the version out of nccl.h if available.
if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
  file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_major REGEX "^#define NCCL_MAJOR +[0-9]+")
  file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_minor REGEX "^#define NCCL_MINOR +[0-9]+")
  file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_patch REGEX "^#define NCCL_PATCH +[0-9]+")
  string(REGEX REPLACE ".*NCCL_MAJOR +([0-9]+).*" "\\1" _nccl_major "${_nccl_major}")
  string(REGEX REPLACE ".*NCCL_MINOR +([0-9]+).*" "\\1" _nccl_minor "${_nccl_minor}")
  string(REGEX REPLACE ".*NCCL_PATCH +([0-9]+).*" "\\1" _nccl_patch "${_nccl_patch}")
  if(_nccl_major MATCHES "^[0-9]+$")
    set(NCCL_VERSION "${_nccl_major}.${_nccl_minor}.${_nccl_patch}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
  REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
  VERSION_VAR NCCL_VERSION)

if(NCCL_FOUND AND NOT TARGET NCCL::nccl)
  add_library(NCCL::nccl UNKNOWN IMPORTED)
  set_target_properties(NCCL::nccl PROPERTIES
    IMPORTED_LOCATION "${NCCL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
