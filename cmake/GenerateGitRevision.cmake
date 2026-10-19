# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

if(NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "OUTPUT_FILE is required")
endif()

set(_revision "unknown")
if(DEFINED GIT_EXECUTABLE AND EXISTS "${GIT_EXECUTABLE}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE _revision_result
        OUTPUT_VARIABLE _revision_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_revision_result EQUAL 0 AND NOT _revision_output STREQUAL "")
        set(_revision "${_revision_output}")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" status --porcelain
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE _status_result
            OUTPUT_VARIABLE _status_output
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_status_result EQUAL 0 AND NOT _status_output STREQUAL "")
            string(APPEND _revision "-dirty")
        endif()
    endif()
endif()

set(_contents
    "#pragma once\n#define GAMAX_GIT_COMMIT \"${_revision}\"\n")
file(CONFIGURE OUTPUT "${OUTPUT_FILE}" CONTENT "${_contents}" @ONLY)
